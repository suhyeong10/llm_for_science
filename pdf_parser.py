"""PDF 배치 파싱 — 저장된 PDF 파일을 고품질 파서로 텍스트 추출.

지원 백엔드:
    pymupdf  : PyMuPDF (베이스라인, 빠름, 설치됨)
    docling  : Docling IBM (CPU 최강, 3.1초/페이지)
    mineru   : MinerU (GPU 최강, 수식 탁월)

사용 예:
    python main.py parse-pdf                          # docling (기본)
    python main.py parse-pdf --backend pymupdf        # 베이스라인 비교용
    python main.py parse-pdf --backend mineru --workers 4
    python main.py parse-pdf --compare               # pymupdf vs 현재 백엔드 비교
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd

import config
from latex_cleaner import clean_pdf_text, count_tokens_approx
from utils import detect_language, load_all_parquet, save_df_to_parquet, setup_logger

logger = setup_logger("pdf_parser", config.LOGS_DIR)


# ── 백엔드별 파서 ──

def parse_with_pymupdf(pdf_path: str) -> str | None:
    """PyMuPDF (fitz) — 베이스라인. 빠르고 가볍지만 레이아웃 취약."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        result = "\n".join(p for p in pages if p.strip())
        return clean_pdf_text(result) if len(result) > 200 else None
    except Exception as e:
        logger.debug(f"PyMuPDF 실패 ({pdf_path}): {e}")
        return None


def parse_with_docling(pdf_path: str) -> str | None:
    """Docling (IBM) — CPU 최강, 2컬럼 레이아웃 우수, 수식 처리 양호.
    설치: pip install docling
    """
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        text = result.document.export_to_markdown()
        return text if len(text) > 200 else None
    except ImportError:
        logger.error("Docling 미설치. 'pip install docling' 실행 필요")
        return None
    except Exception as e:
        logger.debug(f"Docling 실패 ({pdf_path}): {e}")
        return None


def parse_with_mineru(pdf_path: str) -> str | None:
    """MinerU — GPU 최강, 수식 탁월, 학술논문 특화.
    설치: pip install mineru
    GPU 없으면 CPU 모드로 동작 (느림).
    """
    try:
        from mineru.api import pdf_parse
        result = pdf_parse(pdf_path, output_format="markdown")
        return result if len(result) > 200 else None
    except ImportError:
        logger.error("MinerU 미설치. 'pip install mineru' 실행 필요")
        return None
    except Exception as e:
        logger.debug(f"MinerU 실패 ({pdf_path}): {e}")
        return None


BACKENDS = {
    "pymupdf": parse_with_pymupdf,
    "docling": parse_with_docling,
    "mineru": parse_with_mineru,
}


# ── 배치 파서 ──

def parse_pdfs_batch(
    backend: str = "docling",
    max_items: int | None = None,
    workers: int = 2,
    overwrite_pymupdf: bool = False,
) -> dict:
    """저장된 PDF 파일을 배치로 파싱하여 clean_text 채움.

    Args:
        backend: 파서 종류 (pymupdf / docling / mineru)
        max_items: 처리할 최대 건수
        workers: 병렬 워커 수 (docling CPU 기준 2~4 권장)
        overwrite_pymupdf: True면 arxiv_pdf_pymupdf 결과도 덮어씀
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend는 {list(BACKENDS)} 중 하나여야 합니다")

    parse_fn = BACKENDS[backend]
    source_type_tag = f"arxiv_pdf_{backend}"

    df = load_all_parquet(config.DATA_DIR, source="arxiv")
    if df.empty:
        logger.warning("arXiv 데이터 없음")
        return {"parsed": 0, "failed": 0, "skipped": 0}

    # 대상: pdf_downloaded 상태 (+ overwrite_pymupdf 옵션)
    target_mask = df["full_text_status"] == "pdf_downloaded"
    if overwrite_pymupdf:
        target_mask |= df["full_text_source_type"] == "arxiv_pdf_pymupdf"

    targets = df[target_mask].copy()
    if targets.empty:
        logger.info("파싱할 PDF 없음 (pdf_downloaded 상태인 논문이 없습니다)")
        return {"parsed": 0, "failed": 0, "skipped": 0}

    if max_items:
        targets = targets.head(max_items)

    logger.info(
        f"PDF 배치 파싱 시작: {len(targets)}건, backend={backend}, workers={workers}"
    )

    stats = {"parsed": 0, "failed": 0, "skipped": 0}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _parse_one(idx_row):
        idx, row = idx_row
        pdf_path = row.get("pdf_path")
        if not pdf_path or not os.path.exists(str(pdf_path)):
            return idx, None, "no_file"

        t0 = time.time()
        text = parse_fn(str(pdf_path))
        elapsed = time.time() - t0

        if text and len(text) > 200:
            return idx, text, f"ok:{elapsed:.1f}s"
        return idx, None, "empty"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_parse_one, (idx, row)): idx
            for idx, row in targets.iterrows()
        }
        for i, future in enumerate(as_completed(futures), 1):
            idx, text, status = future.result()
            if text:
                df.at[idx, "has_full_text"] = True
                df.at[idx, "full_text"] = text
                df.at[idx, "full_text_format"] = "pdf_markdown"
                df.at[idx, "full_text_source_type"] = source_type_tag
                df.at[idx, "full_text_status"] = "success"
                df.at[idx, "clean_text"] = text  # docling/mineru 출력이 이미 clean
                df.at[idx, "token_count_approx"] = count_tokens_approx(text)
                df.at[idx, "language"] = detect_language(text[:1000])
                stats["parsed"] += 1
            else:
                stats["failed"] += 1

            if i % 20 == 0:
                logger.info(
                    f"  진행: {i}/{len(targets)}, "
                    f"parsed={stats['parsed']}, failed={stats['failed']}"
                )

    # 저장
    save_df_to_parquet(df, config.DATA_DIR, "arxiv", f"pdf_{backend}")
    logger.info(f"PDF 파싱 완료 ({backend}): {stats}")
    return stats


def compare_pdf_backends(
    sample_size: int = 20,
    backends: list[str] | None = None,
) -> None:
    """두 파서의 결과를 나란히 비교 출력 (품질 평가용)."""
    backends = backends or ["pymupdf", "docling"]

    df = load_all_parquet(config.DATA_DIR, source="arxiv")
    # pdf_downloaded 또는 pymupdf 결과 중 샘플
    candidates = df[
        df["full_text_status"].isin(["pdf_downloaded", "success"]) &
        df["full_text_source_type"].isin(["arxiv_pdf_saved", "arxiv_pdf_pymupdf"])
    ].dropna(subset=["pdf_path"])

    if candidates.empty:
        print("비교할 PDF 없음. 먼저 'python main.py arxiv-latex' 실행 필요")
        return

    sample = candidates.head(sample_size)
    print(f"\n{'='*60}")
    print(f" PDF 파서 비교: {backends}")
    print(f" 샘플 {len(sample)}건")
    print(f"{'='*60}\n")

    for _, row in sample.iterrows():
        pdf_path = str(row["pdf_path"])
        if not os.path.exists(pdf_path):
            continue

        print(f"--- {row['arxiv_id']} ---")
        print(f"제목: {row['title'][:70]}")
        for backend in backends:
            fn = BACKENDS.get(backend)
            if not fn:
                continue
            t0 = time.time()
            text = fn(pdf_path)
            elapsed = time.time() - t0
            tokens = count_tokens_approx(text) if text else 0
            latex_rem = len(re.findall(r"\\[a-zA-Z]+", text or ""))
            print(f"\n[{backend}] {elapsed:.1f}초, {tokens}토큰, LaTeX잔존={latex_rem}")
            print((text or "")[:400])
        print()
