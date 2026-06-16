#!/usr/bin/env python3
"""
배치 PDF 파싱 스크립트.
- 청크 단위 parquet 저장 (중간 손실 방지)
- 손상 PDF → BrokenProcessPool 자동 복구 (skip + 재시작)
- 체크포인트 파일로 재시작 시 이어받기
- --backend: 파서 선택 (pymupdf4llm / docling / mineru)
- --replace-source: 특정 source_type의 기존 결과를 새 파서로 교체

실행:
    # 기본 (pymupdf4llm으로 신규 파싱)
    python batch_parse_pdfs.py --workers 4

    # 새 파서로 전체 교체
    python batch_parse_pdfs.py --backend docling --replace-source arxiv_pdf_pymupdf4llm --workers 4
"""
from __future__ import annotations
import argparse, glob, logging, os, time, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
PDF_DIR   = BASE_DIR / "data" / "arxiv" / "pdfs"
DATA_DIR  = BASE_DIR / "data"
CKPT_DIR  = BASE_DIR / "data" / "checkpoints"

BACKENDS = ["pymupdf4llm", "docling", "mineru"]


# ─── 파서별 parse_one 함수 ────────────────────────────────────────────────────

def parse_one_pymupdf4llm(pdf_path: str) -> tuple[str, str | None, str | None]:
    import pymupdf4llm as _pml
    arxiv_id = Path(pdf_path).stem.replace("_", "/")
    try:
        md = _pml.to_markdown(pdf_path)
        return (arxiv_id, md, None) if md and len(md.strip()) > 50 else (arxiv_id, None, "빈 텍스트")
    except Exception as e:
        return arxiv_id, None, str(e)


def parse_one_docling(pdf_path: str) -> tuple[str, str | None, str | None]:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    arxiv_id = Path(pdf_path).stem.replace("_", "/")
    try:
        opts = PdfPipelineOptions()
        opts.do_formula_enrichment = True
        opts.do_table_structure = True
        converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=opts)}
        )
        result = converter.convert(pdf_path)
        md = result.document.export_to_markdown()
        return (arxiv_id, md, None) if md and len(md.strip()) > 50 else (arxiv_id, None, "빈 텍스트")
    except Exception as e:
        return arxiv_id, None, str(e)


def parse_one_mineru(pdf_path: str) -> tuple[str, str | None, str | None]:
    arxiv_id = Path(pdf_path).stem.replace("_", "/")
    try:
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
        import tempfile, os as _os
        with tempfile.TemporaryDirectory() as tmp:
            writer = FileBasedDataWriter(tmp)
            reader = FileBasedDataReader("")
            ds = PymuDocDataset(reader.read(pdf_path))
            method = ds.classify()
            if method == SupportedPdfParseMethod.OCR:
                pipe = ds.apply(doc_analyze, ocr=True)
            else:
                pipe = ds.apply(doc_analyze, ocr=False)
            pipe.pipe_mk_markdown(writer, drop_mode="none")
            md_file = [f for f in _os.listdir(tmp) if f.endswith(".md")]
            if not md_file:
                return arxiv_id, None, "MinerU: 출력 파일 없음"
            with open(_os.path.join(tmp, md_file[0])) as f:
                md = f.read()
        return (arxiv_id, md, None) if md and len(md.strip()) > 50 else (arxiv_id, None, "빈 텍스트")
    except Exception as e:
        return arxiv_id, None, str(e)


PARSE_FUNCS = {
    "pymupdf4llm": parse_one_pymupdf4llm,
    "docling":     parse_one_docling,
    "mineru":      parse_one_mineru,
}

SOURCE_TYPE_MAP = {
    "pymupdf4llm": "arxiv_pdf_pymupdf4llm",
    "docling":     "arxiv_pdf_docling",
    "mineru":      "arxiv_pdf_mineru",
}


# ─── 유틸리티 ─────────────────────────────────────────────────────────────────

def ckpt_path(backend: str) -> Path:
    return CKPT_DIR / f"pdf_parse_{backend}.json"


def load_checkpoint(backend: str) -> set[str]:
    p = ckpt_path(backend)
    if p.exists():
        with open(p) as f:
            return set(json.load(f).get("done", []))
    return set()


def save_checkpoint(backend: str, done: set[str]) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    p = ckpt_path(backend)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"done": list(done)}, f)
    os.replace(tmp, str(p))


def find_replace_target_pdfs(replace_source: str) -> list[str]:
    """replace_source에 해당하는 parquet 레코드의 PDF 경로 반환."""
    targets = []
    for f in sorted(glob.glob(str(DATA_DIR / "arxiv" / "**" / "*.parquet"), recursive=True)):
        try:
            df = pd.read_parquet(f, columns=["arxiv_id", "full_text_source_type", "pdf_path"])
            mask = df["full_text_source_type"] == replace_source
            for _, row in df[mask].iterrows():
                arxiv_id = row["arxiv_id"]
                if not arxiv_id:
                    continue
                # pdf_path 컬럼 우선, 없으면 기본 경로 추정
                pdf_path = row.get("pdf_path")
                if not pdf_path or not Path(pdf_path).exists():
                    safe_id = str(arxiv_id).replace("/", "_")
                    pdf_path = str(PDF_DIR / f"{safe_id}.pdf")
                if Path(pdf_path).exists():
                    targets.append(pdf_path)
        except Exception:
            continue
    return targets


def save_chunk(results: dict[str, str], backend: str) -> int:
    """parquet에 청크 결과 저장. 업데이트 건수 반환."""
    if not results:
        return 0
    source_type = SOURCE_TYPE_MAP[backend]
    parquet_files = sorted(glob.glob(str(DATA_DIR / "arxiv" / "**" / "*.parquet"), recursive=True))
    updated = 0
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        mask = df["arxiv_id"].isin(results)
        if not mask.any():
            continue
        # surrogate 문자 제거 (UnicodeEncodeError 방지)
        def _clean_surrogates(text):
            if isinstance(text, str):
                # surrogate 범위 (U+D800~U+DFFF) 직접 제거
                import re
                return re.sub(r'[\ud800-\udfff]', '\ufffd', text)
            return text
        df.loc[mask, "full_text"]             = df.loc[mask, "arxiv_id"].map(results).apply(_clean_surrogates)
        df.loc[mask, "has_full_text"]          = True
        df.loc[mask, "full_text_format"]       = "markdown"
        df.loc[mask, "full_text_source_type"]  = source_type
        df.loc[mask, "full_text_status"]       = "pdf_parsed"
        tmp = f + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, f)
        updated += int(mask.sum())
    return updated


def process_chunk(chunk: list[str], workers: int, backend: str) -> tuple[dict, list, set]:
    """청크 처리. BrokenProcessPool 시 문제 파일 skip 후 재시작."""
    parse_fn = PARSE_FUNCS[backend]
    results  = {}
    errors   = []
    skip_ids = set()
    remaining = list(chunk)

    while remaining:
        batch_results: dict[str, str] = {}
        batch_errors:  list           = []

        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(parse_fn, p): p for p in remaining}
                for fut in as_completed(futures):
                    pdf_path = futures[fut]
                    try:
                        arxiv_id, text, err = fut.result()
                        if text:
                            batch_results[arxiv_id] = text
                        else:
                            batch_errors.append((arxiv_id, err))
                    except Exception as e:
                        batch_errors.append((Path(pdf_path).stem.replace("_", "/"), str(e)))
            results.update(batch_results)
            errors.extend(batch_errors)
            break

        except Exception as e:
            log.warning(f"ProcessPool 크래시: {e}")
            results.update(batch_results)
            errors.extend(batch_errors)

            done_in_batch = set(batch_results.keys()) | {err[0] for err in batch_errors}
            remaining = [p for p in remaining
                         if Path(p).stem.replace("_", "/") not in done_in_batch]

            if remaining:
                bad    = remaining[0]
                bad_id = Path(bad).stem.replace("_", "/")
                log.warning(f"  → skip 손상 PDF: {Path(bad).name}")
                skip_ids.add(bad_id)
                errors.append((bad_id, "BrokenProcessPool (손상 PDF skip)"))
                remaining = remaining[1:]
            else:
                break

    return results, errors, skip_ids


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="배치 PDF 파싱")
    parser.add_argument("--backend", choices=BACKENDS, default="pymupdf4llm",
                        help="사용할 파서 (기본: pymupdf4llm)")
    parser.add_argument("--pdf-dir", metavar="PATH", default=None,
                        help="PDF 폴더 경로 (기본: data/arxiv/pdfs/)")
    parser.add_argument("--replace-source", metavar="SOURCE_TYPE", default=None,
                        help="교체 모드: 해당 source_type 레코드를 새 파서로 재파싱. "
                             "예: arxiv_pdf_pymupdf4llm")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk",   type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backend = args.backend

    # ── 대상 PDF 목록 수집 ──
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else PDF_DIR

    if args.replace_source:
        log.info(f"[교체 모드] source_type='{args.replace_source}' → backend='{backend}'")
        all_pdfs = find_replace_target_pdfs(args.replace_source)
        log.info(f"  교체 대상: {len(all_pdfs):,}개")
        # 교체 모드는 전용 checkpoint 키 사용
        ckpt_key = f"{backend}_replace_{args.replace_source}"
    else:
        all_pdfs = sorted(glob.glob(str(pdf_dir / "*.pdf")))
        ckpt_key = backend

    # 이미 처리된 항목 제외
    done_ids = load_checkpoint(ckpt_key)
    pdfs = [p for p in all_pdfs
            if Path(p).stem.replace("_", "/") not in done_ids]

    log.info(f"파서: {backend} | 전체: {len(all_pdfs):,} | 완료: {len(done_ids):,} | 남은: {len(pdfs):,}")
    log.info(f"설정: workers={args.workers}, chunk={args.chunk}")

    if args.dry_run:
        log.info("[DRY-RUN] 종료")
        return

    t0 = time.time()
    total_parsed = len(done_ids)
    total_errors = 0

    chunks = [pdfs[i:i+args.chunk] for i in range(0, len(pdfs), args.chunk)]
    for ci, chunk in enumerate(chunks, 1):
        log.info(f"[청크 {ci}/{len(chunks)}] {len(chunk)}개 처리 중...")
        results, errors, skip_ids = process_chunk(chunk, args.workers, backend)

        updated = save_chunk(results, backend)

        done_ids.update(results.keys())
        done_ids.update(skip_ids)
        done_ids.update(e[0] for e in errors)
        save_checkpoint(ckpt_key, done_ids)

        total_parsed += len(results)
        total_errors += len(errors)

        elapsed = time.time() - t0
        rate = total_parsed / elapsed * 60 if elapsed > 0 else 0
        remaining_count = max(0, len(pdfs) - ci * args.chunk)
        eta_h = remaining_count / rate / 60 if rate > 0 else 0
        log.info(f"  저장: {updated}건 | 누적: {total_parsed:,}건 성공, {total_errors}건 실패 "
                 f"| 속도: {rate:.0f}/분 | ETA: {eta_h:.1f}시간")

    log.info(f"\n완료: 성공 {total_parsed:,}건 / 실패 {total_errors}건")
    log.info(f"소요: {(time.time()-t0)/3600:.1f}시간")


if __name__ == "__main__":
    main()
