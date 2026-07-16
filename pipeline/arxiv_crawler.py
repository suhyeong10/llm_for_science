"""arXiv 메타데이터 수집 + LaTeX 소스 다운로드."""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import feedparser
import requests

from . import config
from .schema import PaperRecord
from .utils import (
    CheckpointManager,
    GlobalRateLimiter,
    RateLimiter,
    compute_content_hash,
    detect_language,
    generate_paper_id,
    generate_record_id,
    retry,
    save_df_to_parquet,
    save_records_to_parquet,
    setup_logger,
)

logger = setup_logger("arxiv", config.LOGS_DIR)

# arXiv ID 추출 패턴: URL 또는 순수 ID 모두 처리
_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/abs/)?(\d{4}\.\d{4,5}|[a-z-]+/\d{7})")


class ArxivCrawler:
    """arXiv 논문 크롤러."""

    def __init__(self, batch_id: str, resume: bool = False):
        self.batch_id = batch_id
        self.rate_limiter = RateLimiter(config.ARXIV_RATE_LIMIT_SEC)

        ckpt_path = f"{config.DATA_DIR}/checkpoints/arxiv_metadata.json"
        self.ckpt = CheckpointManager(ckpt_path)
        if not resume:
            self.ckpt.reset()

        self._record_counter = self.ckpt.get("record_counter", 1)
        self._buffer: list[PaperRecord] = []

    # ── Phase 1: 메타데이터 수집 ──

    def crawl_metadata(
        self,
        categories: list[str] | None = None,
        year_range: tuple[int, int] | None = None,
        max_results: int | None = None,
    ) -> int:
        """arXiv API로 메타데이터 수집. 수집 건수 반환."""
        cats = categories or config.ARXIV_CATEGORIES
        yr = year_range or config.YEAR_RANGE

        total_fetched = 0
        for cat in cats:
            n = self._crawl_category(cat, yr, max_results)
            total_fetched += n
            if max_results and total_fetched >= max_results:
                break

        self._flush_buffer()
        logger.info(f"메타데이터 수집 완료: 총 {total_fetched}건")
        return total_fetched

    def _crawl_category(
        self, category: str, year_range: tuple[int, int], max_results: int | None
    ) -> int:
        """카테고리를 연도별로 분할 크롤링 (arXiv deep paging 10,000건 제한 회피)."""
        fetched = 0

        # resume: 마지막으로 완료된 연도 확인
        last_completed_year = self.ckpt.get(f"last_year_{category}", year_range[0] - 1)

        for year in range(year_range[0], year_range[1] + 1):
            if max_results and fetched >= max_results:
                break
            if year <= last_completed_year:
                logger.info(f"  {category}/{year}: 이미 수집됨 (skip)")
                continue

            remaining = (max_results - fetched) if max_results else None
            n = self._crawl_year(category, year, remaining)
            fetched += n

            # 연도 완료 체크포인트
            self.ckpt.set(f"last_year_{category}", year)
            self.ckpt.save()

        return fetched

    def _crawl_year(
        self, category: str, year: int, max_results: int | None
    ) -> int:
        """단일 연도의 메타데이터 수집."""
        ckpt_key = f"offset_{category}_{year}"
        start = self.ckpt.get(ckpt_key, 0)
        fetched = 0
        batch_size = min(config.ARXIV_BATCH_SIZE, 200)

        search_query = (
            f"cat:{category} AND "
            f"submittedDate:[{year}01010000 TO {year}12312359]"
        )

        logger.info(f"  {category}/{year} 크롤링 시작 (offset={start})")

        while True:
            if max_results and fetched >= max_results:
                break

            self.rate_limiter.wait()
            entries = self._fetch_page(search_query, start, batch_size)

            if not entries:
                break

            for entry in entries:
                if max_results and fetched >= max_results:
                    break

                record = self._parse_entry(entry)
                if record and record.year == year:
                    record.record_id = generate_record_id(self._record_counter)
                    record.paper_id = generate_paper_id(self._record_counter)
                    record.crawl_batch_id = self.batch_id
                    record.crawl_date = datetime.now(timezone.utc).isoformat()
                    record.content_hash = compute_content_hash(
                        record.title, record.abstract
                    )
                    record.language = detect_language(record.abstract)
                    self._buffer.append(record)
                    self._record_counter += 1
                    fetched += 1

            start += len(entries)

            if len(self._buffer) >= config.PARQUET_ROWS_PER_FILE:
                self._flush_buffer()

            self.ckpt.set(ckpt_key, start)
            self.ckpt.set("record_counter", self._record_counter)
            self.ckpt.save()

        # 연도 완료 시 잔여 버퍼 flush
        self._flush_buffer()
        logger.info(f"  {category}/{year}: {fetched}건 수집 완료")
        return fetched

    @retry(max_retries=7, backoff=2.0, exceptions=(requests.RequestException,))
    def _fetch_page(self, search_query: str, start: int, max_results: int) -> list:
        """arXiv API 한 페이지 호출. 429/500 시 내부 sleep + decorator retry."""
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        resp = requests.get(config.ARXIV_API_BASE, params=params, timeout=90)
        if resp.status_code == 429:
            logger.warning("arXiv 429 rate limit. 60초 대기...")
            time.sleep(60)
            raise requests.RequestException("429 rate limited")
        if resp.status_code == 500:
            logger.warning(f"arXiv 500 error at offset={start}. 30초 대기...")
            time.sleep(30)
            raise requests.RequestException("500 server error")
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        return feed.entries

    def _parse_entry(self, entry) -> PaperRecord | None:
        """feedparser entry → PaperRecord."""
        try:
            # arXiv ID 추출 (정규식 기반으로 안정적 파싱)
            raw_id = entry.get("id", "")
            m = _ARXIV_ID_RE.search(raw_id)
            arxiv_id = m.group(1) if m else raw_id.rstrip("/").rsplit("/", 1)[-1]

            # 버전 태그 제거 (2001.00123v2 → 2001.00123)
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

            # 연도 추출
            published = entry.get("published", "")
            year = int(published[:4]) if len(published) >= 4 else 0

            # 저자
            authors = [a.get("name", "") for a in entry.get("authors", [])]

            # 카테고리
            tags = entry.get("tags", [])
            categories = [t.get("term", "") for t in tags if t.get("term")]

            # 라이선스
            license_url = None
            pdf_url = None
            for link in entry.get("links", []):
                if link.get("rel") == "license":
                    license_url = link.get("href")
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href")

            # DOI
            doi = entry.get("arxiv_doi")

            title = entry.get("title", "").replace("\n", " ").strip()
            # 연속 공백 정리
            title = re.sub(r"\s+", " ", title)
            abstract = entry.get("summary", "").strip()

            return PaperRecord(
                source="arxiv",
                source_paper_id=arxiv_id,
                arxiv_id=arxiv_id,
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                categories=categories,
                license=license_url,
                pdf_url=pdf_url,
                has_full_text=False,
                full_text_status="abstract_only",
                fetch_success=True,
                raw_source_payload=json.dumps(
                    {"id": arxiv_id, "title": title}, ensure_ascii=False
                ),
            )
        except Exception as e:
            logger.warning(f"entry 파싱 실패: {e}")
            return None

    def _flush_buffer(self):
        """버퍼의 레코드를 parquet로 저장."""
        if not self._buffer:
            return
        save_records_to_parquet(
            self._buffer, config.DATA_DIR, "arxiv", self.batch_id,
        )
        logger.info(f"  {len(self._buffer)}건 parquet 저장")
        self._buffer.clear()

    # ── Phase 2: LaTeX 소스 다운로드 (+ PDF 폴백) ──

    def download_latex_sources(
        self,
        max_items: int | None = None,
        resume: bool = True,
        pdf_fallback: bool = True,
        workers: int = 3,
    ) -> dict:
        """메타데이터에서 arxiv_id를 읽어 LaTeX 소스 다운로드.

        LaTeX 실패 시 pdf_fallback=True이면 PDF → 텍스트 추출.
        workers > 1이면 ThreadPoolExecutor로 병렬 처리.
        """
        from .utils import load_all_parquet

        ckpt_path = f"{config.DATA_DIR}/checkpoints/arxiv_latex.json"
        latex_ckpt = CheckpointManager(ckpt_path)
        ckpt_lock = threading.Lock()

        df = load_all_parquet(config.DATA_DIR, source="arxiv")
        if df.empty:
            logger.warning("arXiv 메타데이터가 없습니다. 먼저 crawl_metadata를 실행하세요.")
            return {"success": 0, "failed": 0, "skipped": 0, "pdf_fallback": 0}

        stats: dict[str, int] = {"success": 0, "failed": 0, "skipped": 0, "pdf_fallback": 0}
        stats_lock = threading.Lock()
        df_lock = threading.Lock()

        # 전역 Rate Limiter: 모든 워커가 공유 → burst 없이 균등 분산
        # interval = ARXIV_RATE_LIMIT_SEC / 1 (전체 처리량 = 1/interval req/sec)
        global_rl = GlobalRateLimiter(config.ARXIV_RATE_LIMIT_SEC)

        # abstract_only 논문 우선 선택 (미다운로드 논문)
        if "full_text_status" in df.columns:
            abstract_only_ids = df[
                df["full_text_status"] == "abstract_only"
            ]["arxiv_id"].dropna().unique().tolist()
            other_ids = df[
                df["full_text_status"] != "abstract_only"
            ]["arxiv_id"].dropna().unique().tolist()
            # abstract_only 먼저, 나머지 뒤에
            arxiv_ids = abstract_only_ids + other_ids
        else:
            arxiv_ids = df["arxiv_id"].dropna().unique().tolist()

        # 이미 처리된 ID 필터 (resume) — max_items 적용 전에 필터
        if resume:
            pending_ids = [
                aid for aid in arxiv_ids
                if not latex_ckpt.is_processed(aid)
            ]
        else:
            pending_ids = arxiv_ids

        with stats_lock:
            stats["skipped"] = len(arxiv_ids) - len(pending_ids)

        if max_items:
            skipped_before_cut = stats["skipped"]
            pending_ids = pending_ids[:max_items]

        logger.info(
            f"LaTeX 다운로드 시작: 대상={len(pending_ids)}건 "
            f"(skip={stats['skipped']}건, workers={workers}, pdf_fallback={pdf_fallback})"
        )

        def _process_one(arxiv_id: str) -> tuple[str, str | None, str]:
            """단일 arxiv_id 처리. (arxiv_id, content_or_path, source_type) 반환.

            LaTeX: 즉시 텍스트 추출 (고품질 확보)
            PDF: 파일만 저장, 텍스트 추출은 나중에 배치 파서로 처리

            개선사항:
            - e-print가 PDF를 직접 반환하는 경우 재다운로드 없이 바로 저장
            - .TEX 대문자 확장자 처리 (_find_main_tex_in_tar에서 수정)
            """
            global_rl.wait()  # 전역 rate limit (모든 워커 공유)

            # e-print 직접 다운로드 (LaTeX or PDF 판별)
            try:
                eprint_bytes = self._download_eprint(arxiv_id)
            except Exception:
                eprint_bytes = None

            if eprint_bytes is not None:
                # e-print가 PDF를 직접 반환하는 경우 (LaTeX 없는 오래된 논문)
                if eprint_bytes[:4] == b"%PDF":
                    if pdf_fallback:
                        pdf_path = self._save_pdf_bytes(arxiv_id, eprint_bytes)
                        if pdf_path:
                            return arxiv_id, pdf_path, "arxiv_pdf_saved"
                else:
                    # LaTeX 추출 시도
                    tex = self._extract_main_tex(eprint_bytes)
                    if tex:
                        return arxiv_id, tex, "arxiv_latex"
                    # e-print에 LaTeX 없음 → PDF 따로 다운로드
                    if pdf_fallback:
                        global_rl.wait()
                        pdf_path = self._download_and_save_pdf(arxiv_id)
                        if pdf_path:
                            return arxiv_id, pdf_path, "arxiv_pdf_saved"
            else:
                # e-print 자체 실패 → PDF 직접 시도
                if pdf_fallback:
                    global_rl.wait()
                    pdf_path = self._download_and_save_pdf(arxiv_id)
                    if pdf_path:
                        return arxiv_id, pdf_path, "arxiv_pdf_saved"

            return arxiv_id, None, "failed"

        save_counter = 0
        SAVE_EVERY = 100

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_one, aid): aid
                for aid in pending_ids
            }

            for future in as_completed(futures):
                arxiv_id, content, source_type = future.result()

                with df_lock:
                    mask = df["arxiv_id"] == arxiv_id
                    if source_type == "arxiv_latex":
                        df.loc[mask, "has_full_text"] = True
                        df.loc[mask, "full_text"] = content
                        df.loc[mask, "full_text_format"] = "latex"
                        df.loc[mask, "full_text_source_type"] = "arxiv_latex"
                        df.loc[mask, "full_text_status"] = "success"
                    elif source_type == "arxiv_pdf_saved":
                        # PDF는 파일 경로 저장, 텍스트 추출은 나중에 배치 파서로
                        df.loc[mask, "has_full_text"] = False
                        df.loc[mask, "pdf_path"] = content  # content = pdf_path
                        df.loc[mask, "full_text_source_type"] = "arxiv_pdf_saved"
                        df.loc[mask, "full_text_status"] = "pdf_downloaded"
                    else:
                        df.loc[mask, "full_text_status"] = "download_failed"

                with ckpt_lock:
                    latex_ckpt.add_processed_id(arxiv_id)

                with stats_lock:
                    if source_type == "arxiv_latex":
                        stats["success"] += 1
                    elif source_type == "arxiv_pdf_saved":
                        stats["success"] += 1
                        stats["pdf_fallback"] += 1
                    else:
                        stats["failed"] += 1
                    save_counter += 1
                    total_done = stats["success"] + stats["failed"]

                    if total_done % 10 == 0 and total_done > 0:
                        logger.info(
                            f"다운로드 진행: latex={stats['success'] - stats['pdf_fallback']}, "
                            f"pdf저장={stats['pdf_fallback']}, failed={stats['failed']}, "
                            f"skip={stats['skipped']}"
                        )

                # 100건마다 저장 (lock 밖에서 I/O)
                if save_counter % SAVE_EVERY == 0:
                    with ckpt_lock:
                        latex_ckpt.save()
                    with df_lock:
                        save_df_to_parquet(df, config.DATA_DIR, "arxiv", self.batch_id)
                    logger.info(f"  중간 저장 완료 ({save_counter}건 처리)")

        # 최종 저장
        latex_ckpt.save()
        save_df_to_parquet(df, config.DATA_DIR, "arxiv", self.batch_id)
        logger.info(f"다운로드 완료: {stats}")
        return stats

    # ── PDF 폴백 ──

    @retry(max_retries=2, backoff=2.0, exceptions=(requests.RequestException,))
    def _download_pdf(self, arxiv_id: str) -> bytes | None:
        """arXiv PDF 다운로드."""
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        resp = requests.get(url, timeout=90, allow_redirects=True)
        if resp.status_code == 404:
            return None
        if resp.status_code == 403:
            logger.debug(f"PDF 403 Forbidden ({arxiv_id}) — 오래된 논문일 수 있음")
            return None
        resp.raise_for_status()
        # PDF 헤더 확인
        if not resp.content.startswith(b"%PDF"):
            return None
        return resp.content

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str | None:
        """PyMuPDF로 PDF 텍스트 추출."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            for page in doc:
                text = page.get_text("text")
                if text.strip():
                    pages.append(text)
            doc.close()
            result = "\n".join(pages).strip()
            return result if len(result) > 200 else None
        except Exception as e:
            logger.debug(f"PyMuPDF 추출 실패: {e}")
            return None

    def _save_pdf_bytes(self, arxiv_id: str, pdf_bytes: bytes) -> str | None:
        """이미 받은 PDF bytes를 디스크에 저장. 경로 반환."""
        import os as _os  # 클로저 스코프 이슈 방지 (from __future__ import annotations)
        try:
            pdf_dir = _os.path.join(config.DATA_DIR, "arxiv", "pdfs")
            _os.makedirs(pdf_dir, exist_ok=True)
            safe_id = arxiv_id.replace("/", "_")
            pdf_path = _os.path.join(pdf_dir, f"{safe_id}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            return pdf_path
        except Exception as e:
            logger.warning(f"PDF bytes 저장 실패 ({arxiv_id}): {e}")
            return None

    def _download_and_save_pdf(self, arxiv_id: str) -> str | None:
        """PDF URL 다운로드 후 디스크 저장. 저장 경로 반환."""
        try:
            pdf_bytes = self._download_pdf(arxiv_id)
            if pdf_bytes is None:
                logger.debug(f"PDF 다운로드 실패 ({arxiv_id}): None 반환")
                return None
            return self._save_pdf_bytes(arxiv_id, pdf_bytes)
        except Exception as e:
            logger.debug(f"PDF 다운로드/저장 실패 ({arxiv_id}): {e}")
            return None

    def _download_and_extract_pdf(self, arxiv_id: str) -> str | None:
        """PDF 다운로드 후 PyMuPDF로 텍스트 즉시 추출 (베이스라인용)."""
        try:
            pdf_bytes = self._download_pdf(arxiv_id)
            if pdf_bytes is None:
                return None
            return self._extract_text_from_pdf(pdf_bytes)
        except Exception as e:
            logger.debug(f"PDF 처리 실패 ({arxiv_id}): {e}")
            return None

    @retry(max_retries=3, backoff=2.0, exceptions=(requests.RequestException,))
    def _download_eprint(self, arxiv_id: str) -> bytes | None:
        """arXiv e-print 다운로드."""
        url = f"{config.ARXIV_EPRINT_BASE}{arxiv_id}"
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content

    def _download_and_extract_tex(self, arxiv_id: str) -> str | None:
        """e-print 다운로드 후 메인 .tex 파일 추출."""
        try:
            content = self._download_eprint(arxiv_id)
            if content is None:
                return None
            return self._extract_main_tex(content)
        except Exception as e:
            logger.warning(f"LaTeX 추출 실패 ({arxiv_id}): {e}")
            return None

    def _extract_main_tex(self, data: bytes) -> str | None:
        """tar.gz 또는 단일 tex에서 메인 .tex 추출."""
        # e-print가 PDF를 직접 반환하는 경우 조기 종료 (PDF 헤더 감지)
        if data[:4] == b"%PDF":
            return None

        # 단일 .tex 파일인 경우
        try:
            text = data.decode("utf-8", errors="replace")
            if "\\documentclass" in text:
                return text
        except Exception:
            pass

        # tar.gz인 경우
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                return self._find_main_tex_in_tar(tar)
        except tarfile.TarError:
            pass

        # tar (비압축)
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
                return self._find_main_tex_in_tar(tar)
        except tarfile.TarError:
            pass

        # gzip된 단일 파일
        try:
            decompressed = gzip.decompress(data)
            text = decompressed.decode("utf-8", errors="replace")
            if "\\documentclass" in text:
                return text
        except Exception:
            pass

        return None

    def _find_main_tex_in_tar(self, tar: tarfile.TarFile) -> str | None:
        """tar 아카이브에서 메인 .tex 파일 찾기."""
        tex_files = []
        for member in tar.getmembers():
            # 버그수정: 대소문자 무관 (.tex / .TEX / .Tex 모두 처리)
            if member.name.lower().endswith(".tex") and member.isfile():
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode("utf-8", errors="replace")
                    tex_files.append((member.name, content, len(content)))

        if not tex_files:
            return None

        # \documentclass 포함된 파일만 필터
        with_docclass = [
            (name, content, size)
            for name, content, size in tex_files
            if "\\documentclass" in content
        ]

        candidates = with_docclass if with_docclass else tex_files

        # 파일명 우선순위
        priority_names = ["main", "paper", "manuscript", "article"]
        for pname in priority_names:
            for name, content, size in candidates:
                if pname in name.lower():
                    return content

        # 가장 큰 파일
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0][1]
