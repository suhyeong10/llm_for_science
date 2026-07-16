"""Semantic Scholar API crawler for the current chemistry-focused pipeline."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import requests

from . import config
from .schema import PaperRecord
from .utils import (
    CheckpointManager,
    RateLimiter,
    compute_content_hash,
    detect_language,
    generate_paper_id,
    generate_record_id,
    normalize_text,
    retry,
    save_records_to_parquet,
    setup_logger,
    load_all_parquet,
)

logger = setup_logger("semantic_scholar", config.LOGS_DIR)


class SemanticScholarCrawler:
    """Semantic Scholar 논문 크롤러."""

    def __init__(self, batch_id: str, resume: bool = False):
        self.batch_id = batch_id
        self.rate_limiter = RateLimiter(config.S2_RATE_LIMIT_SEC)

        ckpt_path = f"{config.DATA_DIR}/checkpoints/s2_metadata.json"
        self.ckpt = CheckpointManager(ckpt_path)
        if not resume:
            self.ckpt.reset()  # reset() 사용 (캐시 무효화 포함)

        self._record_counter = self.ckpt.get("record_counter", 1)
        self._buffer: list[PaperRecord] = []

        # arXiv 데이터와의 중복 체크용 인덱스 (지연 로드)
        self._existing_titles: set[str] | None = None
        self._existing_dois: set[str] | None = None

    def _ensure_existing_data_loaded(self):
        """arXiv 수집분의 title/DOI 인덱스를 지연 로드 (첫 사용 시 1회만)."""
        if self._existing_titles is not None:
            return

        self._existing_titles = set()
        self._existing_dois = set()

        df = load_all_parquet(config.DATA_DIR, source="arxiv")
        if df.empty:
            return

        # 벡터화 연산으로 빠르게 인덱스 구축
        if "title" in df.columns:
            titles = df["title"].dropna().apply(normalize_text)
            self._existing_titles = set(titles.tolist())
        if "doi" in df.columns:
            dois = df["doi"].dropna().str.lower()
            self._existing_dois = set(dois.tolist())

        logger.info(
            f"기존 arXiv 데이터 로드: {len(self._existing_titles)} titles, "
            f"{len(self._existing_dois)} DOIs"
        )

    def _is_duplicate_of_arxiv(self, title: str, doi: str | None) -> bool:
        """arXiv 수집분과 중복인지 체크."""
        self._ensure_existing_data_loaded()
        if doi and doi.lower() in self._existing_dois:
            return True
        if normalize_text(title) in self._existing_titles:
            return True
        return False

    def crawl(self, max_results: int | None = None) -> int:
        """S2 API로 현재 담당 범위의 화학 논문을 수집한다. 수집 건수 반환."""
        offset = self.ckpt.get("offset", 0)
        fetched = 0

        logger.info(f"Semantic Scholar 크롤링 시작 (offset={offset})")

        while True:
            if max_results and fetched >= max_results:
                break

            self.rate_limiter.wait()
            papers, total = self._fetch_page(offset)

            if not papers:
                logger.info(f"S2: 더 이상 결과 없음 (offset={offset})")
                break

            for paper in papers:
                if max_results and fetched >= max_results:
                    break

                record = self._parse_paper(paper)
                if record is None:
                    continue

                if not (config.YEAR_RANGE[0] <= record.year <= config.YEAR_RANGE[1]):
                    continue

                if self._is_duplicate_of_arxiv(record.title, record.doi):
                    continue

                record.record_id = generate_record_id(self._record_counter)
                record.paper_id = generate_paper_id(self._record_counter)
                record.crawl_batch_id = self.batch_id
                record.crawl_date = datetime.now(timezone.utc).isoformat()
                record.content_hash = compute_content_hash(
                    record.title, record.abstract
                )

                self._buffer.append(record)
                self._record_counter += 1
                fetched += 1

            offset += len(papers)

            if len(self._buffer) >= config.PARQUET_ROWS_PER_FILE:
                self._flush_buffer()

            self.ckpt.set("offset", offset)
            self.ckpt.set("record_counter", self._record_counter)
            self.ckpt.save()

            logger.info(f"  S2: {fetched}건 수집 (offset={offset}, total={total})")

            if offset >= total:
                break

        self._flush_buffer()
        logger.info(f"Semantic Scholar 수집 완료: 총 {fetched}건")
        return fetched

    @retry(max_retries=5, backoff=3.0, exceptions=(requests.RequestException,))
    def _fetch_page(self, offset: int) -> tuple[list, int]:
        """S2 API 한 페이지 호출."""
        url = f"{config.S2_API_BASE}/paper/search"
        params = {
            "query": "chemistry",
            "fieldsOfStudy": "Chemistry",
            "year": f"{config.YEAR_RANGE[0]}-{config.YEAR_RANGE[1]}",
            "fields": config.S2_FIELDS,
            "offset": offset,
            "limit": min(config.S2_BATCH_SIZE, 50),
        }
        headers = {}
        if config.S2_API_KEY:
            headers["x-api-key"] = config.S2_API_KEY

        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            logger.warning("S2 429 rate limit. 60초 대기...")
            time.sleep(60)
            raise requests.RequestException("429 rate limited")
        resp.raise_for_status()

        data = resp.json()
        return data.get("data", []), data.get("total", 0)

    def _parse_paper(self, paper: dict) -> PaperRecord | None:
        """S2 API 응답 → PaperRecord."""
        try:
            paper_id_s2 = paper.get("paperId", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            if not title or not abstract:
                return None

            authors = [
                a.get("name", "") for a in paper.get("authors", [])
            ]
            year = paper.get("year") or 0
            categories = paper.get("fieldsOfStudy") or []
            citation_count = paper.get("citationCount")

            ext_ids = paper.get("externalIds") or {}
            doi = ext_ids.get("DOI")
            arxiv_id = ext_ids.get("ArXiv")

            oap = paper.get("openAccessPdf") or {}
            pdf_url = oap.get("url")

            # 언어 감지 (arXiv 크롤러와 동일하게 적용)
            language = detect_language(abstract)

            return PaperRecord(
                source="semantic_scholar",
                source_paper_id=f"S2_{paper_id_s2}",
                arxiv_id=arxiv_id,
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                categories=categories,
                license=None,
                pdf_url=pdf_url,
                has_full_text=False,
                full_text_status="abstract_only",
                clean_text=abstract,
                token_count_approx=len(abstract.split()) if abstract else 0,
                language=language,
                citation_count=citation_count,
                fetch_success=True,
                raw_source_payload=json.dumps(
                    {"paperId": paper_id_s2, "title": title},
                    ensure_ascii=False,
                ),
            )
        except Exception as e:
            logger.warning(f"S2 paper 파싱 실패: {e}")
            return None

    def _flush_buffer(self):
        if not self._buffer:
            return
        save_records_to_parquet(
            self._buffer, config.DATA_DIR, "semantic_scholar", self.batch_id,
        )
        logger.info(f"  S2: {len(self._buffer)}건 parquet 저장")
        self._buffer.clear()
