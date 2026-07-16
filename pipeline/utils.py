"""공통 유틸리티: 해싱, retry, rate limiting, checkpoint, 로깅, I/O."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import functools
from datetime import datetime


# ── 해싱 ──

def normalize_text(text: str) -> str:
    """제목/abstract 정규화: 소문자 + 공백 정리. 해시/비교용."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_content_hash(title: str, abstract: str) -> str:
    """title+abstract 기반 SHA256 해시. 중복 제거용."""
    combined = normalize_text(title) + "|||" + normalize_text(abstract)
    h = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


# ── ID 생성 ──

def generate_record_id(counter: int) -> str:
    """raw_{counter:06d} 형태의 레코드 ID."""
    return f"raw_{counter:06d}"


def generate_paper_id(counter: int) -> str:
    """paper_{counter:06d} 형태의 논문 ID."""
    return f"paper_{counter:06d}"


# ── Retry decorator ──

def retry(max_retries: int = 3, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Exponential backoff retry decorator.

    wait = backoff^attempt (attempt 0부터). 즉 첫 retry는 1초(2^0), 두번째 2초(2^1)...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        wait = backoff ** attempt
                        logging.getLogger("retry").warning(
                            f"{func.__name__} attempt {attempt+1} failed: {e}. "
                            f"Retrying in {wait:.1f}s..."
                        )
                        time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


# ── Rate Limiter ──

class RateLimiter:
    """단일 스레드용 rate limiter. interval_sec 간격을 보장."""

    def __init__(self, interval_sec: float):
        self.interval = interval_sec
        self._last_call = 0.0

    def wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.time()


class GlobalRateLimiter:
    """스레드 안전 전역 rate limiter. 멀티워커 환경에서 전체 요청을 균등하게 분산.

    워커별 독립 RateLimiter와 달리, 모든 워커가 하나의 토큰 풀을 공유.
    burst 없이 interval_sec 간격으로 요청을 고르게 배분.
    """

    def __init__(self, interval_sec: float):
        self.interval = interval_sec
        self._last_call = 0.0
        self._lock = __import__("threading").Lock()

    def wait(self):
        with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last_call = time.time()


# ── Checkpoint Manager ──

class CheckpointManager:
    """JSON 기반 크롤링 상태 관리. 연도별/카테고리별 offset + processed_ids 추적."""

    def __init__(self, path: str):
        self.path = path
        self._data: dict = {}
        self._processed_ids_cache: set[str] | None = None
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}
        self._processed_ids_cache = None

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # atomic write: 임시 파일에 쓰고 rename
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value):
        self._data[key] = value

    def _get_processed_set(self) -> set[str]:
        """processed_ids를 set으로 캐싱하여 반환."""
        if self._processed_ids_cache is None:
            self._processed_ids_cache = set(self._data.get("processed_ids", []))
        return self._processed_ids_cache

    def add_processed_id(self, paper_id: str):
        s = self._get_processed_set()
        s.add(paper_id)
        self._data["processed_ids"] = list(s)

    def is_processed(self, paper_id: str) -> bool:
        return paper_id in self._get_processed_set()

    def reset(self):
        """상태 초기화. 캐시 포함."""
        self._data = {}
        self._processed_ids_cache = None

    @property
    def last_offset(self) -> int:
        return self._data.get("last_offset", 0)

    @last_offset.setter
    def last_offset(self, value: int):
        self._data["last_offset"] = value
        self._data["last_update"] = datetime.now().isoformat()


# ── 로깅 ──

class JsonFormatter(logging.Formatter):
    """JSON lines 포맷 로거."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logger(name: str, log_dir: str, level: int = logging.INFO) -> logging.Logger:
    """JSON lines 로거 설정 (stdout: human-readable, file: JSON lines)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # stdout handler (human-readable)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    # file handler (JSON lines)
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.jsonl"),
        encoding="utf-8",
    )
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)

    return logger


# ── 언어 감지 ──

def detect_language(text: str) -> str:
    """abstract/text의 언어 감지. 20자 미만이거나 실패 시 'en' 반환."""
    if not text or len(text.strip()) < 20:
        return "en"
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


# ── Parquet I/O helpers ──

def save_records_to_parquet(
    records: list,
    data_dir: str,
    source: str,
    batch_id: str,
    rows_per_file: int = 10_000,
):
    """PaperRecord 리스트를 year별 파티셔닝된 parquet으로 저장.

    resume 시 기존 파일 번호에 이어서 작성하여 덮어쓰기 방지.
    """
    import pyarrow.parquet as pq
    from .schema import records_to_table

    if not records:
        return

    by_year: dict[int, list] = {}
    for r in records:
        y = r.year if hasattr(r, "year") else r.get("year", 0)
        by_year.setdefault(y, []).append(r)

    for year, year_records in by_year.items():
        year_dir = os.path.join(data_dir, source, f"year={year}")
        os.makedirs(year_dir, exist_ok=True)

        # 기존 파일 수 확인 (makedirs 이후이므로 FileNotFoundError 없음)
        existing_count = sum(
            1 for f in os.listdir(year_dir)
            if f.startswith(batch_id) and f.endswith(".parquet")
        )

        for i in range(0, len(year_records), rows_per_file):
            chunk = year_records[i : i + rows_per_file]
            chunk_idx = existing_count + (i // rows_per_file) + 1
            file_name = f"{batch_id}_{chunk_idx:03d}.parquet"
            file_path = os.path.join(year_dir, file_name)

            table = records_to_table(chunk)
            pq.write_table(table, file_path)


def save_df_to_parquet(df, data_dir: str, source: str, batch_id: str):
    """DataFrame을 year별 파티셔닝된 parquet으로 저장 (기존 파일 교체)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .schema import PARQUET_SCHEMA

    for year, group in df.groupby("year"):
        year_dir = os.path.join(data_dir, source, f"year={year}")
        os.makedirs(year_dir, exist_ok=True)

        # 기존 parquet 파일 삭제 (중복 방지)
        for f in os.listdir(year_dir):
            if f.endswith(".parquet"):
                os.remove(os.path.join(year_dir, f))

        file_path = os.path.join(year_dir, f"{batch_id}_001.parquet")
        table = pa.Table.from_pandas(group, schema=PARQUET_SCHEMA, preserve_index=False)
        pq.write_table(table, file_path)


def load_all_parquet(data_dir: str, source: str | None = None) -> "pd.DataFrame":
    """data_dir에서 모든 parquet 파일을 읽어 DataFrame 반환.

    source 미지정 시 arxiv + semantic_scholar (merged 제외).
    손상된 parquet 파일은 warning 로그 후 skip.
    """
    import pandas as pd

    frames = []
    search_dirs = []
    if source:
        search_dirs.append(os.path.join(data_dir, source))
    else:
        for d in ["arxiv", "semantic_scholar"]:
            p = os.path.join(data_dir, d)
            if os.path.isdir(p):
                search_dirs.append(p)

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, _, files in os.walk(search_dir):
            for fname in files:
                if fname.endswith(".parquet"):
                    fpath = os.path.join(root, fname)
                    try:
                        frames.append(pd.read_parquet(fpath))
                    except Exception as e:
                        logging.getLogger("utils").warning(
                            f"parquet 읽기 실패 (skip): {fpath}: {e}"
                        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
