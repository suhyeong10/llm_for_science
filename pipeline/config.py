"""크롤러 설정."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# --- arXiv ---
ARXIV_CATEGORIES = [
    # Tier 1: 핵심 화학 (수집 완료)
    "physics.chem-ph",       # Chemical Physics (~18,500건)
    "cond-mat.mtrl-sci",     # Materials Science (~80,000건)
    "physics.atm-clus",      # Atomic/Molecular Clusters (~2,400건)
    # Tier 2: 확장 화학
    "cond-mat.soft",         # Soft Condensed Matter - 고분자, 콜로이드 (~35,000건)
    "physics.comp-ph",       # Computational Physics - 계산화학 겸용 (~19,000건)
    "physics.bio-ph",        # Biological Physics - 생물물리화학 (~13,400건)
    "q-bio.BM",              # Biomolecules - 생체분자 (~4,700건)
    "physics.atom-ph",       # Atomic Physics - 원자/분자 분광학 (~13,000건)
    "cs.CE",                 # Comp. Engineering & Science - 화학정보학 (~3,000건)
    # Tier 3: 선택적 (필요 시 주석 해제, 노이즈 가능):
    # "cond-mat.mes-hall",   # Mesoscale/Nanoscale - 나노화학 일부 (~60,000건)
    # "cond-mat.str-el",     # Strongly Correlated Electrons (~45,000건)
    # "cond-mat.stat-mech",  # Statistical Mechanics - 열역학/반응속도론 (~40,000건)
]

YEAR_RANGE = (2000, 2025)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SEC = 1.0  # 전역 GlobalRateLimiter 기준: 1 req/1sec = 60 req/min
                             # e-print 다운로드 서버는 API보다 관대 (429 발생 시 자동 60초 대기)
ARXIV_EPRINT_BASE = "https://arxiv.org/e-print/"
ARXIV_BATCH_SIZE = 100  # max_results per API call

# --- Semantic Scholar ---
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.environ.get("S2_API_KEY")
S2_RATE_LIMIT_SEC = 3.0
S2_BATCH_SIZE = 100
S2_FIELDS = (
    "paperId,title,abstract,authors,year,citationCount,"
    "fieldsOfStudy,externalIds,openAccessPdf,publicationTypes"
)

# --- 저장 ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(REPO_ROOT / "data")
LOGS_DIR = str(REPO_ROOT / "logs")
REPORTS_DIR = str(REPO_ROOT / "reports")
KEYWORDS_DIR = str(REPO_ROOT / "keywords")
PARQUET_ROWS_PER_FILE = 10_000  # 파일당 행 수 (메모리/파일 크기 균형)

# --- 동시성 ---
CONCURRENT_DOWNLOADS = 8   # 워커 수 (GlobalRateLimiter가 실제 속도 제어)
MAX_RETRIES = 3            # 기본 retry 횟수
RETRY_BACKOFF = 2.0        # retry 간 대기 배수 (2^attempt 초)


def generate_batch_id(custom_id: str | None = None) -> str:
    """배치 식별자 생성."""
    if custom_id:
        return custom_id
    return f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_batch_config(batch_id: str, extra: dict | None = None) -> dict:
    """배치 config snapshot을 dict로 반환 (JSON 저장용)."""
    cfg = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "arxiv_categories": ARXIV_CATEGORIES,
        "year_range": list(YEAR_RANGE),
        "s2_api_key_set": S2_API_KEY is not None,
    }
    if extra:
        cfg.update(extra)
    return cfg
