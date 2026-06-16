"""Parquet 스키마 정의 및 레코드 헬퍼."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import pyarrow as pa


PARQUET_SCHEMA = pa.schema([
    # 식별자
    ("record_id", pa.string()),
    ("paper_id", pa.string()),
    ("source", pa.string()),
    ("source_paper_id", pa.string()),
    # 교차 참조
    ("arxiv_id", pa.string()),
    ("doi", pa.string()),
    # 메타데이터
    ("title", pa.string()),
    ("abstract", pa.string()),
    ("authors", pa.list_(pa.string())),
    ("year", pa.int32()),
    ("categories", pa.list_(pa.string())),
    ("license", pa.string()),
    # 파일 경로
    ("pdf_url", pa.string()),
    ("pdf_path", pa.string()),
    ("latex_source_path", pa.string()),
    # 본문
    ("has_full_text", pa.bool_()),
    ("full_text", pa.string()),
    ("full_text_format", pa.string()),
    ("full_text_source_type", pa.string()),
    ("full_text_status", pa.string()),
    ("clean_text", pa.string()),
    # 통계
    ("citation_count", pa.int32()),
    ("token_count_approx", pa.int32()),
    # 메타
    ("language", pa.string()),
    ("crawl_date", pa.string()),
    ("crawl_batch_id", pa.string()),
    ("fetch_success", pa.bool_()),
    ("error_message", pa.string()),
    ("content_hash", pa.string()),
    ("is_duplicate", pa.bool_()),
    ("raw_source_payload", pa.string()),
])


@dataclass
class PaperRecord:
    """논문 레코드 dataclass."""

    record_id: str = ""
    paper_id: str = ""
    source: str = ""
    source_paper_id: str = ""

    arxiv_id: Optional[str] = None
    doi: Optional[str] = None

    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int = 0
    categories: list[str] = field(default_factory=list)
    license: Optional[str] = None

    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None
    latex_source_path: Optional[str] = None

    has_full_text: bool = False
    full_text: Optional[str] = None
    full_text_format: Optional[str] = None
    full_text_source_type: Optional[str] = None
    full_text_status: str = "abstract_only"
    clean_text: Optional[str] = None

    citation_count: Optional[int] = None
    token_count_approx: Optional[int] = None

    language: str = "en"
    crawl_date: str = ""
    crawl_batch_id: str = ""
    fetch_success: bool = True
    error_message: Optional[str] = None
    content_hash: str = ""
    is_duplicate: bool = False
    raw_source_payload: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def records_to_table(records: list[PaperRecord]) -> pa.Table:
    """PaperRecord 리스트를 PyArrow Table로 변환."""
    if not records:
        return pa.table({f.name: [] for f in PARQUET_SCHEMA}, schema=PARQUET_SCHEMA)

    dicts = [r.to_dict() for r in records]
    columns = {}
    for f in PARQUET_SCHEMA:
        values = [d.get(f.name) for d in dicts]
        columns[f.name] = values

    return pa.table(columns, schema=PARQUET_SCHEMA)
