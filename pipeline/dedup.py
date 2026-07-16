"""중복 제거 모듈."""

from __future__ import annotations

import os
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from . import config
from .schema import PARQUET_SCHEMA
from .utils import load_all_parquet, normalize_text, setup_logger

logger = setup_logger("dedup", config.LOGS_DIR)

FUZZY_THRESHOLD = 90      # token_sort_ratio 제목 유사도 (90%+ = 거의 동일 제목)
ABSTRACT_THRESHOLD = 80   # abstract 유사도 2차 확인 기준 (제목 매칭된 쌍만)

# 중복 발생 시 보존 우선순위 (낮을수록 우선 보존)
SOURCE_TYPE_PRIORITY: dict[str | None, int] = {
    "arxiv_latex":            0,  # 최우선 — LaTeX 원본
    "arxiv_pdf_mineru":       1,  # GPU 파서 (수식 최고 품질)
    "arxiv_pdf_docling":      2,  # CPU 파서 (고품질)
    "arxiv_pdf_pymupdf4llm":  3,  # pymupdf4llm (기본)
    "arxiv_pdf_pymupdf":      4,  # 구버전 baseline
    "arxiv_pdf_saved":        5,  # raw PDF (미파싱)
    None:                     9,  # abstract only
}


def _source_priority(row: pd.Series) -> tuple[int, int]:
    """(source_type 우선순위, source 우선순위) — 낮을수록 보존."""
    st = row.get("full_text_source_type")
    src = row.get("source", "")
    return (
        SOURCE_TYPE_PRIORITY.get(st, 9),
        0 if src == "arxiv" else 1,
    )


def deduplicate(data_dir: str | None = None) -> dict:
    """전체 데이터에서 중복 제거 후 merged/ 에 저장."""
    data_dir = data_dir or config.DATA_DIR

    df = load_all_parquet(data_dir)
    if df.empty:
        logger.warning("데이터 없음")
        return {"original": 0, "duplicates": 0, "final": 0, "hash_duplicates": 0, "fuzzy_duplicates": 0}

    original_count = len(df)
    df["is_duplicate"] = False

    # Phase 1: content_hash 기반 정확 매칭
    hash_dupes = _find_hash_duplicates(df)
    logger.info(f"Phase 1 (hash): {len(hash_dupes)}건 중복 발견")

    # Phase 2: fuzzy title matching (year 블로킹 + rapidfuzz.process 최적화)
    fuzzy_dupes = _find_fuzzy_duplicates_fast(df, exclude_indices=hash_dupes)
    logger.info(f"Phase 2 (fuzzy): {len(fuzzy_dupes)}건 추가 중복 발견")

    all_dupes = hash_dupes | fuzzy_dupes
    df.loc[list(all_dupes), "is_duplicate"] = True

    # merged/ 에 저장
    merged_dir = os.path.join(data_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    for year, group in df.groupby("year"):
        year_dir = os.path.join(merged_dir, f"year={year}")
        os.makedirs(year_dir, exist_ok=True)
        fpath = os.path.join(year_dir, "final.parquet")
        table = pa.Table.from_pandas(group, schema=PARQUET_SCHEMA, preserve_index=False)
        pq.write_table(table, fpath)

    stats = {
        "original": original_count,
        "duplicates": len(all_dupes),
        "final": original_count - len(all_dupes),
        "hash_duplicates": len(hash_dupes),
        "fuzzy_duplicates": len(fuzzy_dupes),
    }
    logger.info(f"중복 제거 완료: {stats}")
    return stats


def _find_hash_duplicates(df: pd.DataFrame) -> set[int]:
    """content_hash 기반 정확 중복. 첫 arXiv 항목만 남김."""
    duplicates = set()
    hash_groups: dict[str, list[int]] = defaultdict(list)

    for idx, row in df.iterrows():
        h = row.get("content_hash", "")
        if h:
            hash_groups[h].append(idx)

    for h, indices in hash_groups.items():
        if len(indices) > 1:
            sorted_idx = sorted(
                indices,
                key=lambda i: _source_priority(df.loc[i]),
            )
            duplicates.update(sorted_idx[1:])

    return duplicates


def _find_fuzzy_duplicates_fast(
    df: pd.DataFrame, exclude_indices: set[int]
) -> set[int]:
    """rapidfuzz.process.extract를 사용한 고속 fuzzy matching. year 블로킹."""
    duplicates = set()
    candidates = df[~df.index.isin(exclude_indices)].copy()

    if len(candidates) < 2:
        return duplicates

    # year 기반 블로킹
    by_year: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for idx, row in candidates.iterrows():
        year = row.get("year", 0)
        title_norm = normalize_text(str(row.get("title", "")))
        by_year[year].append((idx, title_norm))

    years = sorted(by_year.keys())
    total_compared = 0

    for year in years:
        group = by_year[year]
        if len(group) < 2:
            continue

        # rapidfuzz.process.extract로 각 제목에 대해 유사한 것 찾기
        indices = [g[0] for g in group]
        titles = [g[1] for g in group]

        found_in_year = _batch_fuzzy_match(
            indices, titles, df, duplicates, FUZZY_THRESHOLD
        )
        total_compared += len(group)

        if year % 5 == 0:
            logger.info(
                f"  Fuzzy dedup {year}: {len(group)}건 비교, "
                f"누적 {len(duplicates)}건 중복"
            )

    return duplicates


def _batch_fuzzy_match(
    indices: list[int],
    titles: list[str],
    df: pd.DataFrame,
    duplicates: set[int],
    threshold: int,
) -> int:
    """단일 연도 그룹 내에서 배치 fuzzy matching."""
    found = 0
    matched_pairs: set[tuple[int, int]] = set()

    for i in range(len(titles)):
        if indices[i] in duplicates:
            continue

        # 현재 제목과 나머지 비교 (자기 자신 이후만)
        remaining_titles = titles[i + 1:]
        remaining_indices = indices[i + 1:]

        if not remaining_titles:
            continue

        # rapidfuzz.process.extract로 상위 매칭 찾기
        matches = process.extract(
            titles[i],
            remaining_titles,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
            limit=10,  # 한 제목당 최대 10개 유사 제목
        )

        for match_title, score, match_idx in matches:
            real_idx = remaining_indices[match_idx]
            if real_idx in duplicates:
                continue

            pair = tuple(sorted((indices[i], real_idx)))
            if pair in matched_pairs:
                continue
            matched_pairs.add(pair)

            # abstract 2차 확인
            abs_a = normalize_text(str(df.at[indices[i], "abstract"] or ""))
            abs_b = normalize_text(str(df.at[real_idx, "abstract"] or ""))
            abs_score = fuzz.token_sort_ratio(abs_a, abs_b)

            if abs_score >= ABSTRACT_THRESHOLD:
                # LaTeX > 고품질 PDF > 저품질 PDF > abstract only 순으로 보존
                pri_a = _source_priority(df.loc[indices[i]])
                pri_b = _source_priority(df.loc[real_idx])
                if pri_b < pri_a:
                    duplicates.add(indices[i])
                else:
                    duplicates.add(real_idx)
                found += 1

    return found
