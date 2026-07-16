#!/usr/bin/env python3
"""
download_failed 논문 복구 스크립트.

버그(os scope 이슈)로 인해 PDF 저장이 실패하여 download_failed로 잘못 기록된
논문들을 복구합니다.

실행 방법:
    python -m pipeline.recover_failed [--dry-run] [--yes]

실행 순서:
    1. 모든 parquet에서 full_text_status='download_failed' 논문 ID 수집
    2. checkpoint의 processed_ids에서 해당 ID 제거
    3. parquet에서 해당 논문의 status를 'abstract_only'로 초기화
    4. python main.py arxiv-latex --resume 으로 재처리

주의: 현재 크롤이 실행 중이라면 반드시 완료 후 실행할 것.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_PATH = DATA_DIR / "checkpoints" / "arxiv_latex.json"


def find_failed_ids() -> list[str]:
    """download_failed 상태인 논문의 arxiv_id 목록 반환."""
    failed = []
    parquet_files = sorted(glob.glob(str(DATA_DIR / "arxiv" / "**" / "*.parquet"), recursive=True))
    print(f"Parquet 파일 수: {len(parquet_files)}")

    for f in parquet_files:
        try:
            df = pd.read_parquet(f, columns=["arxiv_id", "full_text_status"])
            batch = df[df["full_text_status"] == "download_failed"]["arxiv_id"].dropna().tolist()
            failed.extend(batch)
        except Exception as e:
            print(f"  WARNING: 읽기 실패 (skip) {Path(f).name}: {e}")

    return failed


def reset_checkpoint(failed_ids: set[str], dry_run: bool = False) -> int:
    """checkpoint에서 failed_ids 제거. 제거된 수 반환."""
    if not CHECKPOINT_PATH.exists():
        print("checkpoint 파일 없음 — 건너뜀")
        return 0

    with open(CHECKPOINT_PATH) as f:
        cp = json.load(f)

    processed = set(cp.get("processed_ids", []))
    before = len(processed)
    to_remove = processed & failed_ids
    after_set = processed - failed_ids

    print(f"checkpoint processed_ids: {before:,} → {len(after_set):,} ({len(to_remove):,}건 제거 예정)")

    if not dry_run:
        cp["processed_ids"] = list(after_set)
        # 원자적 쓰기
        tmp_path = str(CHECKPOINT_PATH) + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(cp, f)
        os.replace(tmp_path, CHECKPOINT_PATH)
        print(f"  → checkpoint 업데이트 완료")

    return len(to_remove)


def reset_parquet_status(failed_ids: set[str], dry_run: bool = False) -> int:
    """parquet에서 download_failed → abstract_only 초기화. 처리된 레코드 수 반환."""
    parquet_files = sorted(glob.glob(str(DATA_DIR / "arxiv" / "**" / "*.parquet"), recursive=True))
    total_reset = 0

    for filepath in parquet_files:
        # pandas로 읽기 (스키마 불일치 허용)
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            print(f"  WARNING: {Path(filepath).name} 읽기 실패: {e}")
            continue

        # 이 파일에서 수정 대상 찾기
        mask = (df["full_text_status"] == "download_failed") & (df["arxiv_id"].isin(failed_ids))
        count = int(mask.sum())
        if count == 0:
            continue

        print(f"  {Path(filepath).name}: {count}건 초기화{'(dry-run)' if dry_run else ''}")
        total_reset += count

        if not dry_run:
            df.loc[mask, "full_text_status"] = "abstract_only"
            df.loc[mask, "has_full_text"] = False
            df.loc[mask, "full_text"] = None
            df.loc[mask, "full_text_format"] = None
            df.loc[mask, "full_text_source_type"] = None
            df.loc[mask, "error_message"] = None
            df.loc[mask, "fetch_success"] = False

            # 저장 (pandas → parquet)
            tmp_path = filepath + ".tmp"
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, filepath)

    return total_reset


def main():
    parser = argparse.ArgumentParser(description="download_failed 논문 복구")
    parser.add_argument("--dry-run", action="store_true", help="실제 변경 없이 시뮬레이션만")
    parser.add_argument("--yes", action="store_true", help="확인 없이 바로 실행")
    args = parser.parse_args()

    print("=" * 60)
    print("download_failed 논문 복구 스크립트")
    print("=" * 60)

    # 1. 실패 ID 수집
    print("\n[1/3] download_failed 논문 ID 수집 중...")
    failed_ids = find_failed_ids()
    print(f"  총 {len(failed_ids):,}건 발견")

    if not failed_ids:
        print("  복구할 논문 없음. 종료.")
        return

    failed_set = set(failed_ids)

    if args.dry_run:
        print("\n[DRY RUN 모드] 실제 변경 없이 시뮬레이션합니다.")

    if not args.dry_run and not args.yes:
        print(f"\n{len(failed_ids):,}건을 abstract_only로 초기화하고 checkpoint에서 제거합니다.")
        answer = input("계속하시겠습니까? (y/N): ").strip().lower()
        if answer != "y":
            print("취소됨.")
            sys.exit(0)

    # 2. checkpoint 업데이트
    print("\n[2/3] checkpoint 업데이트...")
    removed = reset_checkpoint(failed_set, dry_run=args.dry_run)

    # 3. parquet 상태 초기화
    print("\n[3/3] parquet 상태 초기화...")
    reset_count = reset_parquet_status(failed_set, dry_run=args.dry_run)
    print(f"  총 {reset_count:,}건 초기화 {'예정' if args.dry_run else '완료'}")

    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"[DRY RUN] checkpoint에서 제거 예정: {removed:,}건")
        print(f"[DRY RUN] parquet 초기화 예정: {reset_count:,}건")
        print("\n실제 실행: python -m pipeline.recover_failed --yes")
    else:
        print(f"복구 완료: {reset_count:,}건")
        print("\n다음 명령어로 재처리하세요:")
        print("  python main.py arxiv-latex --resume --workers 8")
    print("=" * 60)


if __name__ == "__main__":
    main()
