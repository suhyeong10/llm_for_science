"""데이터 품질 리포트 생성."""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from . import config
from .utils import load_all_parquet, setup_logger

logger = setup_logger("stats_report", config.LOGS_DIR)


def generate_report(data_dir: str | None = None, source: str | None = None) -> dict:
    """품질 리포트 생성 및 저장."""
    data_dir = data_dir or config.DATA_DIR
    df = load_all_parquet(data_dir, source=source)

    if df.empty:
        logger.warning("데이터 없음")
        return {}

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_records": len(df),
    }

    # 소스별 분포
    report["by_source"] = df["source"].value_counts().to_dict()

    # 연도별 분포
    year_dist = df["year"].value_counts().sort_index().to_dict()
    report["by_year"] = {str(k): v for k, v in year_dist.items()}

    # full_text_status 분포
    report["full_text_status"] = df["full_text_status"].value_counts().to_dict()

    # full_text 확보율
    has_ft = df["has_full_text"].sum() if "has_full_text" in df.columns else 0
    report["full_text_rate"] = round(has_ft / len(df) * 100, 2)

    # clean_text 유무
    has_clean = df["clean_text"].notna().sum() if "clean_text" in df.columns else 0
    report["clean_text_rate"] = round(has_clean / len(df) * 100, 2)

    # token_count 통계
    if "token_count_approx" in df.columns:
        tc = df["token_count_approx"].dropna()
        if len(tc) > 0:
            report["token_stats"] = {
                "mean": round(tc.mean(), 1),
                "median": round(tc.median(), 1),
                "p95": round(tc.quantile(0.95), 1),
                "total": int(tc.sum()),
                "under_100": int((tc < 100).sum()),
                "over_200k": int((tc > 200_000).sum()),
            }

    # 언어 분포
    if "language" in df.columns:
        report["language_dist"] = df["language"].value_counts().head(10).to_dict()

    # 중복 통계
    if "is_duplicate" in df.columns:
        dupe_count = df["is_duplicate"].sum()
        report["duplicates"] = {
            "count": int(dupe_count),
            "rate": round(dupe_count / len(df) * 100, 2),
        }

    # 메타데이터 완전성
    required_fields = ["title", "abstract", "year", "source", "authors"]
    completeness = {}
    for field in required_fields:
        if field in df.columns:
            if field == "authors":
                non_null = df[field].apply(lambda x: hasattr(x, '__len__') and len(x) > 0).sum()
            else:
                non_null = df[field].notna().sum()
            completeness[field] = round(non_null / len(df) * 100, 2)
    report["field_completeness"] = completeness

    # 저장
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # JSON
    json_path = os.path.join(
        config.REPORTS_DIR,
        f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Markdown
    md_path = json_path.replace(".json", ".md")
    _write_markdown_report(report, md_path)

    logger.info(f"리포트 저장: {json_path}")
    return report


def _write_markdown_report(report: dict, path: str):
    """Markdown 포맷 리포트 작성."""
    lines = [
        f"# 데이터 품질 리포트",
        f"생성 시각: {report['generated_at']}",
        f"",
        f"## 기본 통계",
        f"- 총 레코드: **{report['total_records']:,}**건",
        f"- Full-text 확보율: **{report.get('full_text_rate', 0)}%**",
        f"- Clean text 확보율: **{report.get('clean_text_rate', 0)}%**",
        f"",
        f"## 소스별 분포",
    ]
    for src, cnt in report.get("by_source", {}).items():
        lines.append(f"- {src}: {cnt:,}건")

    lines += ["", "## full_text_status 분포"]
    for status, cnt in report.get("full_text_status", {}).items():
        lines.append(f"- {status}: {cnt:,}건")

    if "token_stats" in report:
        ts = report["token_stats"]
        lines += [
            "", "## 토큰 통계",
            f"- 평균: {ts['mean']:,.1f}",
            f"- 중앙값: {ts['median']:,.1f}",
            f"- P95: {ts['p95']:,.1f}",
            f"- 총합: {ts['total']:,}",
            f"- 100 미만: {ts['under_100']}건",
            f"- 200K 초과: {ts['over_200k']}건",
        ]

    if "duplicates" in report:
        d = report["duplicates"]
        lines += [
            "", "## 중복 통계",
            f"- 중복 수: {d['count']:,}건",
            f"- 중복율: {d['rate']}%",
        ]

    lines += ["", "## 연도별 분포"]
    for year, cnt in sorted(report.get("by_year", {}).items()):
        bar = "#" * min(cnt // 10, 50)
        lines.append(f"- {year}: {cnt:>5,} {bar}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_report(report: dict):
    """리포트를 콘솔에 간결하게 출력."""
    print(f"\n{'='*50}")
    print(f" 데이터 품질 리포트")
    print(f"{'='*50}")
    print(f" 총 레코드: {report.get('total_records', 0):,}건")
    print(f" Full-text 확보율: {report.get('full_text_rate', 0)}%")
    print(f" Clean text 확보율: {report.get('clean_text_rate', 0)}%")

    print(f"\n 소스별:")
    for src, cnt in report.get("by_source", {}).items():
        print(f"   {src}: {cnt:,}")

    print(f"\n full_text_status:")
    for status, cnt in report.get("full_text_status", {}).items():
        print(f"   {status}: {cnt:,}")

    if "token_stats" in report:
        ts = report["token_stats"]
        print(f"\n 토큰 통계:")
        print(f"   평균: {ts['mean']:,.1f}  중앙값: {ts['median']:,.1f}  P95: {ts['p95']:,.1f}")
        print(f"   총합: {ts['total']:,}  이상치(< 100): {ts['under_100']}  이상치(> 200K): {ts['over_200k']}")

    if "duplicates" in report:
        d = report["duplicates"]
        print(f"\n 중복: {d['count']:,}건 ({d['rate']}%)")

    print(f"{'='*50}\n")
