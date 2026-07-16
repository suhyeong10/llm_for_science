"""품질 점수화 + Karpathy Loop 지원.

평가 차원 (가중치):
  - 스키마 완전성 (15%): 필수 필드 null 비율
  - 본문 확보율 (20%): has_full_text=True 비율
  - LaTeX 파싱 품질 (20%): clean_text에서 잔존 LaTeX 명령어 비율
  - 메타데이터 정확성 (10%): year, categories, authors 유효성
  - 중복 제거 효과 (10%): 중복 발견율 (2~15% 범위 내인지)
  - 언어 순도 (5%): language="en" 비율
  - 토큰 분포 건전성 (10%): 이상치 비율
  - 해시 유일성 (10%): content_hash 유일 비율
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

import pandas as pd

from . import config
from .utils import load_all_parquet, setup_logger

logger = setup_logger("eval", config.LOGS_DIR)

DIMENSIONS = {
    "schema_completeness": 0.15,
    "fulltext_coverage": 0.20,
    "latex_parse_quality": 0.20,
    "metadata_accuracy": 0.10,
    "dedup_effectiveness": 0.10,
    "language_purity": 0.05,
    "token_distribution": 0.10,
    "hash_uniqueness": 0.10,
}


def _has_items(x) -> bool:
    """list/numpy array 등 길이가 있는 컬렉션이 비어있지 않은지 체크."""
    try:
        return hasattr(x, "__len__") and len(x) > 0
    except Exception:
        return False


def score_sample(
    df: pd.DataFrame | None = None,
    data_dir: str | None = None,
    sample_size: int | None = None,
) -> dict:
    """샘플(또는 전체)에 대한 품질 점수 산출. 0~100점."""
    if df is None:
        data_dir = data_dir or config.DATA_DIR
        # merged/ 가 있으면 dedup 결과 사용, 없으면 원본
        merged_dir = os.path.join(data_dir, "merged")
        if os.path.isdir(merged_dir) and any(
            f.endswith(".parquet") for _, _, fs in os.walk(merged_dir) for f in fs
        ):
            df = load_all_parquet(data_dir, source="merged")
        else:
            df = load_all_parquet(data_dir)

    if df.empty:
        logger.warning("데이터 없음")
        return {"total_score": 0, "dimensions": {}, "issues": []}

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    issues = []
    scores = {}

    scores["schema_completeness"] = _score_schema(df, issues)
    scores["fulltext_coverage"] = _score_fulltext(df, issues)
    scores["latex_parse_quality"] = _score_latex_quality(df, issues)
    scores["metadata_accuracy"] = _score_metadata(df, issues)
    scores["dedup_effectiveness"] = _score_dedup(df, issues)
    scores["language_purity"] = _score_language(df, issues)
    scores["token_distribution"] = _score_tokens(df, issues)
    scores["hash_uniqueness"] = _score_hash(df, issues)

    total = sum(scores[k] * DIMENSIONS[k] for k in DIMENSIONS) * 10
    total = round(total, 1)

    return {
        "total_score": total,
        "dimensions": {k: round(v, 2) for k, v in scores.items()},
        "sample_size": len(df),
        "issues": issues,
        "timestamp": datetime.now().isoformat(),
    }


def _score_schema(df: pd.DataFrame, issues: list) -> float:
    """필수 필드 완전성. 10점 만점."""
    required = ["title", "abstract", "year", "source", "content_hash"]
    total_checks = len(required) * len(df)
    if total_checks == 0:
        return 0

    missing = 0
    for field in required:
        if field in df.columns:
            null_count = df[field].isna().sum()
            if field in ("title", "abstract"):
                null_count += (df[field] == "").sum()
            missing += null_count
        else:
            missing += len(df)

    rate = 1 - (missing / total_checks)
    if missing > 0:
        issues.append(f"[Schema] {missing}/{total_checks} 필수 필드 누락")
    return rate * 10


def _score_fulltext(df: pd.DataFrame, issues: list) -> float:
    """full_text 확보율. 10점 만점."""
    if "has_full_text" not in df.columns:
        return 0
    rate = df["has_full_text"].sum() / len(df)
    if rate < 0.5:
        issues.append(f"[Full-text] 확보율 {rate*100:.1f}% (50% 미만)")
    return rate * 10


def _score_latex_quality(df: pd.DataFrame, issues: list) -> float:
    """clean_text에서 잔존 LaTeX 비율. 10점 만점."""
    if "clean_text" not in df.columns:
        return 5.0

    has_clean = df["clean_text"].dropna()
    if len(has_clean) == 0:
        return 5.0

    latex_pattern = re.compile(
        r"\\(?:begin|end|documentclass|usepackage|section|subsection|"
        r"cite|ref|label|textbf|textit|emph|newcommand|def)\b"
    )

    total_tokens = 0
    latex_tokens = 0
    for text in has_clean:
        tokens = str(text).split()
        total_tokens += len(tokens)
        latex_tokens += sum(1 for t in tokens if latex_pattern.search(t))

    if total_tokens == 0:
        return 10.0

    contamination_rate = latex_tokens / total_tokens
    if contamination_rate > 0.01:
        issues.append(
            f"[LaTeX] clean_text에 {contamination_rate*100:.2f}% LaTeX 명령어 잔존"
        )

    # 0% → 10점, 5%+ → 0점
    score = max(0, 10 * (1 - contamination_rate * 20))
    return score


def _score_metadata(df: pd.DataFrame, issues: list) -> float:
    """메타데이터 유효성. 10점 만점."""
    checks = 0
    passed = 0

    if "year" in df.columns:
        valid_year = df["year"].between(config.YEAR_RANGE[0], config.YEAR_RANGE[1])
        checks += len(df)
        passed += valid_year.sum()
        invalid = (~valid_year).sum()
        if invalid > 0:
            issues.append(f"[Metadata] {invalid}건 연도 범위 밖")

    if "categories" in df.columns:
        has_cats = df["categories"].apply(_has_items)
        checks += len(df)
        passed += has_cats.sum()

    if "authors" in df.columns:
        has_authors = df["authors"].apply(_has_items)
        checks += len(df)
        passed += has_authors.sum()
        no_authors = (~has_authors).sum()
        if no_authors > 0:
            issues.append(f"[Metadata] {no_authors}건 저자 정보 없음")

    if checks == 0:
        return 5.0
    return (passed / checks) * 10


def _score_dedup(df: pd.DataFrame, issues: list) -> float:
    """중복 제거 효과. 10점 만점."""
    if "is_duplicate" not in df.columns:
        return 5.0

    dupe_rate = df["is_duplicate"].sum() / len(df) if len(df) > 0 else 0

    if 0.02 <= dupe_rate <= 0.15:
        score = 10.0
    elif dupe_rate < 0.02:
        score = 7.0
        issues.append(f"[Dedup] 중복율 {dupe_rate*100:.1f}% (너무 낮음 - 미탐지 가능)")
    else:
        score = max(0, 10 - (dupe_rate - 0.15) * 50)
        issues.append(f"[Dedup] 중복율 {dupe_rate*100:.1f}% (높음)")

    return score


def _score_language(df: pd.DataFrame, issues: list) -> float:
    """언어 순도. 10점 만점."""
    if "language" not in df.columns:
        return 5.0

    en_rate = (df["language"] == "en").sum() / len(df)
    if en_rate < 0.95:
        issues.append(f"[Language] 영어 비율 {en_rate*100:.1f}%")
    return en_rate * 10


def _score_tokens(df: pd.DataFrame, issues: list) -> float:
    """토큰 분포 건전성. 10점 만점."""
    if "token_count_approx" not in df.columns:
        return 5.0

    tc = df["token_count_approx"].dropna()
    if len(tc) == 0:
        return 5.0

    outlier_count = ((tc < 50) | (tc > 200_000)).sum()
    outlier_rate = outlier_count / len(tc)

    if outlier_rate > 0.1:
        issues.append(f"[Token] 이상치 {outlier_count}건 ({outlier_rate*100:.1f}%)")

    return max(0, 10 * (1 - outlier_rate * 5))


def _score_hash(df: pd.DataFrame, issues: list) -> float:
    """content_hash 유일성. 10점 만점."""
    if "content_hash" not in df.columns:
        return 5.0

    # is_duplicate=False인 레코드만 대상 (DataFrame에서 안전하게 접근)
    if "is_duplicate" in df.columns:
        non_dupe = df[~df["is_duplicate"].fillna(False).astype(bool)]
    else:
        non_dupe = df

    if len(non_dupe) == 0:
        return 10.0

    unique_rate = non_dupe["content_hash"].nunique() / len(non_dupe)
    if unique_rate < 0.99:
        issues.append(
            f"[Hash] non-duplicate 레코드 중 해시 유일 비율: {unique_rate*100:.1f}%"
        )
    return unique_rate * 10


# ── Karpathy Loop 지원 ──

def run_eval(
    data_dir: str | None = None,
    sample_size: int | None = None,
) -> dict:
    """Eval 실행 + 결과 저장 + 콘솔 출력."""
    result = score_sample(data_dir=data_dir, sample_size=sample_size)

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    json_path = os.path.join(
        config.REPORTS_DIR,
        f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    history_path = os.path.join(config.REPORTS_DIR, "eval_history.jsonl")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print_eval_result(result)
    return result


def print_eval_result(result: dict):
    """Eval 결과 테이블 출력."""
    print(f"\n{'='*55}")
    print(f" Eval Score (Karpathy Loop)")
    print(f"{'='*55}")
    print(f" Sample size: {result.get('sample_size', '?')}")
    print(f"{'─'*55}")

    dims = result.get("dimensions", {})
    labels = {
        "schema_completeness": "Schema Completeness",
        "fulltext_coverage": "Full-text Coverage",
        "latex_parse_quality": "LaTeX Parse Quality",
        "metadata_accuracy": "Metadata Accuracy",
        "dedup_effectiveness": "Dedup Effectiveness",
        "language_purity": "Language Purity",
        "token_distribution": "Token Distribution",
        "hash_uniqueness": "Hash Uniqueness",
    }

    for key, label in labels.items():
        score = dims.get(key, 0)
        weight = DIMENSIONS.get(key, 0)
        bar = "█" * int(score) + "░" * (10 - int(score))
        marker = " ← LOW" if score < 5 else ""
        print(f"  {label:<22} {bar} {score:>5.1f}/10 (x{weight:.0%}){marker}")

    print(f"{'─'*55}")
    total = result.get("total_score", 0)
    print(f"  {'TOTAL':<22} {'':>10} {total:>5.1f}/100")
    print(f"{'─'*55}")

    issues = result.get("issues", [])
    if issues:
        print(f"\n Issues ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")

    print(f"{'='*55}\n")


def print_eval_history():
    """Eval 히스토리 추이 출력."""
    history_path = os.path.join(config.REPORTS_DIR, "eval_history.jsonl")
    if not os.path.exists(history_path):
        print("히스토리 없음")
        return

    entries = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        print("히스토리 없음")
        return

    print(f"\n{'='*45}")
    print(f" Eval Score History")
    print(f"{'='*45}")

    for i, entry in enumerate(entries):
        ts = entry.get("timestamp", "?")[:19]
        score = entry.get("total_score", 0)
        delta = ""
        if i > 0:
            prev = entries[i - 1].get("total_score", 0)
            diff = score - prev
            delta = f" ({'+' if diff >= 0 else ''}{diff:.1f})"

        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        print(f"  #{i+1} {ts}  {bar} {score:>5.1f}{delta}")

    print(f"{'='*45}\n")
