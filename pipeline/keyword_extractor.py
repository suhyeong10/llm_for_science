"""키워드 추출 모듈.

TF-IDF 기반 기본 키워드 추출 (LLM 없이도 동작).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from math import log

import pandas as pd

from . import config
from .utils import load_all_parquet, setup_logger

logger = setup_logger("keywords", config.LOGS_DIR)

# 화학 분야 불용어 (일반 영어 불용어 + 학술 논문 공통 표현)
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "of", "in",
    "to", "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "and", "but", "or", "nor", "not", "no", "so", "if", "than", "that",
    "this", "these", "those", "it", "its", "we", "our", "they", "their",
    "them", "he", "she", "her", "him", "his", "my", "your", "which",
    "who", "whom", "what", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "also", "very", "just", "because", "while",
    # 학술 공통
    "results", "study", "using", "based", "show", "shows", "shown",
    "method", "methods", "approach", "paper", "work", "present",
    "proposed", "propose", "new", "used", "use", "two", "one", "first",
    "however", "well", "different", "respectively", "obtained", "found",
    "order", "case", "three", "number", "high", "low", "large", "small",
    "good", "possible", "important", "particular", "general", "given",
    "several", "many", "various", "within", "upon", "thus", "hence",
    "therefore", "moreover", "furthermore", "addition", "recently",
}


def extract_keywords(
    data_dir: str | None = None,
    top_n: int = 200,
    min_doc_freq: int = 3,
) -> list[tuple[str, float]]:
    """TF-IDF 기반 키워드 추출."""
    data_dir = data_dir or config.DATA_DIR
    df = load_all_parquet(data_dir)

    if df.empty:
        logger.warning("데이터 없음")
        return []

    # abstract 기반 (가장 넓은 커버리지)
    texts = df["abstract"].dropna().tolist()
    if not texts:
        logger.warning("abstract 없음")
        return []

    logger.info(f"키워드 추출: {len(texts)}개 abstract 분석 중...")

    # 토큰화
    doc_tokens = []
    for text in texts:
        tokens = _tokenize(text)
        doc_tokens.append(tokens)

    # Document frequency
    doc_freq = Counter()
    for tokens in doc_tokens:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1

    # TF-IDF 계산
    n_docs = len(doc_tokens)
    tfidf_scores = Counter()

    for tokens in doc_tokens:
        tf = Counter(tokens)
        n_tokens = len(tokens)
        for token, count in tf.items():
            if doc_freq[token] < min_doc_freq:
                continue
            tf_val = count / n_tokens
            idf_val = log(n_docs / (1 + doc_freq[token]))
            tfidf_scores[token] += tf_val * idf_val

    # 정렬 및 상위 N개
    keywords = tfidf_scores.most_common(top_n)

    # 저장
    _save_keywords(keywords, data_dir)

    logger.info(f"키워드 추출 완료: {len(keywords)}개")
    return keywords


def _tokenize(text: str) -> list[str]:
    """텍스트 토큰화 (소문자, 영문+숫자만, 불용어 제거)."""
    text = text.lower()
    tokens = re.findall(r"[a-z][a-z0-9-]{2,}", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _save_keywords(keywords: list[tuple[str, float]], data_dir: str):
    """키워드를 markdown 파일로 저장."""
    os.makedirs(config.KEYWORDS_DIR, exist_ok=True)
    path = os.path.join(config.KEYWORDS_DIR, "chemistry_keywords.md")

    lines = [
        "# 화학 논문 키워드 (자동 추출)",
        "",
        f"추출 시각: {__import__('datetime').datetime.now().isoformat()}",
        f"방법: TF-IDF (abstract 기반)",
        f"총 키워드 수: {len(keywords)}",
        "",
        "| 순위 | 키워드 | TF-IDF 점수 |",
        "|------|--------|-------------|",
    ]

    for i, (kw, score) in enumerate(keywords, 1):
        lines.append(f"| {i} | {kw} | {score:.4f} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"키워드 저장: {path}")
