"""LaTeX → plain text 변환."""

from __future__ import annotations

import os
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from schema import PARQUET_SCHEMA
from utils import detect_language, load_all_parquet, save_df_to_parquet, setup_logger

logger = setup_logger("latex_cleaner", config.LOGS_DIR)

# LaTeX 그리스 문자 → 유니코드 매핑
_GREEK_MAP = {
    "alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "delta": "\u03b4",
    "epsilon": "\u03b5", "zeta": "\u03b6", "eta": "\u03b7", "theta": "\u03b8",
    "iota": "\u03b9", "kappa": "\u03ba", "lambda": "\u03bb", "mu": "\u03bc",
    "nu": "\u03bd", "xi": "\u03be", "pi": "\u03c0", "rho": "\u03c1",
    "sigma": "\u03c3", "tau": "\u03c4", "upsilon": "\u03c5", "phi": "\u03c6",
    "chi": "\u03c7", "psi": "\u03c8", "omega": "\u03c9",
    "Gamma": "\u0393", "Delta": "\u0394", "Theta": "\u0398", "Lambda": "\u039b",
    "Xi": "\u039e", "Pi": "\u03a0", "Sigma": "\u03a3", "Phi": "\u03a6",
    "Psi": "\u03a8", "Omega": "\u03a9",
    # 추가 그리스 문자 변형
    "varepsilon": "\u03b5", "varphi": "\u03c6", "varrho": "\u03c1",
    "vartheta": "\u03b8", "varsigma": "\u03c2",
    # 수학 기호
    "infty": "\u221e", "nabla": "\u2207", "partial": "\u2202",
    "int": "\u222b", "sum": "\u2211", "prod": "\u220f",
    "approx": "\u2248", "neq": "\u2260", "leq": "\u2264", "geq": "\u2265",
    "pm": "\u00b1", "times": "\u00d7", "cdot": "\u00b7", "sim": "~",
    "rightarrow": "\u2192", "leftarrow": "\u2190", "Rightarrow": "\u21d2",
    "Leftarrow": "\u21d0", "leftrightarrow": "\u2194",
    "sqrt": "\u221a", "forall": "\u2200", "exists": "\u2203",
    "in": "\u2208", "subset": "\u2282", "cup": "\u222a", "cap": "\u2229",
    "star": "\u2605", "circ": "\u2218", "bullet": "\u2022",
    "langle": "\u27e8", "rangle": "\u27e9",
    "hbar": "\u210f", "ell": "\u2113",
    "prime": "\u2032",
    # 삼각함수/수학함수 (텍스트로 유지)
    "cos": "cos", "sin": "sin", "tan": "tan", "log": "log", "exp": "exp",
    "ln": "ln", "max": "max", "min": "min", "lim": "lim",
    # 단위/특수
    "AA": "\u00c5",  # 옹스트롬
    "deg": "\u00b0",  # 도
}

def _replace_greek_letters(text: str) -> str:
    """LaTeX 그리스 문자/수학 기호를 유니코드로 변환."""
    for cmd, uni in _GREEK_MAP.items():
        text = text.replace(f"\\{cmd}", uni)
    return text


def clean_latex(raw_tex: str) -> str:
    """LaTeX 소스를 plain text로 변환."""
    text = raw_tex

    # 1. 프리앰블 제거 (\documentclass ~ \begin{document})
    match = re.search(r"\\begin\{document\}", text)
    if match:
        text = text[match.end():]

    match = re.search(r"\\end\{document\}", text)
    if match:
        text = text[:match.start()]

    # 2. 주석 제거 (% 로 시작, \% 제외)
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    # 3. figure 환경 제거
    text = re.sub(
        r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "", text, flags=re.DOTALL,
    )

    # 4. table 환경 → 캡션만 보존
    def _table_replacer(m):
        caption = re.search(r"\\caption\{([^}]*)\}", m.group(0))
        return f"[TABLE: {caption.group(1)}]" if caption else ""

    text = re.sub(
        r"\\begin\{table\*?\}.*?\\end\{table\*?\}",
        _table_replacer, text, flags=re.DOTALL,
    )

    # 5. 블록 수식 → [EQUATION]
    for env in ["equation", "equation*", "align", "align*", "eqnarray", "eqnarray*",
                 "gather", "gather*", "multline", "multline*", "displaymath"]:
        pattern = rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}"
        text = re.sub(pattern, "[EQUATION]", text, flags=re.DOTALL)

    text = re.sub(r"\\\[.*?\\\]", "[EQUATION]", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", "[EQUATION]", text, flags=re.DOTALL)

    # 6. 인라인 수식 $...$ → 내용 유지 후 수학 기호 정리
    text = re.sub(r"\$([^$]+?)\$", r"\1", text)

    # 6a. 그리스 문자 → 유니코드
    text = _replace_greek_letters(text)

    # 6b. 수학 표기 정리: 상하첨자, 분수 등
    text = re.sub(r"\^{([^}]*)}", r"\1", text)    # ^{2} → 2
    text = re.sub(r"_{([^}]*)}", r"\1", text)     # _{i} → i
    text = re.sub(r"\^(\w)", r"\1", text)          # ^2 → 2
    text = re.sub(r"_(\w)", r"\1", text)           # _i → i
    text = re.sub(r"\\frac{([^}]*)}{([^}]*)}", r"\1/\2", text)  # \frac{a}{b} → a/b

    # 7. 섹션 제목 → 텍스트
    text = re.sub(r"\\(?:sub)*section\*?\{([^}]*)\}", r"\n\n\1\n\n", text)
    text = re.sub(r"\\paragraph\*?\{([^}]*)\}", r"\n\1\n", text)

    # 8. 텍스트 서식 → 내용만
    for cmd in ["textbf", "textit", "textrm", "texttt", "textsf", "textsc",
                 "emph", "underline", "textup", "mbox", "mathrm", "mathbf",
                 "mathit", "mathcal", "boldsymbol",
                 "overline", "underline", "tilde", "hat", "bar", "vec",
                 "widetilde", "widehat", "overrightarrow"]:
        text = re.sub(rf"\\{cmd}\{{([^}}]*)\}}", r"\1", text)

    # 8a. 옛 LaTeX2e 이전 서식 (인자 없는 선언형)
    text = re.sub(r"\\(?:bf|it|rm|tt|sf|sc|sl|em)\b", "", text)

    # 9. 참조 제거
    text = re.sub(r"\\(?:cite|citep|citet|citealt|citeauthor|citenum)\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:ref|eqref|pageref|label|autoref|cref|Cref|nameref)\{[^}]*\}", "", text)

    # 10. \footnote → 내용 유지
    text = re.sub(r"\\footnote\{([^}]*)\}", r" (\1)", text)

    # 11. 리스트 환경
    text = re.sub(r"\\begin\{(?:itemize|enumerate|description)\}", "", text)
    text = re.sub(r"\\end\{(?:itemize|enumerate|description)\}", "", text)
    text = re.sub(r"\\item\s*(\[[^\]]*\])?\s*", "\n- ", text)

    # 12. abstract 환경 태그만 제거 (내용 보존)
    text = re.sub(r"\\begin\{abstract\}", "", text)
    text = re.sub(r"\\end\{abstract\}", "", text)

    # 13. 기타 환경 태그 제거 (내용 보존)
    text = re.sub(r"\\begin\{[^}]*\}", "", text)
    text = re.sub(r"\\end\{[^}]*\}", "", text)

    # 14. 레이아웃 명령어 제거
    text = re.sub(r"\\(?:newline|linebreak|pagebreak|newpage|clearpage|noindent|maketitle)\b", "", text)
    text = re.sub(r"\\(?:small|large|Large|LARGE|huge|Huge|tiny|normalsize|footnotesize|scriptsize)\b", "", text)
    text = re.sub(r"\\(?:vspace|hspace)\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:centering|raggedright|raggedleft)\b", "", text)
    text = re.sub(r"\\(?:def|newcommand|renewcommand|DeclareMathOperator)\b[^\n]*", "", text)

    # 15. 빈 중괄호, 남은 단순 명령어
    text = re.sub(r"\{\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\b", "", text)  # 남은 \command 제거

    # 16. 특수 문자
    text = text.replace("~", " ")
    text = text.replace("\\&", "&")
    text = text.replace("\\%", "%")
    text = text.replace("\\$", "$")
    text = text.replace("\\#", "#")
    text = text.replace("\\_", "_")
    text = text.replace("\\\\", "\n")
    text = text.replace("``", '"')
    text = text.replace("''", '"')

    # 17. 중괄호 정리
    text = re.sub(r"[{}]", "", text)

    # 18. 연속 공백/줄바꿈 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def clean_pdf_text(raw_text: str) -> str:
    """PDF 추출 텍스트 정제 (PyMuPDF 결과물 기준)."""
    text = raw_text

    # 1. 하이픈 줄바꿈 복원 (PDF 컬럼 분리 아티팩트)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 2. 단어 중간 줄바꿈 제거 (한 문장이 두 줄로 나뉜 경우)
    text = re.sub(r"(?<=[a-z,;])\n(?=[a-z])", " ", text)

    # 3. 참고문헌 섹션 제거 (References 이후)
    text = re.sub(
        r"\n(References|Bibliography|REFERENCES|BIBLIOGRAPHY)\n.*",
        "", text, flags=re.DOTALL
    )

    # 4. 페이지 번호 / 헤더/푸터 패턴 제거
    # 예: "12\n", "Page 3 of 10", "3 von 12" (독일어), arXiv 헤더
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\s+von\s+\d+\s*$", "", text, flags=re.MULTILINE)  # 독일어
    text = re.sub(r"^Page\s+\d+\s+of\s+\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"arXiv:\S+\s+\[[\w.-]+\]\s+\d{1,2}\s+\w+\s+\d{4}", "", text)
    # 날짜만 있는 줄 (DD.MM.YY 형식)
    text = re.sub(r"^\d{2}\.\d{2}\.\d{2,4}\s*$", "", text, flags=re.MULTILINE)

    # 5. 연속 공백/줄바꿈 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def count_tokens_approx(text: str) -> int:
    """Whitespace 기반 대략적 토큰 수."""
    if not text:
        return 0
    return len(text.split())


def clean_all_records(data_dir: str | None = None, batch_id: str | None = None):
    """data_dir의 모든 arXiv 레코드에 clean_text 적용."""
    data_dir = data_dir or config.DATA_DIR
    df = load_all_parquet(data_dir, source="arxiv")

    if df.empty:
        logger.warning("arXiv 데이터 없음")
        return 0

    cleaned = 0

    # full_text가 있는 레코드: 포맷에 따라 정제
    has_text = df["has_full_text"] == True
    for idx in df[has_text].index:
        raw = df.at[idx, "full_text"]
        if raw and pd.notna(raw):
            fmt = df.at[idx, "full_text_format"] if "full_text_format" in df.columns else "latex"
            if fmt == "pdf_text":
                ct = clean_pdf_text(raw)
            else:
                ct = clean_latex(raw)
            df.at[idx, "clean_text"] = ct
            df.at[idx, "token_count_approx"] = count_tokens_approx(ct)
            df.at[idx, "language"] = detect_language(ct[:1000])
            cleaned += 1

    # abstract-only 레코드: clean_text = abstract
    abstract_only = (df["full_text_status"] == "abstract_only") | (~has_text)
    for idx in df[abstract_only].index:
        ab = df.at[idx, "abstract"]
        if ab and pd.notna(ab):
            df.at[idx, "clean_text"] = ab
            df.at[idx, "token_count_approx"] = count_tokens_approx(ab)

    # 저장 (공통 함수 사용)
    bid = batch_id or "cleaned"
    save_df_to_parquet(df, data_dir, "arxiv", bid)

    logger.info(f"LaTeX 클리닝 완료: {cleaned}건 full_text 변환")
    return cleaned
