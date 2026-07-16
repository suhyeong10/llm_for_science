#!/usr/bin/env python3
"""논문 데이터 파이프라인 CLI.

사용법:
    python main.py arxiv [--categories CAT...] [--years START-END] [--max-results N] [--resume] [--batch-id ID]
    python main.py arxiv-latex [--max-items N] [--resume] [--workers N]
    python main.py parse-pdf [--backend docling|pymupdf|mineru] [--max-items N] [--workers N]
    python main.py parse-pdf --compare [--sample-size N]
    python main.py s2 [--max-results N] [--resume] [--batch-id ID]
    python main.py clean
    python main.py dedup
    python main.py stats
    python main.py keywords [--top-n N]
    python main.py eval [--sample-size N]
    python main.py eval-history
    python main.py all [--max-results N] [--batch-id ID]
"""

from __future__ import annotations

import argparse
import os
import sys

from pipeline import config


def cmd_arxiv(args):
    """arXiv 메타데이터 수집."""
    from pipeline.arxiv_crawler import ArxivCrawler

    batch_id = config.generate_batch_id(args.batch_id)
    crawler = ArxivCrawler(batch_id=batch_id, resume=args.resume)

    cats = args.categories or config.ARXIV_CATEGORIES
    yr = _parse_years(args.years) if args.years else config.YEAR_RANGE

    n = crawler.crawl_metadata(
        categories=cats, year_range=yr, max_results=args.max_results
    )
    print(f"\narXiv 메타데이터 수집 완료: {n}건")
    return batch_id


def cmd_arxiv_latex(args):
    """arXiv LaTeX 소스 다운로드 (PDF 폴백 옵션 포함)."""
    from pipeline.arxiv_crawler import ArxivCrawler

    batch_id = config.generate_batch_id(args.batch_id)
    crawler = ArxivCrawler(batch_id=batch_id, resume=True)
    stats = crawler.download_latex_sources(
        max_items=args.max_items,
        resume=args.resume,
        pdf_fallback=args.pdf_fallback,
        workers=args.workers,
    )
    latex_only = stats["success"] - stats["pdf_fallback"]
    print(f"\n다운로드 완료:")
    print(f"  LaTeX 성공: {latex_only:,}건")
    print(f"  PDF 폴백:   {stats['pdf_fallback']:,}건")
    print(f"  실패:       {stats['failed']:,}건")
    print(f"  Skip:       {stats['skipped']:,}건")


def cmd_parse_pdf(args):
    """저장된 PDF 배치 파싱."""
    from pipeline.pdf_parser import compare_pdf_backends, parse_pdfs_batch

    if args.compare:
        compare_pdf_backends(
            sample_size=args.sample_size or 5,
            backends=args.backends or ["pymupdf", "docling"],
        )
        return

    stats = parse_pdfs_batch(
        backend=args.backend,
        max_items=args.max_items,
        workers=args.workers,
        overwrite_pymupdf=args.overwrite_pymupdf,
    )
    print(f"\nPDF 파싱 완료 (backend={args.backend}):")
    print(f"  파싱 성공: {stats['parsed']:,}건")
    print(f"  실패:      {stats['failed']:,}건")


def cmd_s2(args):
    """Semantic Scholar 수집."""
    from pipeline.semantic_scholar_crawler import SemanticScholarCrawler

    batch_id = config.generate_batch_id(args.batch_id)
    crawler = SemanticScholarCrawler(batch_id=batch_id, resume=args.resume)
    n = crawler.crawl(max_results=args.max_results)
    print(f"\nSemantic Scholar 수집 완료: {n}건")


def cmd_clean(args):
    """LaTeX → plain text 변환."""
    from pipeline.latex_cleaner import clean_all_records

    n = clean_all_records()
    print(f"\nLaTeX 클리닝 완료: {n}건 변환")


def cmd_dedup(args):
    """중복 제거."""
    from pipeline.dedup import deduplicate

    stats = deduplicate()
    print(f"\n중복 제거 완료:")
    print(f"  원본: {stats['original']:,}건")
    print(f"  중복: {stats['duplicates']:,}건 (hash: {stats['hash_duplicates']}, fuzzy: {stats['fuzzy_duplicates']})")
    print(f"  최종: {stats['final']:,}건")


def cmd_stats(args):
    """품질 리포트."""
    from pipeline.stats_report import generate_report, print_report

    report = generate_report()
    print_report(report)


def cmd_keywords(args):
    """키워드 추출."""
    from pipeline.keyword_extractor import extract_keywords

    keywords = extract_keywords(top_n=args.top_n)
    print(f"\n키워드 추출 완료: {len(keywords)}개")
    print("\nTop 20:")
    for i, (kw, score) in enumerate(keywords[:20], 1):
        print(f"  {i:>3}. {kw:<30} {score:.4f}")


def cmd_eval(args):
    """Eval 점수화."""
    from pipeline.eval_scorer import run_eval

    run_eval(sample_size=args.sample_size)


def cmd_eval_history(args):
    """Eval 히스토리 출력."""
    from pipeline.eval_scorer import print_eval_history

    print_eval_history()


def cmd_index(args):
    """DuckDB로 parquet 인덱스(뷰) 생성."""
    import duckdb, glob as _glob
    from pipeline import config

    db_path = os.path.join(config.DATA_DIR, "index.db")
    parquet_pattern = os.path.join(config.DATA_DIR, "arxiv", "**", "*.parquet")

    # 읽을 수 있는 parquet 파일만 확인
    files = sorted(_glob.glob(parquet_pattern, recursive=True))
    readable = []
    for f in files:
        try:
            import pyarrow.parquet as pq
            pq.read_schema(f)
            readable.append(f)
        except Exception:
            pass
    print(f"인덱싱 대상: {len(readable)}/{len(files)} parquet 파일")

    con = duckdb.connect(db_path)

    # 메타데이터 전용 뷰 (full_text 제외 — 빠른 쿼리용)
    meta_cols = ", ".join([
        "arxiv_id", "title", "abstract", "year", "categories", "authors",
        "has_full_text", "full_text_source_type", "full_text_status",
        "full_text_format", "token_count_approx", "is_duplicate",
        "content_hash", "citation_count", "language", "license",
        "pdf_path", "crawl_date", "source",
    ])
    con.execute(f"""
        CREATE OR REPLACE VIEW papers AS
        SELECT {meta_cols}
        FROM read_parquet({readable!r})
    """)

    # 전체 뷰 (full_text 포함)
    con.execute(f"""
        CREATE OR REPLACE VIEW papers_full AS
        SELECT * FROM read_parquet({readable!r})
    """)

    count = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"\n✅ 인덱스 생성 완료: {db_path}")
    print(f"  총 레코드: {count:,}건")
    print(f"\n사용법:")
    print(f"  import duckdb; con = duckdb.connect('{db_path}')")
    print(f"  con.sql(\"SELECT full_text_source_type, COUNT(*) FROM papers GROUP BY 1\").show()")
    print(f"  con.sql(\"SELECT * FROM papers WHERE year=2020 LIMIT 5\").show()")
    print(f"\n  # 파서별 통계")
    con.sql("SELECT full_text_source_type, COUNT(*) as cnt FROM papers GROUP BY 1 ORDER BY 2 DESC").show()
    con.close()


def cmd_query(args):
    """DuckDB로 parquet 직접 쿼리."""
    import duckdb
    from pipeline import config

    db_path = os.path.join(config.DATA_DIR, "index.db")
    if not os.path.exists(db_path):
        print(f"인덱스 없음. 먼저 실행: python main.py index")
        return

    con = duckdb.connect(db_path)
    sql = args.sql
    print(f"쿼리: {sql}\n")
    try:
        result = con.sql(sql)
        result.show(max_rows=args.limit)
        if args.output:
            result.df().to_csv(args.output, index=False)
            print(f"\nCSV 저장: {args.output}")
    except Exception as e:
        print(f"오류: {e}")
    con.close()


def cmd_all(args):
    """전체 파이프라인 실행."""
    print("=" * 50)
    print(" 전체 파이프라인 시작")
    print("=" * 50)

    # Step 1: arXiv 메타데이터
    print("\n[1/6] arXiv 메타데이터 수집...")
    args.categories = None
    args.years = None
    args.resume = False
    batch_id = cmd_arxiv(args)
    args.batch_id = batch_id

    # Step 2: arXiv LaTeX
    print("\n[2/6] arXiv LaTeX 다운로드...")
    args.max_items = args.max_results
    cmd_arxiv_latex(args)

    # Step 3: LaTeX 클리닝
    print("\n[3/6] LaTeX 클리닝...")
    cmd_clean(args)

    # Step 4: Semantic Scholar (실패해도 계속 진행)
    print("\n[4/6] Semantic Scholar 수집...")
    try:
        cmd_s2(args)
    except Exception as e:
        print(f"  S2 수집 실패 (건너뜀): {e}")

    # Step 5: 중복 제거
    print("\n[5/6] 중복 제거...")
    cmd_dedup(args)

    # Step 6: Eval
    print("\n[6/6] 품질 평가...")
    args.sample_size = None
    cmd_eval(args)

    print("\n" + "=" * 50)
    print(" 전체 파이프라인 완료!")
    print("=" * 50)


def _parse_years(year_str: str) -> tuple[int, int]:
    """'2000-2022' → (2000, 2022)."""
    parts = year_str.split("-")
    return (int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="과학 논문 데이터 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="서브커맨드")

    # arxiv
    p = subparsers.add_parser("arxiv", help="arXiv 메타데이터 수집")
    p.add_argument("--categories", nargs="+", help="arXiv 카테고리")
    p.add_argument("--years", help="연도 범위 (예: 2000-2022)")
    p.add_argument("--max-results", type=int, help="최대 수집 건수")
    p.add_argument("--resume", action="store_true", help="이전 크롤링 이어서")
    p.add_argument("--batch-id", help="배치 식별자")

    # arxiv-latex
    p = subparsers.add_parser("arxiv-latex", help="arXiv LaTeX 소스 다운로드")
    p.add_argument("--max-items", type=int, help="최대 다운로드 건수")
    p.add_argument("--resume", action="store_true", help="이미 다운로드된 ID skip")
    p.add_argument("--pdf-fallback", action="store_true", default=True,
                   help="LaTeX 실패 시 PDF로 폴백 (기본: True)")
    p.add_argument("--no-pdf-fallback", dest="pdf_fallback", action="store_false",
                   help="PDF 폴백 비활성화")
    p.add_argument("--workers", type=int, default=8,
                   help="병렬 다운로드 워커 수 (기본: 8, GlobalRateLimiter가 속도 제어)")
    p.add_argument("--batch-id", help="배치 식별자")

    # parse-pdf
    p = subparsers.add_parser("parse-pdf", help="저장된 PDF 배치 파싱 (고품질 파서)")
    p.add_argument("--backend", default="docling",
                   choices=["pymupdf", "docling", "mineru"],
                   help="파서 선택 (기본: docling)")
    p.add_argument("--max-items", type=int, help="최대 처리 건수")
    p.add_argument("--workers", type=int, default=2,
                   help="병렬 워커 수 (기본: 2)")
    p.add_argument("--overwrite-pymupdf", action="store_true",
                   help="기존 PyMuPDF 결과도 덮어쓰기")
    p.add_argument("--compare", action="store_true",
                   help="파서 품질 비교 모드")
    p.add_argument("--backends", nargs="+",
                   help="비교할 백엔드 목록 (기본: pymupdf docling)")
    p.add_argument("--sample-size", type=int, default=5,
                   help="비교 샘플 수")

    # s2
    p = subparsers.add_parser("s2", help="Semantic Scholar 수집")
    p.add_argument("--max-results", type=int, help="최대 수집 건수")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-id", help="배치 식별자")

    # clean
    subparsers.add_parser("clean", help="LaTeX → plain text 변환")

    # dedup
    subparsers.add_parser("dedup", help="중복 제거")

    # stats
    subparsers.add_parser("stats", help="품질 리포트")

    # keywords
    p = subparsers.add_parser("keywords", help="키워드 추출")
    p.add_argument("--top-n", type=int, default=200, help="추출할 키워드 수")

    # eval
    p = subparsers.add_parser("eval", help="Eval 점수화 (Karpathy Loop)")
    p.add_argument("--sample-size", type=int, help="샘플 크기")

    # eval-history
    subparsers.add_parser("eval-history", help="Eval 히스토리 추이")

    # index
    subparsers.add_parser("index", help="DuckDB 인덱스 생성 (parquet → SQL 뷰)")

    # query
    p = subparsers.add_parser("query", help="DuckDB SQL 쿼리 실행")
    p.add_argument("sql", help="SQL 쿼리 문자열")
    p.add_argument("--limit", type=int, default=20, help="출력 행 수 제한 (기본 20)")
    p.add_argument("--output", help="결과를 CSV로 저장할 경로")

    # all
    p = subparsers.add_parser("all", help="전체 파이프라인 실행")
    p.add_argument("--max-results", type=int, help="소스별 최대 수집 건수")
    p.add_argument("--batch-id", help="배치 식별자")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "arxiv": cmd_arxiv,
        "arxiv-latex": cmd_arxiv_latex,
        "parse-pdf": cmd_parse_pdf,
        "s2": cmd_s2,
        "clean": cmd_clean,
        "dedup": cmd_dedup,
        "stats": cmd_stats,
        "keywords": cmd_keywords,
        "eval": cmd_eval,
        "eval-history": cmd_eval_history,
        "index": cmd_index,
        "query": cmd_query,
        "all": cmd_all,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
