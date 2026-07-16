# 프로젝트 상세 가이드

`llm_for_science`는 과학 분야 논문 코퍼스를 구축하고 full fine-tune CPT 실험을 실행하기 위한 저장소입니다.

현재 데이터 파이프라인의 수집/파싱 기준은 담당 범위인 화학 분야에 맞춰져 있습니다. 다만 학습 대상은 화학 단독이 아니라 과학 분야 코퍼스 전체로 확장하는 것을 전제로 합니다.

## 현재 상태

### 데이터 구축 결과 (2026-04-09 기준)

| 단계 | 상태 | 결과 |
|------|------|------|
| arXiv 메타데이터 수집 | 완료 | 254,376건 (9개 카테고리, 2000~2025) |
| arXiv LaTeX/PDF 다운로드 | 완료 | 167,895건 LaTeX + 48,652건 PDF |
| LaTeX -> plain text 클리닝 | 완료 | 잔존율 0.000% |
| PDF 배치 파싱 | 완료 | 48,630건 성공 / 22건 실패 |
| Semantic Scholar 보충 | 미시작 | API key 및 범위 합의 필요 |
| 중복 제거 | 완료 | hash 30,140건 + fuzzy 중복 제거 |
| DuckDB 인덱싱 | 완료 | `data/index.db` |
| Eval 점수화 | 완료 | 96.8/100 |

### 본문 확보 현황

| source_type | 건수 | 설명 |
|-------------|------|------|
| `arxiv_latex` | 167,895 | LaTeX 원본 |
| `arxiv_pdf_pymupdf4llm` | 54,728 | PDF -> Markdown 파싱 |
| `arxiv_pdf_saved` | 415 | PDF 파싱 실패 또는 raw PDF 보관 |
| `NULL` | 1,089 | abstract만 보유 |
| 합계 | 224,161 | Full-text 확보율 99.3% |

대용량 `data/`와 `outputs/` 산출물은 로컬/외부 산출물이며 저장소에 모두 포함되어 있지 않을 수 있습니다.

## 저장소 구조

```text
.
├── main.py                         # 데이터 파이프라인 CLI
├── pipeline/                       # 데이터 수집/파싱/평가 구현
│   ├── config.py                   # 카테고리, rate limit, 경로 설정
│   ├── schema.py                   # Parquet 스키마 + PaperRecord
│   ├── arxiv_crawler.py            # arXiv 메타데이터/LaTeX/PDF 수집
│   ├── semantic_scholar_crawler.py # Semantic Scholar 보충 수집
│   ├── latex_cleaner.py            # LaTeX -> plain text 클리닝
│   ├── batch_parse_pdfs.py         # PDF 배치 파싱
│   ├── dedup.py                    # hash/fuzzy 중복 제거
│   ├── eval_scorer.py              # 데이터셋 품질 점수화
│   └── scripts/                    # 운영 보조 스크립트
├── training/
│   └── full_cpt/                   # full fine-tune CPT 학습 코드/설정/스크립트
├── docs/                           # 상세 문서
├── keywords/                       # 도메인 키워드
├── data/                           # 로컬 데이터 산출물
└── logs/                           # 실행 로그
```

## 데이터 파이프라인

```text
arXiv categories
  -> metadata crawl
  -> LaTeX/PDF download
  -> LaTeX cleaning
  -> PDF parsing
  -> optional Semantic Scholar supplement
  -> dedup
  -> DuckDB index
  -> eval/stats
```

주요 명령:

```bash
python main.py arxiv --years 2000-2025
python main.py arxiv-latex --resume --workers 8
python main.py clean
python -m pipeline.batch_parse_pdfs --backend pymupdf4llm --workers 4 --chunk 500
python main.py dedup
python main.py index
python main.py eval
python main.py stats
```

PDF 파싱 상세는 [PDF_PARSING_GUIDE.md](PDF_PARSING_GUIDE.md)를 참고하세요.

## Full-CPT 학습

이 저장소의 현재 학습 코드는 Qwen 계열 causal LM을 전체 파라미터로 CPT하는 full fine-tune 경로입니다.

실행 환경 확인:

```bash
python -m training.full_cpt.train \
  --config training/full_cpt/tests/smoke_cpt_config.yaml \
  --dry_run
```

실데이터 CPT 실행:

```bash
bash training/full_cpt/scripts/run_cpt_train.sh \
  training/full_cpt/configs/qwen35_4b_cpt_h100.yaml
```

`training/full_cpt/configs/qwen35_4b_cpt_h100.yaml`은 다음 JSONL 파일이 준비되어 있다고 가정합니다.

```text
data/processed/corpus_v1/train.jsonl
data/processed/corpus_v1/valid.jsonl
```

기본 설정은 W&B 로깅(`report_to: ["wandb"]`)을 사용합니다. W&B를 쓰지 않는 환경에서는 설정 파일에서 `report_to: []`로 바꿔 실행하세요.

## 데이터 산출물

```text
data/
├── arxiv/
│   ├── year=YYYY/
│   └── pdfs/
├── merged/
├── processed/
├── checkpoints/
└── index.db
```

현재 `main.py index`는 `data/arxiv/**/*.parquet`를 대상으로 DuckDB 뷰를 생성합니다. dedup 이후의 `data/merged/`를 쿼리 대상으로 삼으려면 인덱싱 로직 확장이 필요합니다.

DuckDB 쿼리 예시:

```bash
python main.py query "SELECT full_text_source_type, COUNT(*) FROM papers GROUP BY 1 ORDER BY 2 DESC"
python main.py query "SELECT * FROM papers WHERE title LIKE '%catalyst%' LIMIT 10" --output results.csv
```

## 품질 평가

최종 Eval Score: 96.8/100

```text
Schema Completeness     10.0/10 (x15%)
Full-text Coverage       8.8/10 (x20%)
LaTeX Parse Quality     10.0/10 (x20%)
Metadata Accuracy       10.0/10 (x10%)
Dedup Effectiveness     10.0/10 (x10%)
Language Purity         10.0/10 (x5%)
Token Distribution       9.4/10 (x10%)
Hash Uniqueness         10.0/10 (x10%)
```

## 현재 담당 arXiv 카테고리

현재 파이프라인은 화학 담당 데이터를 수집하기 위해 아래 카테고리를 사용합니다. arXiv에는 전용 Chemistry 카테고리가 없어 화학 관련 논문은 physics, cond-mat, q-bio, cs 등에 분산되어 있습니다.

| 카테고리 | 분야 |
|----------|------|
| `physics.chem-ph` | 화학물리 |
| `cond-mat.mtrl-sci` | 재료과학 |
| `physics.atm-clus` | 원자/분자 클러스터 |
| `cond-mat.soft` | 연성물질 |
| `physics.comp-ph` | 계산물리/계산화학 |
| `physics.bio-ph` | 생물물리화학 |
| `q-bio.BM` | 생체분자 |
| `physics.atom-ph` | 원자물리/분광학 |
| `cs.CE` | 화학정보학 |
