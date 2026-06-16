# PDF 파싱 가이드

논문 PDF를 텍스트(Markdown)로 변환하는 배치 파싱 도구 사용법.

---

## Quick Start

```bash
# 1. 설치
pip install pymupdf4llm

# 2. 파싱 실행 (PDF 폴더 지정)
python3 batch_parse_pdfs.py --pdf-dir /path/to/pdfs --workers 4

# 3. 결과 확인
python main.py index
python main.py query "SELECT full_text_source_type, COUNT(*) FROM papers GROUP BY 1"
```

---

## 지원 파서 3종

### 1. pymupdf4llm (권장 - CPU)

```bash
pip install pymupdf4llm
python3 batch_parse_pdfs.py --backend pymupdf4llm --workers 4 --chunk 500
```

- **속도**: 0.45초/페이지 (CPU)
- **수식**: 지원 안 됨 (수식이 텍스트로 흘러나옴)
- **장점**: 빠르고 안정적, 설치 간단
- **단점**: 수식 구조 완전 손실
- **48K PDF 소요**: ~12시간 (4 workers)

### 2. Docling + Formula Enrichment (CPU, 수식 필요 시)

```bash
pip install docling
python3 batch_parse_pdfs.py --backend docling --workers 2 --chunk 100
```

- **속도**: 14초/페이지 (수식 포함 시)
- **수식**: LaTeX 코드로 변환 (`$$...$$`)
- **장점**: 수식을 LaTeX로 추출, GPU 불필요
- **단점**: 매우 느림
- **48K PDF 소요**: ~56일 (비현실적)

### 3. MinerU (GPU 권장, 최고 품질)

```bash
# Python 3.10+ 필요, conda 환경 권장
conda create -n mineru python=3.12 -y && conda activate mineru
pip install "magic-pdf[full]"
# 모델 다운로드 필요 (HuggingFace: opendatalab/PDF-Extract-Kit-1.0)

python3 batch_parse_pdfs.py --backend mineru --workers 4 --chunk 100
```

- **속도**: 0.21초/페이지 (GPU) / 23초/페이지 (CPU)
- **수식**: LaTeX 코드로 변환, 정확도 CDM 0.968 (상용 Mathpix 수준)
- **장점**: 수식+표+레이아웃 모두 최고 품질
- **단점**: GPU 25GB VRAM 필요, 설치 복잡, AGPL 라이선스
- **48K PDF 소요**: ~1일 (GPU) / ~13일 (CPU)

---

## 파서 비교표

| 항목 | pymupdf4llm | Docling+수식 | MinerU (GPU) | MinerU (CPU) |
|------|-------------|-------------|-------------|-------------|
| **속도** | 0.45초/p | 14초/p | 0.21초/p | 23초/p |
| **수식 추출** | ❌ | ✅ LaTeX | ✅ LaTeX (최고) | ✅ LaTeX (최고) |
| **표 추출** | 기본 | 양호 | 최고 | 최고 |
| **레이아웃** | 기본 | 양호 (93.1%) | 최고 (97.5%) | 최고 |
| **Python** | 3.8+ | 3.8+ | 3.10+ | 3.10+ |
| **GPU** | 불필요 | 불필요 | 필요 (25GB) | 불필요 (느림) |
| **48K 소요** | **~12h** | ~56일 | **~1일** | ~13일 |
| **설치** | `pip install pymupdf4llm` | `pip install docling` | 복잡 (모델 다운로드) | 복잡 |
| **라이선스** | AGPL | Apache 2.0 | AGPL | AGPL |

---

## 실제 품질 비교 (동일 논문)

테스트 논문: `0704.3887` — "On the accurate evaluation of overlap integrals over Slater type orbitals" (6페이지, 수식 다수)

### 원본 PDF 수식 (Eq.1):

> S_{nlλ,n'l'λ}(p,t) = ∫ χ*_{nlm}(ζ,r̄_a) χ_{n'l'm}(ζ',r̄_b) dV

---

### pymupdf4llm 출력 (3초):

```
## S nl λ, n l ′ ′λ ( p t, ) = ∫ χ nlm ( ζ, r a ) χ n l m ′ ′ ( ζ, r b ) dV,    (1)

where 0 ≤ λ≤ l m, = ±λ, p = R ( ζ +ζ′ ), t = ( ζ −ζ′ ) /( ζ +ζ′ ), R ≡ R ab = r a − r b and

##### χ nlm ( ζ, r ) = ( 2 ζ ) n + 12 ⎡⎣ ( 2 n ) ! ⎤⎦ − 12 r n − 1 e −ζ r S lm (, θ ϕ )
```

**평가**: 수식이 텍스트로 흘러나옴. 위/아래 첨자 구조 완전 손실. `##`, `#####` 같은 가짜 마크다운 헤더로 오인식. LLM이 수식의 의미를 파악하기 어려움.

---

### Docling + Formula Enrichment 출력 (85초):

```
$$S _ { n l _ { \lambda } , n ^ { \prime } l ^ { \prime } } \left ( p , t \right ) =
  \int \chi _ { n l m } ^ { ^ { * } } \left ( \varsigma , \bar { r } _ { a } \right )
  \chi _ { n ^ { \prime } l m } \left ( \varsigma ^ { \prime } , \bar { r } _ { b } \right ) d V$$

$$\chi _ { n l m } \left ( \zeta , \bar { r } \right ) = \left ( 2 \zeta \right ) ^ { n + \frac { 1 } { 2 } }
  \left [ \left ( 2 n \right ) \right ] ^ { - \frac { 1 } { 2 } }
  r ^ { n - 1 } e ^ { - \zeta r } S _ { l m } ( \theta , \varphi )$$
```

**평가**: 완전한 LaTeX 수식으로 변환. 첨자, 분수, 적분 기호 모두 정확. 렌더링하면 원본과 동일. 단, 14초/페이지로 대규모 처리 비현실적.

---

### MinerU 출력 (137초, CPU):

```
$S _ { n l \lambda , n ^ { \prime } l ^ { \prime } \lambda } \left( p , t \right) =
  \int \chi _ { n l m } ^ { * } \left( \zeta , \vec { r } _ { a } \right)
  \chi _ { n ^ { \prime } l ^ { \prime } m } \left( \zeta ^ { \prime } , \vec { r } _ { b } \right) d V ,$

$\chi _ { n l m } \left( \zeta , \vec { r } \right) = \left( 2 \zeta \right) ^ { n + \frac { 1 } { 2 } }
  \left[ \left( 2 n \right) ! \right] ^ { - \frac 1 2 }
  r ^ { n - 1 } e ^ { - \zeta r } S _ { l m } ( \theta , \varphi ) .$
```

**평가**: Docling과 동일 수준의 LaTeX 수식 + 벡터 기호(`\vec{r}`)까지 정확히 인식. 전체적으로 Docling보다 디테일이 약간 더 정확. GPU에서는 0.21초/페이지로 압도적.

---

### 복잡한 수식 (Eq.3) 비교:

**pymupdf4llm**:
```
S nl λ n l ′ ′λ p t, = N nn ′ ( ) t g ( l λ, l λ ) F
```
→ 완전히 깨짐. Σ(summation), 첨자, 곱 기호 전부 손실.

**Docling + Formula**:
```
$$S _ { n l , \lambda ^ { \prime \prime } n } ( p , t ) = N _ { m ^ { \prime } m } ( t )
  \sum _ { \alpha = - \lambda } ^ { l } \sum _ { \beta = \lambda } ^ { ( 2 ) ^ { l } }
  g _ { \alpha \beta } ^ { 0 } ( l \lambda , l ^ { \prime } \lambda )
  \sum _ { q = 0 } ^ { \alpha + \beta } F _ { q } ( \alpha + \lambda , \beta - \lambda )$$
```
→ 이중 합(Σ)과 복잡한 첨자 구조까지 정확히 추출.

**MinerU (CPU)**:
```
$$S _ { n l \lambda , n ^ { \prime } \lambda ^ { \prime } } \big ( p , t \big ) =
  N _ { n n ^ { \prime } } ( t ) \sum _ { \alpha = - \lambda } ^ { l }
  { ^ { ( 2 ) } \sum _ { \beta = \lambda } ^ { l ^ { \prime } }
  { ^ { ( 2 ) } g _ { \alpha \beta } ^ { 0 } ( l \lambda , l ^ { \prime } \lambda )
  \sum _ { q = 0 } ^ { \alpha + \beta } { F _ { q } ( \alpha + \lambda , \beta - \lambda ) } } }$$
```
→ (2) 표기까지 정확하게 캡처. 가장 정확한 결과.

---

## 사용법 상세

### 기본 사용 (기존 크롤링 데이터)

```bash
# data/arxiv/pdfs/ 의 PDF를 파싱하여 parquet에 저장
python3 batch_parse_pdfs.py --backend pymupdf4llm --workers 4 --chunk 500
```

### 커스텀 PDF 폴더 지정

```bash
# 임의 폴더의 PDF 파싱
python3 batch_parse_pdfs.py --pdf-dir /path/to/my/pdfs --workers 4
```

### 파서 교체 (기존 결과 덮어쓰기)

```bash
# pymupdf4llm 결과를 MinerU로 교체
python3 batch_parse_pdfs.py \
  --backend mineru \
  --replace-source arxiv_pdf_pymupdf4llm \
  --workers 4

# pymupdf4llm 결과를 Docling으로 교체
python3 batch_parse_pdfs.py \
  --backend docling \
  --replace-source arxiv_pdf_pymupdf4llm \
  --workers 2
```

### Dry Run (실제 파싱 없이 대상 확인)

```bash
python3 batch_parse_pdfs.py --dry-run
# 출력: 파서: pymupdf4llm | 전체: 48,652 | 완료: 48,630 | 남은: 22
```

---

## CLI 옵션

```
python3 batch_parse_pdfs.py [OPTIONS]

옵션:
  --backend {pymupdf4llm,docling,mineru}   파서 선택 (기본: pymupdf4llm)
  --pdf-dir PATH                           PDF 폴더 경로 (기본: data/arxiv/pdfs/)
  --replace-source SOURCE_TYPE             교체 모드: 해당 source_type 레코드를 새 파서로 재파싱
  --workers N                              병렬 워커 수 (기본: 4)
  --chunk N                                저장 단위 (기본: 500, 크래시 시 최대 N건 손실)
  --dry-run                                실제 파싱 없이 대상 확인만
```

---

## 크래시 복구

### 1. 체크포인트 자동 이어받기

스크립트를 다시 실행하면 자동으로 이어서 처리합니다:

```bash
# 중간에 Ctrl+C로 종료해도 OK
python3 batch_parse_pdfs.py --workers 4
# → "파서: pymupdf4llm | 전체: 48,652 | 완료: 30,000 | 남은: 18,652"
```

체크포인트 위치: `data/checkpoints/pdf_parse_{backend}.json`

### 2. BrokenProcessPool 자동 복구

손상된 PDF가 worker 프로세스를 segfault시키면:
1. 해당 PDF를 skip으로 마킹
2. ProcessPool 자동 재생성
3. 나머지 PDF 계속 처리

### 3. 청크 기반 저장

500건마다 parquet 저장 + 체크포인트 갱신. 크래시 시 최대 500건만 손실.

---

## 기술 노트

### ProcessPoolExecutor를 쓰는 이유

MuPDF C 라이브러리가 thread-safe하지 않아 `ThreadPoolExecutor`에서 90% 실패합니다.
`ProcessPoolExecutor`로 전환하여 각 프로세스가 독립 MuPDF 인스턴스를 사용합니다.

### surrogate 문자 처리

일부 PDF에서 surrogate 문자(U+D800~U+DFFF)가 추출되면 parquet 저장 시
`UnicodeEncodeError`가 발생합니다. `save_chunk()`에서 자동으로 `\ufffd`(replacement character)로 치환합니다.

### MinerU CPU 설치 (Mac)

시스템 Python이 3.9인 경우:

```bash
# standalone Python 3.12 다운로드
curl -L -o /tmp/cpython.tar.gz \
  "https://github.com/indygreg/python-build-standalone/releases/download/20241206/cpython-3.12.8+20241206-aarch64-apple-darwin-install_only.tar.gz"
tar xzf /tmp/cpython.tar.gz -C /tmp/py312

# venv 생성 + MinerU 설치
/tmp/py312/python/bin/python3.12 -m venv /tmp/mineru_env
/tmp/mineru_env/bin/pip install "magic-pdf[full]" opencv-python-headless

# 모델 다운로드
/tmp/mineru_env/bin/python3.12 -c "
from huggingface_hub import snapshot_download
snapshot_download('opendatalab/PDF-Extract-Kit-1.0', local_dir='/tmp/mineru_models')
"

# config 생성
cat > ~/magic-pdf.json << 'EOF'
{
    "models-dir": "/tmp/mineru_models/models",
    "device-mode": "cpu",
    "formula-config": {"is_formula_recog_enable": true},
    "table-config": {"is_table_recog_enable": false},
    "layout-config": {"model": "doclayout_yolo"},
    "latex-delimiter-config": {
        "display": {"left": "$$", "right": "$$"},
        "inline": {"left": "$", "right": "$"}
    }
}
EOF

# 실행
/tmp/mineru_env/bin/magic-pdf -p input.pdf -o output_dir -m txt
```

### 향후 파서 교체 전략

Raw PDF(`data/arxiv/pdfs/`)를 영구 보관하므로, 더 좋은 파서가 나오면:

```bash
# 1줄로 전체 교체
python3 batch_parse_pdfs.py --backend mineru --replace-source arxiv_pdf_pymupdf4llm --workers 4
```

교체 후 dedup 재실행 권장:
```bash
python main.py index && python main.py dedup && python main.py eval
```
