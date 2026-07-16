# llm_for_science

과학 분야 논문 코퍼스를 구축하고, 해당 코퍼스로 Qwen 계열 full fine-tune CPT를 실행하기 위한 작업공간입니다.

현재 `pipeline/`의 수집/파싱 기준은 담당 범위인 화학 분야에 맞춰져 있습니다. 학습 워크스페이스는 화학에 한정하지 않고 과학 분야 코퍼스 전체를 대상으로 확장하는 것을 전제로 둡니다.

## 구조

```text
.
├── main.py                  # 데이터 파이프라인 CLI
├── pipeline/                # 수집, PDF/LaTeX 파싱, dedup, eval 구현
├── training/
│   └── full_cpt/            # full fine-tune CPT 학습 코드/설정/스크립트
├── docs/                    # 상세 문서
├── keywords/                # 도메인 키워드
├── data/                    # 로컬 데이터 산출물, git 제외
└── logs/                    # 실행 로그, git 제외
```

## 빠른 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 데이터 준비 파이프라인

아래 명령은 순서대로 실행하는 것을 권장합니다.

```bash
python main.py arxiv --years 2000-2025                               # [1]
python main.py arxiv-latex --resume --workers 8                      # [2]
python main.py clean                                                  # [3]
python -m pipeline.batch_parse_pdfs --backend pymupdf4llm --workers 4 --chunk 500  # [4]
python main.py dedup                                                  # [5]
python main.py index                                                  # [6]
python main.py eval                                                   # [7]
```

[1] arXiv 메타데이터를 연도 범위 기준으로 수집합니다.  
[2] arXiv LaTeX 소스를 병렬로 내려받고 중단 지점부터 재개합니다.  
[3] 텍스트 정제/노이즈 제거 전처리를 수행합니다.  
[4] PDF를 배치 파싱해 학습용 텍스트 조각으로 변환합니다.  
[5] 문서/청크 중복을 제거해 코퍼스 품질을 높입니다.  
[6] 검색/조회용 인덱스를 구축합니다.  
[7] 코퍼스 품질 지표를 계산해 결과를 점검합니다.  

## 학습 실행 순서 (권장)

### 1) 오프라인 스모크 (코드 경로 점검)

```bash
python -m training.full_cpt.train \
  --config training/full_cpt/tests/smoke_cpt_config.yaml \
  --dry_run
```

이 단계는 실제 Qwen 모델을 로드하지 않고, 로컬 tiny debug 모델로 학습 루프를 확인합니다.
설정 파일: [smoke_cpt_config.yaml](training/full_cpt/tests/smoke_cpt_config.yaml)

### 2) 실제 모델 로드 이후 1-step 검증

```bash
bash training/full_cpt/scripts/run_cpt_train.sh \
  training/full_cpt/configs/qwen35_4b_cpt_model_load_postcheck_1step.yaml
```

이 단계에서 `Qwen/Qwen3.5-4B-Base`를 실제로 다운로드/로드하고, 1 step 학습 경로를 점검합니다.
설정 파일: [qwen35_4b_cpt_model_load_postcheck_1step.yaml](training/full_cpt/configs/qwen35_4b_cpt_model_load_postcheck_1step.yaml)

- 모델 파일 위치:
  - 기본: Hugging Face 캐시(`~/.cache/huggingface/hub`)에 자동 저장
  - 커스텀 캐시를 쓰려면 실행 전 `export HF_HOME=/path/to/hf_cache`
  - 이미 내려받은 로컬 모델을 쓰려면 config의 `model.base_model`을 로컬 경로로 변경
- 학습 데이터 위치(1-step 검증용):
  - `training/full_cpt/tests/smoke_train.jsonl`
  - `training/full_cpt/tests/smoke_valid.jsonl`
  - 목적: 품질 검증이 아니라 "실제 모델 로드 + 1 step 학습 경로" 검증

### 3) 실제 학습 실행

```bash
bash training/full_cpt/scripts/run_cpt_train.sh \
  training/full_cpt/configs/qwen35_4b_cpt_h100_stable_4k.yaml
```

실험 목적에 따라 `training/full_cpt/configs/`의 다른 전략 config(긴 문맥/메모리 완화)를 선택합니다.
기본 학습 설정: [qwen35_4b_cpt_h100_stable_4k.yaml](training/full_cpt/configs/qwen35_4b_cpt_h100_stable_4k.yaml)  
긴 문맥 실험: [qwen35_4b_cpt_h100_longctx_8k.yaml](training/full_cpt/configs/qwen35_4b_cpt_h100_longctx_8k.yaml)  
메모리 완화: [qwen35_4b_cpt_h100_oom_safe_2k_bnb8.yaml](training/full_cpt/configs/qwen35_4b_cpt_h100_oom_safe_2k_bnb8.yaml)  
메모리 fallback(선택): [ds_zero2_offload.json](training/full_cpt/configs/ds_zero2_offload.json)
H100 기본 설정값/설명/출력 예시: [training/full_cpt/README.md](training/full_cpt/README.md)

- 실제 학습 데이터 위치:
  - 기본 config 기준
    - `data/processed/corpus_v1/train.jsonl`
    - `data/processed/corpus_v1/valid.jsonl`
  - 파일 형식: JSONL, 각 줄에 `{"text": "..."}`
- 각 실험 config의 컨셉:
  - `stable_4k`: 첫 기준선 확보용(수렴/메모리/속도 균형)
    - 핵심 옵션: `max_seq_length=4096`, `grad_accum=32`, `optim=adamw_torch_fused`
  - `longctx_8k`: 더 긴 문서를 한 번에 보도록 하는 실험용(메모리 부담 증가)
    - 핵심 옵션: `max_seq_length=8192`, `grad_accum=16` (토큰/업데이트 규모는 유사하게 유지)
  - `oom_safe_2k_bnb8`: OOM 완화 우선(학습 가능성 확보)
    - 핵심 옵션: `max_seq_length=2048`, `optim=adamw_bnb_8bit`, `grad_accum=32`
  - `ds_zero2_offload`: GPU 메모리 한계 시 최후 수단(fallback)
    - 핵심 옵션: `runtime.deepspeed_config=ds_zero2_offload.json` (optimizer state 일부 CPU offload)

### 3-1) H100 기본 학습 설정값(요약)

기준 config: `training/full_cpt/configs/qwen35_4b_cpt_h100_stable_4k.yaml`

```yaml
model.base_model: Qwen/Qwen3.5-4B-Base
model.torch_dtype: bfloat16
model.attn_implementation: sdpa
model.gradient_checkpointing: true
data.pack_sequences: true
data.max_seq_length: 4096
trainer.per_device_train_batch_size: 1
trainer.gradient_accumulation_steps: 32
optimizer.name: adamw_torch_fused
optimizer.learning_rate: 1.0e-5
scheduler.type: cosine
scheduler.warmup_ratio: 0.03
runtime.tf32: true
```

- `max_seq_length=4096`: 긴 문서 문맥과 메모리 안정성의 균형점
- `batch=1`, `grad_accum=32`: 단일 H100에서 effective batch 확보
- `bf16 + tf32`: H100에서 일반적으로 성능/안정성 균형이 좋음
- `adamw_torch_fused`: 기본 AdamW 대비 처리량 개선 기대
- `pack_sequences=true`: padding 낭비를 줄여 token 효율 개선

예상 출력 예시(정상 동작 시):

```text
[INFO] Starting CPT training
[INFO] Config: training/full_cpt/configs/qwen35_4b_cpt_h100_stable_4k.yaml
...
{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
...
[DONE] Training completed
```

## 문서

- [프로젝트 상세 가이드](docs/PROJECT_GUIDE.md)
- [PDF 파싱 가이드](docs/PDF_PARSING_GUIDE.md)
- [Full-CPT 학습 가이드](training/full_cpt/README.md)
