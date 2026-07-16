# Full-CPT Training Workspace

`Qwen/Qwen3.5-4B-Base` 기반 full fine-tune CPT(Continued Pretraining) 실행을 위한 작업공간입니다.

## 포함 범위

- 학습 엔트리포인트: `training/full_cpt/train.py`
- 실행 스크립트: `training/full_cpt/scripts/run_cpt_train.sh`
- 카파시티 루프: `training/full_cpt/scripts/run_capacity_loop.sh`
- 설정 파일:
  - `training/full_cpt/configs/qwen35_4b_cpt_h100.yaml`
  - `training/full_cpt/configs/quality_rules.yaml`
- 스모크 테스트 데이터/설정:
  - `training/full_cpt/tests/smoke_cpt_config.yaml`
  - `training/full_cpt/tests/smoke_train.jsonl`
  - `training/full_cpt/tests/smoke_valid.jsonl`

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 1. 오프라인 스모크 테스트

```bash
python -m training.full_cpt.train \
  --config training/full_cpt/tests/smoke_cpt_config.yaml \
  --dry_run
```

`smoke_cpt_config.yaml`은 `model.local_debug_tiny=true`로 설정되어 네트워크 없이 실행 가능합니다.

### 1-1. 실제 모델 로드 후 1-step 검증

오프라인 스모크와 달리, 아래 설정은 `Qwen/Qwen3.5-4B-Base`를 실제로 로드합니다.

```bash
bash training/full_cpt/scripts/run_cpt_train.sh \
  training/full_cpt/configs/qwen35_4b_cpt_model_load_postcheck_1step.yaml
```

이 검증은 다음을 확인합니다.

- Hugging Face에서 모델/토크나이저 실제 로드 가능 여부
- 1-step forward/backward/optimizer/update/save/eval 최소 경로 정상 동작

주의:

- `training/full_cpt/tests/smoke_*.jsonl`을 데이터로 쓰므로 코퍼스 품질 검증용이 아니라 런타임 경로 검증용입니다.
- GPU/네트워크/권한 상태에 따라 실패할 수 있습니다.
- 모델 파일은 기본적으로 Hugging Face 캐시(`~/.cache/huggingface/hub`)에 저장됩니다.
- 캐시 경로를 바꾸려면 실행 전에 `export HF_HOME=/path/to/hf_cache`를 설정합니다.
- 로컬 모델을 쓰려면 config의 `model.base_model` 값을 로컬 디렉터리로 변경합니다.

### 2. 기본 실행

```bash
bash training/full_cpt/scripts/run_cpt_train.sh \
  training/full_cpt/configs/qwen35_4b_cpt_h100.yaml
```

### 2-1. H100 기본 학습 설정값(요약)

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

- `max_seq_length=4096`: 긴 문맥 학습과 메모리 안정성의 균형점
- `batch=1`, `grad_accum=32`: 단일 H100에서 effective batch를 확보
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

### 3. 체크포인트 재시작

```bash
bash training/full_cpt/scripts/run_cpt_train.sh \
  training/full_cpt/configs/qwen35_4b_cpt_h100.yaml \
  --resume outputs/qwen35_4b_cpt_v2/checkpoint-XXXX
```

### 4. 카파시티 루프

```bash
bash training/full_cpt/scripts/run_capacity_loop.sh \
  training/full_cpt/tests/smoke_cpt_config.yaml
```

결과 요약: `training/full_cpt/tests/capacity_loop/summary.tsv`

## `pack_sequences` 동작 정리

`data.pack_sequences`는 기본적으로 `false`이며, `true`일 때만 시퀀스 패킹이 적용됩니다.

- `pack_sequences: false`
  - 문서별 토크나이즈 + `truncation=True` + `max_seq_length` 초과분 절단
  - 구현 단순, 디버그에 유리
- `pack_sequences: true`
  - 문서별 토크나이즈는 `truncation=False`
  - 이후 여러 샘플을 이어붙여 `max_seq_length` 고정 블록으로 재분할
  - `labels = input_ids`로 CPT(autoregressive next-token) loss 계산
  - 장점: 패딩 낭비 감소, 토큰 효율 향상
  - 주의: 문서 경계 보존이 약해질 수 있음

## 문서 기준 반영 체크 (Qwen3.5 CPT)

아래는 운영 가이드 문서 기준으로 현재 코드/설정에 반영한 항목입니다.

- 반영됨
  - `labels = input_ids` 기반 CPT(next-token loss)
  - `pack_sequences` 경로 구현(문서 concat 후 block 분할)
  - `optimizer.name`을 `TrainingArguments.optim`으로 연결
  - `runtime.tf32` 지원
  - `model.attn_implementation` 지원(`sdpa`/`flash_attention_2`)
  - `gradient_checkpointing_kwargs` 지원
  - `runtime.deepspeed_config` 지원
- 의도적으로 기본 미적용
  - `flash_attention_2` 강제 기본값: 설치/환경 의존성이 있어 기본은 `sdpa`
  - DeepSpeed offload 기본 활성화: 단일 H100 4K 기준은 offload 없이 먼저 검증
  - vision branch 별도 freeze 로직: 현재 엔트리포인트는 `AutoModelForCausalLM` 경로 중심

## 권장 실행 순서 (의미 설명 포함)

1. 기본 학습 시작점(균형형 4K)
   - 파일: `training/full_cpt/configs/qwen35_4b_cpt_h100_stable_4k.yaml`
   - 의미: 단일 H100에서 메모리/속도/수렴의 균형이 좋은 기본 프로파일입니다.
   - 언제 사용: 첫 실험, 기준선(baseline) 확보가 목적일 때
   - 핵심 옵션: `max_seq_length=4096`, `gradient_accumulation_steps=32`, `optimizer.name=adamw_torch_fused`

2. 더 긴 문서를 한 번에 학습하는 실험(8K)
   - 파일: `training/full_cpt/configs/qwen35_4b_cpt_h100_longctx_8k.yaml`
   - 의미: 한 샘플에서 더 긴 문서 구간을 함께 보도록 하여, 장문 문서 처리 성능 차이를 비교하는 프로파일입니다.
   - 언제 사용: 법률/논문/코드처럼 긴 문맥이 중요한 데이터에서 성능 비교가 필요할 때
   - 핵심 옵션: `max_seq_length=8192`, `gradient_accumulation_steps=16` (토큰/업데이트 규모는 유사하게 유지)

3. 메모리 여유가 부족할 때(완화형 2K + 8bit optimizer)
   - 파일: `training/full_cpt/configs/qwen35_4b_cpt_h100_oom_safe_2k_bnb8.yaml`
   - 의미: OOM 가능성을 낮추는 대신 처리량/품질 특성이 달라질 수 있는 프로파일입니다.
   - 언제 사용: 4K/8K 설정에서 반복적으로 OOM이 발생할 때
   - 핵심 옵션: `max_seq_length=2048`, `optimizer.name=adamw_bnb_8bit`, `gradient_accumulation_steps=32`

4. 최종 fallback(DeepSpeed offload)
   - 설정 키: `runtime.deepspeed_config`
   - 파일: `training/full_cpt/configs/ds_zero2_offload.json`
   - 의미: optimizer state 일부를 CPU로 오프로드해 GPU 메모리 압박을 줄이는 방법입니다.
   - 언제 사용: 위 1~3 단계로도 메모리 문제가 해소되지 않을 때
   - 핵심 옵션: ZeRO stage 2 + `offload_optimizer.device=cpu`

## 학습 데이터 배치 규칙

- 기본 학습 config가 참조하는 경로:
  - `data/processed/corpus_v1/train.jsonl`
  - `data/processed/corpus_v1/valid.jsonl`
- 데이터 형식:
  - JSONL
  - 각 줄은 최소 `text` 필드를 포함 (`{"text": "..."}`)
- 1-step 검증 config는 품질 검증이 아니라 런타임 경로 검증 목적이므로
  - `training/full_cpt/tests/smoke_train.jsonl`
  - `training/full_cpt/tests/smoke_valid.jsonl`
  를 그대로 사용합니다.

## 학습이 잘 안될 때 빠른 대응 가이드

- OOM:
  - `max_seq_length` 8192 → 4096 → 2048
  - `optimizer.name`을 `adamw_bnb_8bit`로 변경
- loss 진동/발산:
  - `learning_rate`를 절반으로 감소
  - `warmup_ratio` 증가(예: 0.03 → 0.05)
- 처리량 저하:
  - `attn_implementation: sdpa`로 고정 후 기준선 확인
  - dataloader worker 수를 시스템 I/O에 맞춰 조정

## 최소 완료 기준 (실험 종료 체크리스트)

- 통합 코퍼스 1개 버전 이상 준비
- CPT 1회 완주 및 체크포인트 저장
- train/valid loss 추세 확인
- 실행 로그 및 설정 파일 보관
