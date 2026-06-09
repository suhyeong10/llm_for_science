# 06_cpt — CPT 중형 모델 학습

**담당자**: 김승환
**실행자**: 박수형 (수형님 서버에서 일괄 실행 예정)

회의에서 정리한 통합 데이터(8분야, peS2o 스키마)를 입력으로,
80~85% 과학 + 15~20% 일반 비율로 데이터 믹싱하여 CPT 중형 모델을 학습합니다.

---

## 1. 설치

```bash
cd 06_cpt
pip install -r requirements.txt
```

## 2. 설정 (`config.yaml` 수정)

수정해야 할 부분은 5곳입니다:

| 항목 | 위치 | 설명 |
|------|------|------|
| `data.path` | 데이터 | 수형님 서버의 통합 데이터 경로 |
| `data.science_source_values` | 데이터 | 통합본의 source 필드에서 "과학"으로 간주할 값 (기본: `["pes2o_or_crawled"]`) |
| `data.general_source_values` | 데이터 | source 필드에서 "일반"으로 간주할 값 (없으면 빈 리스트) |
| `model.base` + `model.fsdp_transformer_layer_cls` | 모델 | **두 줄 같이 변경** (아래 표 참고) |
| `gpu.num_devices` | GPU | 실행 환경의 GPU 장 수 (run.sh가 `CUDA_VISIBLE_DEVICES`로 자동 인식) |

### 모델별 설정 매핑 (`model.base` 와 `fsdp_transformer_layer_cls` 같이)

| 모델 | `model.base` | `model.fsdp_transformer_layer_cls` | 인증 |
|---|---|---|---|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | `Qwen2DecoderLayer` | 불필요 |
| Qwen3-8B-Base | `Qwen/Qwen3-8B-Base` | `Qwen3DecoderLayer` | 불필요 |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B` | `LlamaDecoderLayer` | `huggingface-cli login` 필요 |

데이터 포맷 자동 인식:
- HF `save_to_disk` 디렉토리 (`dataset_info.json` 포함)
- `.parquet` 단일 파일 또는 디렉토리
- `.jsonl` 단일 파일 또는 디렉토리

### `source` 컬럼 실제 값 확인 (필수)

`data.science_source_values` / `data.general_source_values`에 어떤 값을 넣어야 할지
모르겠다면, 통합 데이터에서 직접 확인:

```python
from datasets import load_from_disk  # 또는 load_dataset
ds = load_from_disk("/실제/통합데이터/경로")
print(set(ds["source"][:5000]))   # 어떤 값들이 박혀있는지
```

여기서 나온 값을 config.yaml에 그대로 넣으면 됨.

## 3. 실행

### (선택) 사전 검증 — dry-run

학습 들어가기 전에 데이터 경로/source 값/패킹이 정상 동작하는지 한 번 확인:

```bash
cd 06_cpt
python -c "
import yaml
from transformers import AutoTokenizer
from data_loader import load_corpus

cfg = yaml.safe_load(open('config.yaml'))
tok = AutoTokenizer.from_pretrained(cfg['model']['base'], trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ds = load_corpus(cfg, tok)
print('Train:', len(ds['train']), 'Eval:', len(ds['eval']))
"
```

콘솔에 비율 로그(`초기 비율: 과학 ... / 일반 ...`)와 토큰 통계가 정상 출력되면 OK.
모델 가중치는 다운로드되지만 학습은 안 시작합니다.

### 본 학습

```bash
# 단일 GPU
bash run.sh

# 멀티 GPU (FSDP 자동 활성화)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh
```

`run.sh`가 `CUDA_VISIBLE_DEVICES`의 장 수를 보고:
- 1장: `python train.py`
- 2장 이상: `accelerate launch --use_fsdp ...` 로 자동 분기

### 디스크 공간 주의

7B 모델 체크포인트 1개당 약 **14GB**. `save_total_limit: 3` 설정이라 동시에 최대 3개 + `final/` 까지 → 학습 디렉토리 **약 60GB** 점유.
추가로 토큰화 캐시(`~/.cache/huggingface/datasets/`)가 데이터 양에 따라 **수십 GB** 소요.

→ 학습 시작 전 **여유 디스크 100GB 이상** 권장.

## 4. 결과

- 체크포인트: `./checkpoints/checkpoint-<step>/`
- 최종 모델: `./checkpoints/final/` (eval loss 최저 체크포인트 기준)
- TensorBoard 로그: `./checkpoints/runs/`

```bash
tensorboard --logdir ./checkpoints/runs
```

---

## 학습 파이프라인 동작 요약

```
1. 통합 데이터 로드 (parquet/jsonl/save_to_disk 자동 인식)
2. source 필드로 과학 / 일반 분리
3. 80~85% / 15~20% 비율 검증·보정 (auto_balance 토글)
4. 토큰화
5. 시퀀스 패킹 — 짧은 문서를 EOS로 이어붙여 max_seq_length 꽉 채움
6. train/eval 분할 (eval_split=0.5%)
7. 토큰 카운트 로깅 (epoch 수 결정 근거)
8. FSDP full_shard로 모델/grad/optim 분산
9. cosine LR scheduler + bf16 + gradient checkpointing
10. eval_steps마다 검증 → load_best_model_at_end로 최고 성능 체크포인트 저장
11. 중간에 죽으면 resume_from_checkpoint로 자동 이어 학습
```

---

## 데이터 믹싱 동작 (자동)

`data_loader.py`가 통합 데이터를 읽으면서 비율을 검증·보정합니다:

| 상황 | 동작 |
|------|------|
| 비율 80~85% 안에 있음 | 그대로 진행 |
| 과학 > 85% | (1) 일반 HF 데이터셋(`HuggingFaceFW/fineweb`) 보충 → 안 되면 (2) 과학 다운샘플 |
| 과학 < 80% | (1) 과학 HF 데이터셋(`allenai/peS2o`) 보충 → 안 되면 (2) 일반 다운샘플 |
| `auto_balance: false` | 보정 없이 경고만 띄우고 진행 |

콘솔에 비율 변화가 출력되니 학습 시작 전에 한 번 보고 이상 있으면 중단 가능합니다.

### 두 가지 모드 — `auto_balance` 토글

- **🅰️ 검증만 모드** — `auto_balance: false`. 80/20 맞나 확인하고 안 맞으면 경고만.
- **🅱️ 보정까지 모드** — `auto_balance: true` (기본). 안 맞으면 HF에서 자동 보충.

통합 데이터를 이미 신뢰한다면 `false`로 두면 됩니다.

---

## 7가지 안전·효율 장치 (적용됨)

| # | 항목 | 효과 |
|---|------|------|
| 1 | **시퀀스 패킹** | padding 90%+ 제거 → 같은 시간에 학습 토큰량 5~10배 |
| 2 | **resume_from_checkpoint** | 중간에 죽어도 마지막 체크포인트부터 자동 재개 |
| 3 | **FSDP full_shard** | params/grads/optim_states를 GPU들에 분산 → OOM 회피 |
| 4 | **TensorBoard 로깅** | loss/lr/grad_norm 실시간 모니터링 |
| 5 | **cosine LR scheduler** | linear 대비 부드러운 감쇠 → 최종 품질 향상 |
| 6 | **eval_dataset + load_best_model_at_end** | best 체크포인트를 final/로 자동 저장 |
| 7 | **토큰 카운트 로깅** | epoch 수 결정의 정량적 근거 |

---

## 파일 구성

```
06_cpt/
├── config.yaml          # 설정 (모델·데이터·학습·FSDP)
├── data_loader.py       # peS2o 로드 + 80/20 비율 보정 + 패킹 + eval split
├── train.py             # HF Trainer 학습 진입점 (resume + FSDP + best ckpt)
├── run.sh               # 단일/멀티 GPU + FSDP 실행 래퍼
├── requirements.txt     # Python 의존성
└── README.md            # 이 파일
```
