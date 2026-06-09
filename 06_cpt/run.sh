#!/bin/bash
# CPT 중형 모델 학습 실행 스크립트
# 담당자: 김승환
#
# 사용:
#   bash run.sh                              # 단일 GPU
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh # 멀티 GPU (FSDP 자동)
#   bash run.sh path/to/other_config.yaml    # 다른 config 파일

set -e

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG_PATH="${1:-config.yaml}"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# config에서 FSDP wrap 대상 레이어 클래스명 추출
# (없거나 yq 미설치 시 Qwen2DecoderLayer로 폴백)
WRAP_CLS=$(python - <<PY
import yaml, sys
try:
    cfg = yaml.safe_load(open("$CONFIG_PATH"))
    print(cfg.get("model", {}).get("fsdp_transformer_layer_cls", "Qwen2DecoderLayer"))
except Exception:
    print("Qwen2DecoderLayer")
PY
)

echo "================================================================"
echo "CPT 중형 모델 학습 시작"
echo "  Config:        $CONFIG_PATH"
echo "  GPUs:          $CUDA_VISIBLE_DEVICES (${NUM_GPUS}장)"
echo "  FSDP wrap cls: $WRAP_CLS"
echo "================================================================"

if [ "$NUM_GPUS" -le 1 ]; then
    # 단일 GPU — FSDP 없이 일반 실행 (메모리 부족하면 OOM 가능)
    python train.py --config "$CONFIG_PATH"
else
    # 멀티 GPU — accelerate launch + FSDP full_shard
    accelerate launch \
        --num_processes="$NUM_GPUS" \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --use_fsdp \
        --fsdp_sharding_strategy=FULL_SHARD \
        --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
        --fsdp_transformer_layer_cls_to_wrap="$WRAP_CLS" \
        --fsdp_state_dict_type=FULL_STATE_DICT \
        --fsdp_backward_prefetch=BACKWARD_PRE \
        --fsdp_use_orig_params=true \
        train.py --config "$CONFIG_PATH"
fi
