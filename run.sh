#!/bin/bash

# 시스템의 3번 GPU(마지막 GPU)만 이 스크립트에서 보이도록 강제 지정합니다.
export CUDA_VISIBLE_DEVICES=3

export MKL_SERVICE_FORCE_INTEL=1


# 위에서 3번 GPU 1대만 보이도록 필터링했으므로, 
# DeepSpeed 입장에서는 가용 GPU가 1대(0번으로 인식됨)가 됩니다.
# deepspeed --num_gpus=1 src/train.py

deepspeed --num_gpus=1 src/train.py \
    --ds_config config/ds_config.json \
    --model_name_or_path gpt2 \
    --epochs 2 \
    --output_dir output/checkpoints