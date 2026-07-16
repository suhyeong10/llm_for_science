# Full-CPT Configs

이 디렉터리에는 full fine-tune CPT 실험 재현성에 필요한 설정 파일만 저장합니다.

현재 포함:

- `qwen35_4b_cpt_h100.yaml`: 단일 H100 기준 기본 권장값(4K, 안정형)
- `qwen35_4b_cpt_model_load_postcheck_1step.yaml`: 실제 모델 로드 이후 1-step 경로 검증
- `qwen35_4b_cpt_h100_stable_4k.yaml`: 운영 시작점(안정형)
- `qwen35_4b_cpt_h100_longctx_8k.yaml`: 긴 문맥 실험용
- `qwen35_4b_cpt_h100_oom_safe_2k_bnb8.yaml`: OOM 완화용(8bit optimizer)
- `ds_zero2_offload.json`: GPU 메모리 한계 시 DeepSpeed offload fallback
- `quality_rules.yaml`: 코퍼스 품질 필터링/정제 규칙

## 왜 이렇게 구성했나

- `pack_sequences: true`
  - CPT에서 padding 낭비를 줄이고 token utilization을 높이기 위함
- `bf16: true` + `runtime.tf32: true`
  - H100 단일 GPU에서 속도/안정성 균형이 좋음
- `optimizer.name: adamw_torch_fused`
  - 기본 AdamW 대비 H100에서 처리량 개선 여지가 큼
- `max_seq_length 4096`부터 시작
  - 8192는 유의미하지만 OOM/속도 리스크가 커서 2단계 실험으로 분리

## 학습이 잘 안될 때 config 전략

1. OOM 발생:
   - `qwen35_4b_cpt_h100_longctx_8k.yaml` → `qwen35_4b_cpt_h100_stable_4k.yaml`
   - 계속 OOM이면 `qwen35_4b_cpt_h100_oom_safe_2k_bnb8.yaml`
2. 계속 메모리 부족:
   - `runtime.deepspeed_config: training/full_cpt/configs/ds_zero2_offload.json` 추가
3. 수렴 불안정:
   - `learning_rate`를 `1e-5` → `5e-6`
   - `warmup_ratio`를 `0.03` → `0.05`
