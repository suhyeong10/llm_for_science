"""
CPT 중형 모델 학습 진입점
담당자: 김승환

사용:
    python train.py --config config.yaml

또는 멀티 GPU (FSDP):
    accelerate launch --num_processes=N --use_fsdp \
        --fsdp_transformer_layer_cls_to_wrap=Qwen2DecoderLayer \
        --fsdp_sharding_strategy=FULL_SHARD \
        train.py --config config.yaml

(run.sh가 위 명령을 자동 구성합니다)
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="CPT 중형 모델 학습")
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def build_fsdp_args(model_cfg: dict, gpu_cfg: dict) -> dict:
    """FSDP 관련 TrainingArguments 인자 구성.

    accelerate launch --use_fsdp ... 로 실행되면 accelerate가 우선이고,
    이 함수는 단일 프로세스(`python train.py`)일 때 안전한 기본값을 채워둠.
    """
    fsdp_cfg = (gpu_cfg or {}).get("fsdp", {}) or {}
    if not fsdp_cfg.get("enabled", False):
        return {}

    strategy_map = {
        "full_shard": "full_shard auto_wrap",
        "shard_grad_op": "shard_grad_op auto_wrap",
        "hybrid_shard": "hybrid_shard auto_wrap",
        "no_shard": "no_shard",
    }
    fsdp_str = strategy_map.get(
        fsdp_cfg.get("sharding_strategy", "full_shard"),
        "full_shard auto_wrap",
    )
    if fsdp_cfg.get("offload_params", False):
        fsdp_str = fsdp_str + " offload"

    layer_cls = model_cfg.get("fsdp_transformer_layer_cls", "Qwen2DecoderLayer")

    return {
        "fsdp": fsdp_str,
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": [layer_cls],
            "state_dict_type": fsdp_cfg.get("state_dict_type", "FULL_STATE_DICT"),
            "use_orig_params": True,
            "limit_all_gathers": True,
        },
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    # ─── 모델 / 토크나이저 ───
    model_name = config["model"]["base"]
    trust_remote_code = config["model"].get("trust_remote_code", True)
    logger.info(f"모델 로드: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        # CausalLM은 EOS를 pad로 재사용 (collator는 mlm=False라 영향 없음)
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_dtype(config["model"].get("torch_dtype", "bfloat16")),
        trust_remote_code=trust_remote_code,
    )

    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ─── 데이터 ───
    datasets = load_corpus(config, tokenizer)
    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"] if len(datasets["eval"]) > 0 else None

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # next-token prediction
    )

    # ─── Trainer 인자 ───
    t = config["training"]
    fsdp_kwargs = build_fsdp_args(config.get("model", {}), config.get("gpu", {}))

    do_eval = eval_dataset is not None
    training_args_kwargs = dict(
        output_dir=t["output_dir"],
        num_train_epochs=t.get("num_epochs", 1),
        per_device_train_batch_size=t.get("per_device_batch_size", 1),
        per_device_eval_batch_size=t.get("per_device_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 16),
        learning_rate=float(t.get("learning_rate", 2e-5)),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.01),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        save_strategy="steps",
        save_steps=t.get("save_steps", 1000),
        eval_strategy="steps" if do_eval else "no",
        eval_steps=t.get("eval_steps", 1000) if do_eval else None,
        logging_steps=t.get("logging_steps", 50),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=do_eval and t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss") if do_eval else None,
        greater_is_better=t.get("greater_is_better", False) if do_eval else None,
        seed=seed,
        report_to=t.get("report_to", "tensorboard"),
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        # gradient clipping (NaN/spike 방어)
        max_grad_norm=t.get("max_grad_norm", 1.0),
        **fsdp_kwargs,
    )
    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # ─── 학습 ───
    # 마지막 체크포인트가 있으면 자동 이어서 (resume_from_checkpoint)
    last_ckpt = None
    output_dir = Path(t["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        last_ckpt = get_last_checkpoint(str(output_dir))
        if last_ckpt:
            logger.info(f"이전 체크포인트 발견 → 이어 학습: {last_ckpt}")
        else:
            logger.info("output_dir에 체크포인트 없음 → 처음부터 학습")

    logger.info("학습 시작")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # ─── 저장 ───
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"최종 모델 저장: {final_dir}")

    # 최종 eval (best 체크포인트 기준)
    if do_eval:
        metrics = trainer.evaluate()
        logger.info(f"최종 eval metrics: {metrics}")


if __name__ == "__main__":
    main()
