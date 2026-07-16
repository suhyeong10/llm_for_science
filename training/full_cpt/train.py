"""CPT training entrypoint for Qwen-style causal language models."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from itertools import chain
from dataclasses import dataclass
from typing import Any, Dict, List


def _enable_tf32_if_configured(cfg: Dict[str, Any]) -> None:
    runtime_cfg = cfg.get("runtime", {})
    trainer_cfg = cfg.get("trainer", {})
    tf32 = bool(runtime_cfg.get("tf32", trainer_cfg.get("tf32", False)))
    if not tf32:
        return
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass
class BuildResult:
    train_dataset: Any
    eval_dataset: Any
    tokenizer: Any

def _simple_causal_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch

    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attention_mask = [], [], []
    for f in features:
        ids = f["input_ids"]
        lbs = f["labels"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [0] * pad_len)
        labels.append(lbs + [-100] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
class CharTokenizer:
    def __init__(self, texts: List[str], max_vocab: int = 512):
        chars = {}
        for t in texts:
            for c in t:
                chars[c] = chars.get(c, 0) + 1
        sorted_chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
        selected = [c for c, _ in sorted_chars[: max_vocab - 3]]
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.eos_token = "<eos>"
        self.stoi = {c: i + 3 for i, c in enumerate(selected)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text: str, max_length: int) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_token_id) for ch in text][: max_length - 1]
        ids.append(self.eos_token_id)
        return ids

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "char_tokenizer_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)


def _build_local_debug(cfg: Dict[str, Any]) -> BuildResult:
    from datasets import load_dataset

    data_cfg = cfg["data"]
    train_ds = load_dataset("json", data_files=data_cfg["train_file"], split="train")
    valid_ds = load_dataset("json", data_files=data_cfg["valid_file"], split="train")

    text_field = data_cfg.get("text_field", "text")
    max_len = int(data_cfg.get("max_seq_length", 256))

    train_texts = [x.get(text_field, "") or "" for x in train_ds]
    valid_texts = [x.get(text_field, "") or "" for x in valid_ds]
    tokenizer = CharTokenizer(train_texts + valid_texts)

    def preprocess(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        input_ids = [tokenizer.encode(t or "", max_len) for t in batch[text_field]]
        return {"input_ids": input_ids, "labels": [x[:] for x in input_ids]}

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    valid_ds = valid_ds.map(preprocess, batched=True, remove_columns=valid_ds.column_names)
    return BuildResult(train_dataset=train_ds, eval_dataset=valid_ds, tokenizer=tokenizer)


def _build_hf(cfg: Dict[str, Any], local_files_only: bool) -> BuildResult:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    tok_cfg = cfg.get("tokenizer", {})

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        use_fast=bool(tok_cfg.get("use_fast", True)),
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = tok_cfg.get("padding_side", "right")
    tokenizer.truncation_side = tok_cfg.get("truncation_side", "right")

    train_ds = load_dataset("json", data_files=data_cfg["train_file"], split="train")
    valid_ds = load_dataset("json", data_files=data_cfg["valid_file"], split="train")

    text_field = data_cfg.get("text_field", "text")
    max_len = int(data_cfg.get("max_seq_length", 2048))
    add_eos = bool(data_cfg.get("add_eos_token", True))
    pack_sequences = bool(data_cfg.get("pack_sequences", False))

    def preprocess(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        texts = []
        for t in batch[text_field]:
            s = t if t is not None else ""
            if add_eos and tokenizer.eos_token and not s.endswith(tokenizer.eos_token):
                s = s + tokenizer.eos_token
            texts.append(s)
        out = tokenizer(
            texts,
            truncation=not pack_sequences,
            max_length=max_len if not pack_sequences else None,
            padding=False,
            add_special_tokens=bool(tok_cfg.get("add_special_tokens", True)),
        )
        out["labels"] = out["input_ids"].copy()
        return out

    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated_examples = {k: list(chain.from_iterable(examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])
        total_length = (total_length // max_len) * max_len
        if total_length == 0:
            empty = {k: [] for k in concatenated_examples.keys()}
            empty["labels"] = []
            return empty
        result = {
            k: [t[i : i + max_len] for i in range(0, total_length, max_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = [ids[:] for ids in result["input_ids"]]
        return result

    train_ds = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
        num_proc=int(data_cfg.get("preprocessing_num_workers", 1)),
    )
    valid_ds = valid_ds.map(
        preprocess,
        batched=True,
        remove_columns=valid_ds.column_names,
        num_proc=int(data_cfg.get("preprocessing_num_workers", 1)),
    )
    if pack_sequences:
        train_ds = train_ds.map(group_texts, batched=True, num_proc=int(data_cfg.get("preprocessing_num_workers", 1)))
        valid_ds = valid_ds.map(group_texts, batched=True, num_proc=int(data_cfg.get("preprocessing_num_workers", 1)))
    return BuildResult(train_dataset=train_ds, eval_dataset=valid_ds, tokenizer=tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPT training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    _enable_tf32_if_configured(cfg)
    seed = int(cfg.get("run", {}).get("seed", 42))
    _set_seed(seed)

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    model_cfg = cfg["model"]
    local_debug = bool(model_cfg.get("local_debug_tiny", False))

    if local_debug:
        from transformers import GPT2Config, GPT2LMHeadModel

        build = _build_local_debug(cfg)
        vocab_size = max(build.tokenizer.stoi.values(), default=2) + 1
        model = GPT2LMHeadModel(
            GPT2Config(vocab_size=vocab_size, n_positions=256, n_layer=2, n_head=2, n_embd=64)
        )
    else:
        from transformers import AutoModelForCausalLM

        try:
            build = _build_hf(cfg, local_files_only=args.local_files_only)
            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
                "torch_dtype": model_cfg.get("torch_dtype", "auto"),
                "local_files_only": args.local_files_only,
            }
            attn_impl = model_cfg.get("attn_implementation")
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl
            model = AutoModelForCausalLM.from_pretrained(
                model_cfg["base_model"],
                **model_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load model/tokenizer from Hugging Face. "
                "If this is a restricted network, use one of:\n"
                "1) --local_files_only with pre-cached model\n"
                "2) config model.local_debug_tiny=true for offline smoke test"
            ) from exc

        if bool(model_cfg.get("gradient_checkpointing", False)):
            gc_kwargs = model_cfg.get("gradient_checkpointing_kwargs", {})
            if isinstance(gc_kwargs, dict) and gc_kwargs:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
            else:
                model.gradient_checkpointing_enable()

    run_cfg = cfg["run"]
    tr_cfg = cfg["trainer"]
    opt_cfg = cfg["optimizer"]
    sch_cfg = cfg["scheduler"]

    os.makedirs(run_cfg["output_dir"], exist_ok=True)
    max_steps = 5 if args.dry_run else int(tr_cfg.get("max_steps", -1))

    ta_kwargs = {
        "output_dir": run_cfg["output_dir"],
        "per_device_train_batch_size": int(tr_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(tr_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(tr_cfg.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(tr_cfg.get("num_train_epochs", 1)),
        "max_steps": max_steps,
        "bf16": bool(tr_cfg.get("bf16", False)),
        "fp16": bool(tr_cfg.get("fp16", False)),
        "dataloader_num_workers": int(tr_cfg.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(tr_cfg.get("dataloader_pin_memory", True)),
        "remove_unused_columns": bool(tr_cfg.get("remove_unused_columns", False)),
        "evaluation_strategy": tr_cfg.get("evaluation_strategy", "steps"),
        "eval_steps": int(run_cfg.get("eval_interval_steps", 500)),
        "save_strategy": tr_cfg.get("save_strategy", "steps"),
        "save_steps": int(run_cfg.get("save_interval_steps", 500)),
        "save_total_limit": int(run_cfg.get("save_total_limit", 2)),
        "logging_strategy": tr_cfg.get("logging_strategy", "steps"),
        "logging_steps": int(run_cfg.get("log_interval_steps", 10)),
        "learning_rate": float(opt_cfg.get("learning_rate", 1e-5)),
        "weight_decay": float(opt_cfg.get("weight_decay", 0.0)),
        "adam_beta1": float(opt_cfg.get("betas", [0.9, 0.999])[0]),
        "adam_beta2": float(opt_cfg.get("betas", [0.9, 0.999])[1]),
        "adam_epsilon": float(opt_cfg.get("eps", 1e-8)),
        "max_grad_norm": float(opt_cfg.get("max_grad_norm", 1.0)),
        "lr_scheduler_type": sch_cfg.get("type", "cosine"),
        "optim": opt_cfg.get("name", "adamw_torch"),
        "report_to": run_cfg.get("report_to", []),
        "seed": seed,
        "tf32": bool(cfg.get("runtime", {}).get("tf32", tr_cfg.get("tf32", False))),
    }

    # transformers 버전에 따라 warmup_ratio 지원/경고가 달라서
    # 가능한 경우 warmup_steps를 우선 사용한다.
    warmup_steps_cfg = sch_cfg.get("warmup_steps")
    if warmup_steps_cfg is not None:
        ta_kwargs["warmup_steps"] = int(warmup_steps_cfg)
    else:
        warmup_ratio = float(sch_cfg.get("warmup_ratio", 0.0))
        if warmup_ratio > 0:
            if max_steps > 0:
                total_steps = max_steps
            else:
                epochs = float(tr_cfg.get("num_train_epochs", 1))
                train_rows = max(1, len(build.train_dataset))
                micro_batch = max(1, int(tr_cfg.get("per_device_train_batch_size", 1)))
                grad_accum = max(1, int(tr_cfg.get("gradient_accumulation_steps", 1)))
                steps_per_epoch = max(1, math.ceil(train_rows / micro_batch / grad_accum))
                total_steps = max(1, int(math.ceil(epochs * steps_per_epoch)))
            ta_kwargs["warmup_steps"] = max(1, int(total_steps * warmup_ratio))
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    deepspeed_cfg = cfg.get("runtime", {}).get("deepspeed_config")
    if deepspeed_cfg and "deepspeed" in valid_params:
        ta_kwargs["deepspeed"] = str(deepspeed_cfg)
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in valid_params}
    training_args = TrainingArguments(**ta_kwargs)

    data_collator = _simple_causal_collator if local_debug else DataCollatorForLanguageModeling(tokenizer=build.tokenizer, mlm=False)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": build.train_dataset,
        "eval_dataset": build.eval_dataset,
        "tokenizer": None if local_debug else build.tokenizer,
        "data_collator": data_collator,
    }
    trainer_valid = set(inspect.signature(Trainer.__init__).parameters.keys())
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in trainer_valid}
    trainer = Trainer(**trainer_kwargs)

    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    if hasattr(build.tokenizer, "save_pretrained"):
        build.tokenizer.save_pretrained(run_cfg["output_dir"])

    metrics = result.metrics
    eval_metrics = trainer.evaluate()
    metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    with open(os.path.join(run_cfg["output_dir"], "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[DONE] Training completed")


if __name__ == "__main__":
    main()
