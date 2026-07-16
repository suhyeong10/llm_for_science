"""
CPT 데이터 로더 — peS2o 스키마 통합 데이터 + 과학/일반 80~20 비율 동적 보정
담당자: 김승환

스키마: id, source, url, text, created, added, version
처리 흐름:
    1. 수형님 서버 통합 데이터 로드 (parquet / jsonl / save_to_disk 디렉토리)
    2. source 필드 보고 과학 / 일반 분리
    3. 비율 검증 → 안 맞으면 자동 보정
       - 과학 부족 → HF에서 과학 데이터 보충 (또는 일반 다운샘플)
       - 과학 과다 → HF에서 일반 데이터 보충 (또는 과학 다운샘플)
    4. 토큰화
    5. 시퀀스 패킹 (짧은 문서를 EOS로 이어붙여 max_length 꽉 채움)
    6. train/eval 분할 + 토큰 통계 로깅
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

logger = logging.getLogger(__name__)

PES2O_SCHEMA_FIELDS = ["id", "source", "url", "text", "created", "added", "version"]


# =============================================================================
# 1. 로컬 통합 데이터 로드
# =============================================================================
def load_local_corpus(data_path: str) -> Dataset:
    """수형님 서버의 통합 데이터 자동 로드.

    지원 포맷:
        - HF save_to_disk 디렉토리 (`dataset_info.json` 포함)
        - 단일 parquet / jsonl 파일
        - parquet / jsonl 파일이 들어있는 디렉토리
    """
    path = Path(data_path)

    if path.is_dir() and (path / "dataset_info.json").exists():
        logger.info(f"HF save_to_disk 디렉토리 로드: {path}")
        return load_from_disk(str(path))

    if path.is_file():
        if path.suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        if path.suffix in {".jsonl", ".json"}:
            return load_dataset("json", data_files=str(path), split="train")

    if path.is_dir():
        parquet_files = sorted(path.glob("**/*.parquet"))
        if parquet_files:
            logger.info(f"parquet 파일 {len(parquet_files)}개 로드")
            return load_dataset(
                "parquet",
                data_files=[str(f) for f in parquet_files],
                split="train",
            )
        jsonl_files = sorted(path.glob("**/*.jsonl"))
        if jsonl_files:
            logger.info(f"jsonl 파일 {len(jsonl_files)}개 로드")
            return load_dataset(
                "json",
                data_files=[str(f) for f in jsonl_files],
                split="train",
            )

    raise ValueError(f"데이터 경로를 해석할 수 없습니다: {data_path}")


# =============================================================================
# 2. source 필드로 과학 / 일반 분리
# =============================================================================
def split_by_source(
    dataset: Dataset,
    source_field: str,
    science_values: List[str],
    general_values: List[str],
) -> Tuple[Dataset, Dataset]:
    """source 필드 기준으로 과학 / 일반 분리."""
    if source_field not in dataset.column_names:
        raise KeyError(
            f"'{source_field}' 컬럼이 없습니다. 컬럼: {dataset.column_names}"
        )

    science_set = set(science_values)
    general_set = set(general_values)

    science = dataset.filter(lambda ex: ex[source_field] in science_set)
    general = dataset.filter(lambda ex: ex[source_field] in general_set)
    return science, general


# =============================================================================
# 3. 비율 검증 + 자동 보정 (이 모듈의 핵심)
# =============================================================================
def adjust_mixing(
    science: Dataset,
    general: Dataset,
    mix_cfg: Dict[str, Any],
    text_field: str,
    seed: int = 42,
) -> Dataset:
    """과학 : 일반 = 80~85 : 15~20 비율로 맞춰서 합친 데이터셋 반환.

    두 가지 모드 — config의 mix_cfg["auto_balance"]로 토글:

      🅰️ 검증만 모드 (auto_balance=False)
          80/20 맞나 확인하고 안 맞으면 경고만 띄우고 그대로 진행.
          HF 다운로드 X. 수형님 통합 데이터를 그대로 신뢰.

      🅱️ 보정까지 모드 (auto_balance=True)  ← 기본
          안 맞으면 HF에서 자동 보충해서 비율을 맞춤.
            - 과학 < 80%: science_fallback_hf로 과학 보충
            - 과학 > 85%: general_fallback_hf로 일반 보충
            - fallback HF 미설정 시: 다운샘플로 비율 맞춤
    """
    s_min: float = mix_cfg["science_ratio_min"]
    s_max: float = mix_cfg["science_ratio_max"]
    target: float = (s_min + s_max) / 2.0
    auto_balance: bool = mix_cfg.get("auto_balance", True)

    n_s, n_g = len(science), len(general)
    total = n_s + n_g

    if total == 0:
        raise ValueError("데이터가 비어있습니다.")

    current_ratio = n_s / total
    logger.info("=" * 60)
    logger.info(
        f"초기 비율: 과학 {n_s:,} ({current_ratio:.2%}) / "
        f"일반 {n_g:,} ({1 - current_ratio:.2%})"
    )
    logger.info(f"목표 비율: 과학 {s_min:.0%}~{s_max:.0%}")

    if s_min <= current_ratio <= s_max:
        logger.info("✓ 비율 OK — 그대로 진행")
        logger.info("=" * 60)
        return _shuffle_concat(science, general, seed)

    if not auto_balance:
        logger.warning("⚠ 비율 안 맞지만 auto_balance=false → 그대로 진행")
        logger.info("=" * 60)
        return _shuffle_concat(science, general, seed)

    if current_ratio > s_max:
        # 과학이 너무 많음 → 일반 보충 (선호) 또는 과학 다운샘플
        science, general = _handle_science_overflow(
            science, general, target, mix_cfg, text_field, seed,
        )
    else:
        # 과학이 부족 → 과학 보충 (선호) 또는 일반 다운샘플
        science, general = _handle_science_shortage(
            science, general, target, mix_cfg, text_field, seed,
        )

    n_s2, n_g2 = len(science), len(general)
    total2 = n_s2 + n_g2
    final_ratio = n_s2 / total2 if total2 else 0.0
    logger.info(
        f"보정 후: 과학 {n_s2:,} ({final_ratio:.2%}) / "
        f"일반 {n_g2:,} ({1 - final_ratio:.2%})"
    )
    logger.info("=" * 60)
    return _shuffle_concat(science, general, seed)


def _handle_science_overflow(
    science: Dataset,
    general: Dataset,
    target: float,
    mix_cfg: Dict[str, Any],
    text_field: str,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    n_s, n_g = len(science), len(general)
    fallback_hf = mix_cfg.get("general_fallback_hf")

    if fallback_hf:
        # 일반 보충 — 일반을 늘려서 비율 맞춤
        # n_s / (n_s + n_g_new) = target  →  n_g_new = n_s * (1 - target) / target
        n_g_target = int(n_s * (1.0 - target) / target)
        need = max(0, n_g_target - n_g)
        if need > 0:
            logger.info(f"⚠ 과학 과다 → HF '{fallback_hf}'에서 일반 {need:,}개 보충")
            extra = _load_hf_streaming_as_dataset(
                fallback_hf,
                need,
                source_label="general",
                text_field=mix_cfg.get("fallback_text_field_general", "text"),
                streaming=mix_cfg.get("fallback_streaming", True),
            )
            general = _normalize_to_schema(general)
            general = concatenate_datasets([general, extra])
        return science, general

    # 일반 보충 안 되면 과학 다운샘플
    n_s_new = int(n_g * target / (1.0 - target))
    n_s_new = max(1, n_s_new)
    logger.info(f"⚠ 과학 다운샘플: {n_s:,} → {n_s_new:,} (general_fallback_hf 미설정)")
    science = science.shuffle(seed=seed).select(range(n_s_new))
    return science, general


def _handle_science_shortage(
    science: Dataset,
    general: Dataset,
    target: float,
    mix_cfg: Dict[str, Any],
    text_field: str,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    n_s, n_g = len(science), len(general)
    fallback_hf = mix_cfg.get("science_fallback_hf")

    if fallback_hf:
        # 과학 보충
        # n_s_new / (n_s_new + n_g) = target  →  n_s_new = n_g * target / (1 - target)
        n_s_target = int(n_g * target / (1.0 - target))
        need = max(0, n_s_target - n_s)
        if need > 0:
            logger.info(f"⚠ 과학 부족 → HF '{fallback_hf}'에서 과학 {need:,}개 보충")
            extra = _load_hf_streaming_as_dataset(
                fallback_hf,
                need,
                source_label="pes2o_or_crawled",
                text_field=mix_cfg.get("fallback_text_field_science", "text"),
                streaming=mix_cfg.get("fallback_streaming", True),
            )
            science = _normalize_to_schema(science)
            science = concatenate_datasets([science, extra])
        return science, general

    # 과학 보충 안 되면 일반 다운샘플
    n_g_new = int(n_s * (1.0 - target) / target)
    n_g_new = max(1, n_g_new)
    logger.info(f"⚠ 일반 다운샘플: {n_g:,} → {n_g_new:,} (science_fallback_hf 미설정)")
    general = general.shuffle(seed=seed).select(range(n_g_new))
    return science, general


# =============================================================================
# 보조 유틸
# =============================================================================
def _shuffle_concat(a: Dataset, b: Dataset, seed: int) -> Dataset:
    if len(a) == 0:
        return b.shuffle(seed=seed)
    if len(b) == 0:
        return a.shuffle(seed=seed)
    a = _normalize_to_schema(a)
    b = _normalize_to_schema(b)
    return concatenate_datasets([a, b]).shuffle(seed=seed)


def _normalize_to_schema(ds: Dataset) -> Dataset:
    """peS2o 스키마(id, source, url, text, created, added, version)에 맞춰 컬럼 정규화."""
    cols = set(ds.column_names)
    for field in PES2O_SCHEMA_FIELDS:
        if field not in cols:
            ds = ds.add_column(field, [None] * len(ds))
    keep = [c for c in PES2O_SCHEMA_FIELDS if c in ds.column_names]
    extra_to_drop = [c for c in ds.column_names if c not in keep]
    if extra_to_drop:
        ds = ds.remove_columns(extra_to_drop)
    return ds


def _load_hf_streaming_as_dataset(
    hf_id: str,
    take: int,
    source_label: str,
    text_field: str = "text",
    streaming: bool = True,
) -> Dataset:
    """HF 대용량 데이터셋에서 streaming으로 take개 가져와 peS2o 스키마로 변환."""
    iterable = load_dataset(hf_id, split="train", streaming=streaming)

    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(iterable):
        if i >= take:
            break
        text = ex.get(text_field, "")
        if not text:
            continue
        rows.append({
            "id": str(ex.get("id", f"{hf_id}:{i}")),
            "source": source_label,
            "url": ex.get("url"),
            "text": text,
            "created": ex.get("created"),
            "added": None,
            "version": ex.get("version"),
        })
    return Dataset.from_list(rows)


# =============================================================================
# 4. 토큰화
# =============================================================================
def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    text_field: str = "text",
    num_proc: int = 4,
    packing: bool = False,
) -> Dataset:
    """text 필드 → input_ids로 토큰화.

    packing=False: 문서 1개 = 시퀀스 1개 (truncation, max_length까지).
    packing=True:  truncation 없이 전체 토큰화 → 이후 pack_sequences로 합침.
    """

    if packing:
        # 패킹은 truncation 없이 — 어차피 다음 단계에서 잘라낼 것
        def _tokenize(batch):
            return tokenizer(batch[text_field], add_special_tokens=False)
    else:
        def _tokenize(batch):
            return tokenizer(
                batch[text_field],
                truncation=True,
                max_length=max_length,
                padding=False,
            )

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="토큰화",
    )
    return tokenized


# =============================================================================
# 5. 시퀀스 패킹 — 짧은 문서를 EOS로 이어붙여 max_length 꽉 채움
# =============================================================================
def pack_sequences(
    tokenized: Dataset,
    max_length: int,
    eos_token_id: int,
    num_proc: int = 4,
) -> Dataset:
    """문서 단위 토큰을 EOS로 구분해서 max_length 길이의 청크로 재배열.

    효과: short doc 다수가 max_length까지 padding 되어 GPU 연산이 낭비되는 문제 해소.
          같은 wall-clock에 학습 토큰량 5~10배 증가.
    """

    def _pack(examples):
        # 배치 내 모든 input_ids를 EOS로 구분해 concat
        concatenated: List[int] = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
            concatenated.append(eos_token_id)

        # max_length 단위로 자름 (마지막 짜투리는 버림)
        total_length = (len(concatenated) // max_length) * max_length
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        chunks = [concatenated[i : i + max_length] for i in range(0, total_length, max_length)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * max_length for _ in chunks],
            "labels": [list(c) for c in chunks],
        }

    packed = tokenized.map(
        _pack,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=tokenized.column_names,
        desc="시퀀스 패킹",
    )
    return packed


# =============================================================================
# 6. 토큰 통계 로깅
# =============================================================================
def log_token_stats(dataset: Dataset, max_length: int, label: str = "train") -> int:
    """학습 토큰 수 통계 — epoch 수 결정의 근거."""
    n_seq = len(dataset)
    if n_seq == 0:
        logger.warning(f"[{label}] 시퀀스 0개")
        return 0

    # 패킹된 시퀀스는 max_length 가정, 안 패킹된 경우 실제 길이 합산
    sample = dataset[0]
    if len(sample["input_ids"]) == max_length:
        total_tokens = n_seq * max_length
    else:
        total_tokens = sum(len(ids) for ids in dataset["input_ids"])

    logger.info(
        f"[{label}] 시퀀스 {n_seq:,}개 / 학습 토큰 약 {total_tokens:,} "
        f"({total_tokens / 1e9:.2f}B)"
    )
    return total_tokens


# =============================================================================
# 진입점
# =============================================================================
def load_corpus(config: Dict[str, Any], tokenizer) -> DatasetDict:
    """전체 파이프라인 — train.py에서 호출.

    Returns:
        DatasetDict({"train": ..., "eval": ...})
        eval_split=0이면 "eval"은 비어있음.
    """
    data_cfg = config["data"]
    train_cfg = config.get("training", {})

    dataset = load_local_corpus(data_cfg["path"])
    logger.info(f"로드 완료: {len(dataset):,}개 레코드")
    logger.info(f"컬럼: {dataset.column_names}")

    science, general = split_by_source(
        dataset,
        source_field=data_cfg.get("source_field", "source"),
        science_values=data_cfg.get("science_source_values", []),
        general_values=data_cfg.get("general_source_values", []),
    )

    text_field = data_cfg.get("text_field", "text")
    seed = train_cfg.get("seed", 42)
    mixed = adjust_mixing(science, general, data_cfg["mixing"], text_field, seed=seed)

    max_length = train_cfg.get("max_seq_length", 4096)
    num_proc = train_cfg.get("preprocess_num_proc", 4)
    packing_cfg = data_cfg.get("packing", {})
    use_packing = packing_cfg.get("enabled", True)

    tokenized = tokenize_dataset(
        mixed,
        tokenizer,
        max_length=max_length,
        text_field=text_field,
        num_proc=num_proc,
        packing=use_packing,
    )
    logger.info(f"토큰화 완료: {len(tokenized):,}개 문서")

    if use_packing:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id가 None — 패킹 불가")
        packed = pack_sequences(tokenized, max_length, eos_id, num_proc=num_proc)
        logger.info(f"패킹 완료: {len(tokenized):,}개 문서 → {len(packed):,}개 시퀀스")
        tokenized = packed

    # train/eval 분할
    eval_ratio = float(packing_cfg.get("eval_split", 0.005))
    if eval_ratio > 0 and len(tokenized) > 100:
        split = tokenized.train_test_split(test_size=eval_ratio, seed=seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = tokenized
        eval_ds = tokenized.select(range(0))  # 빈 데이터셋

    log_token_stats(train_ds, max_length, label="train")
    if len(eval_ds) > 0:
        log_token_stats(eval_ds, max_length, label="eval")

    return DatasetDict({"train": train_ds, "eval": eval_ds})
