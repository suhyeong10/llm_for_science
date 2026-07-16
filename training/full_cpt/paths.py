"""Shared repository paths for full-CPT training code."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TRAINING_DIR = REPO_ROOT / "training"
FULL_CPT_DIR = TRAINING_DIR / "full_cpt"
TRAINING_CONFIG_DIR = FULL_CPT_DIR / "configs"
TRAINING_SCRIPT_DIR = FULL_CPT_DIR / "scripts"
TRAINING_RUN_DIR = TRAINING_DIR / "runs"
TRAINING_CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
