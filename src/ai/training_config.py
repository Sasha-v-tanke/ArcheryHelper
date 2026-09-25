from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ai.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_TEST_SPLIT


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    train_test_split: float = TRAIN_TEST_SPLIT
    seed: int = 42
    num_aug: int = 4
    weight_decay: float = 1e-5
    loss_alpha: float = 1.0
    loss_beta: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "TrainingConfig":
        return TrainingConfig(
            epochs=int(data.get("epochs", EPOCHS)),
            batch_size=int(data.get("batch_size", BATCH_SIZE)),
            learning_rate=float(data.get("learning_rate", LEARNING_RATE)),
            train_test_split=float(data.get("train_test_split", TRAIN_TEST_SPLIT)),
            seed=int(data.get("seed", 42)),
            num_aug=int(data.get("num_aug", 4)),
            weight_decay=float(data.get("weight_decay", 1e-5)),
            loss_alpha=float(data.get("loss_alpha", 1.0)),
            loss_beta=float(data.get("loss_beta", 10.0)),
        )


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_training_config(config: TrainingConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def set_training_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
