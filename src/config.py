from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from dacite import from_dict


@dataclass
class ModelConfig:
    lr: float = 1e-3


@dataclass
class Config:
    model_config: ModelConfig
    batch_size: int = 32
    device: str = "cpu"
    data_filepath: str = "data/variables.csv"
    model_filepath: str | None = None

    @staticmethod
    def from_yaml(path: Path) -> Config:
        with open(path, "r") as stream:
            data = yaml.full_load(stream)
        return from_dict(data_class=Config, data=data)

    def to_yaml(self, path: Path) -> None:
        with open(path, "w") as stream:
            yaml.dump(asdict(self), stream)
