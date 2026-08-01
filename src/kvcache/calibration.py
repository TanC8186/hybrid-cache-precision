"""校准数据采集与算法。

校准是量化论文最常见的审稿攻击面，必须可复现：
- 数据集 / split / 采样数 / 序列长度 / 采样顺序 / seed 全部固化
- 每次运行把采样的 calibration indices 记入 run provenance
- 双环境（4060 dev / 5090 final）共用同一份校准 spec，并断言参数一致

算法：minmax / percentile / mse（配置见 configs/quantization/*.yaml）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CalibrationSpec:
    """校准配置——必须与 configs/quantization/*.yaml 的 calibration 块一致。"""

    dataset: str
    num_samples: int
    seq_len: int
    algorithm: str  # minmax | percentile | mse
    seed: int = 42
    percentile: float | None = None
    sample_order: str = "seeded_random"
    sampled_indices: list[int] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "CalibrationSpec":
        data = yaml.safe_load(path.read_text())["calibration"]
        return cls(**data)

    def to_provenance(self) -> dict:
        """固化到 run 快照的字段。"""
        return {
            "dataset": self.dataset,
            "num_samples": self.num_samples,
            "seq_len": self.seq_len,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "sample_order": self.sample_order,
            "sampled_indices": self.sampled_indices,
        }
