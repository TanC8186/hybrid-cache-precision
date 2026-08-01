"""量化器正确性门禁（pytest）。

论文任何表格产生之前，这些测试必须通过（design §5.12 / §7）。
"""
from __future__ import annotations

import torch

from kvcache.utils.sanity import assert_sanity, kv_reconstruction_mse


class _DummyQuantizer:
    """占位量化器：演示接口。实现后替换为 kvcache.quantizers 的真实量化器。"""

    def __init__(self, bits: int = 8, scale: float = 0.01) -> None:
        self.bits = bits
        self.scale = scale

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        # 简化演示：定点量化（非对称）
        qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        return torch.clamp(torch.round(x / self.scale), qmin, qmax)

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return q * self.scale


def test_roundtrip_error_within_tolerance() -> None:
    x = torch.randn(64, 128)
    assert_sanity(x, _DummyQuantizer(bits=8, scale=0.01), tol=1e-2)


def test_kv_reconstruction_mse_small() -> None:
    k_fp16 = torch.randn(64, 128)
    quant = _DummyQuantizer(bits=8, scale=0.01)
    k_quant = quant.dequantize(quant.quantize(k_fp16))
    assert kv_reconstruction_mse(k_fp16, k_quant) < 1e-4


def test_quantizer_baselines_known_answer() -> None:
    """TODO: 复现 H2O/SnapKV/KIVI 数字对照发表值，作为 baseline 正确性门禁。"""
    pass  # noqa: S101
