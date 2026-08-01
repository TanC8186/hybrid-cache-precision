"""量化器正确性门禁（pytest）。

论文任何表格产生之前，这些测试必须通过（design §5.12 / §7）。
"""
from __future__ import annotations

import torch

from kvcache.utils.sanity import assert_sanity, kv_reconstruction_mse


class _DummyQuantizer:
    """对称量化器测试桩：scale 从数据推导（与真实量化器行为一致）。

    实现后替换为 kvcache.quantizers 的真实量化器。
    """

    def __init__(self, bits: int = 8) -> None:
        self.bits = bits
        self.scale: float = 1.0

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        self.scale = (x.abs().max().item() or 1.0) / qmax
        return torch.clamp(torch.round(x / self.scale), qmin, qmax).to(torch.int8)

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return q.to(torch.float32) * self.scale


def test_roundtrip_error_within_tolerance() -> None:
    """8-bit 对称量化，roundtrip 最大误差应 < scale/2（数据推导 scale 时 ≈0.016）。"""
    x = torch.randn(64, 128)
    assert_sanity(x, _DummyQuantizer(bits=8), tol=2e-2)


def test_kv_reconstruction_mse_small() -> None:
    """8-bit 量化 KV 相对 fp16 的重建 MSE 应很小。"""
    k_fp16 = torch.randn(64, 128)
    quant = _DummyQuantizer(bits=8)
    k_quant = quant.dequantize(quant.quantize(k_fp16))
    assert kv_reconstruction_mse(k_fp16, k_quant) < 1e-3


def test_quantizer_baselines_known_answer() -> None:
    """TODO: 复现 H2O/SnapKV/KIVI 数字对照发表值，作为 baseline 正确性门禁。"""
    pass  # noqa: S101
