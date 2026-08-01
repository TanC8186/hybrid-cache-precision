"""量化器正确性门禁（pytest）。

论文任何表格产生之前，这些测试必须通过（design §5.12 / §7）。
"""
from __future__ import annotations

import pytest
import torch

from kvcache.eviction import keep_mask
from kvcache.quantizers import KVQuantizer
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


# ---- 真实 KVQuantizer 测试 ----

@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("granularity", ["per_token", "per_channel", "per_tensor"])
def test_real_quantizer_roundtrip(bits: int, granularity: str) -> None:
    """8/4-bit 各粒度 roundtrip 相对误差应小（按 scale 归一）。"""
    x = torch.randn(64, 128) * 0.1
    q = KVQuantizer(bits=bits, granularity=granularity)
    qt = q.quantize(x)
    xr = q.dequantize(qt)
    scale = qt.scale
    rel_err = ((xr.float() - x) / (scale.float() + 1e-12)).abs().max().item()
    assert rel_err < 0.6, f"bits={bits} granularity={granularity} rel_err={rel_err}"


def test_2bit_uses_asymmetric() -> None:
    """2-bit 默认非对称（对称在 2-bit 下只有 1 个正电平，质量差）。"""
    q = KVQuantizer(bits=2)
    assert not q.symmetric


def test_bytes_accounting_scales_with_bits() -> None:
    """同样数据，位数越高 bytes 越大。"""
    x = torch.randn(64, 128)
    b8 = KVQuantizer(bits=8).quantize(x).bytes_used()
    b2 = KVQuantizer(bits=2).quantize(x).bytes_used()
    assert b8 > b2


def test_eviction_keeps_recent_and_heavy_hitters() -> None:
    """H2O 驱逐：保留最近 window 个 + 累计分数最高的 rest。"""
    scores = torch.tensor([0.1, 0.05, 0.4, 0.2, 0.25])
    mask = keep_mask(scores, budget=3, window=2)
    # 最近 2 个（索引 3,4）必须保留；剩余 3 个里按分数保留 top-1（索引 2, score=0.4）
    assert mask[-2:].all()
    assert mask[2]  # score 0.4 保留
    assert not mask[0] and not mask[1]
    assert mask.sum() == 3
