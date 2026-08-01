"""KV 量化器：2/4/8-bit，per-token / per-channel / per-tensor 粒度，对称/非对称。

这是第一步 Pareto 验证实验的核心。设计要点：
- 存储 int8 tensor（真实 2-bit 打包留到 vLLM 集成的 kernel 阶段）
- bytes 记账按 `bits/8 × 元素数 + scale/zero 开销` 计，保证跨方法公平对比
- 2-bit 默认非对称（对称的 -2..1 只有 1 个正电平，浪费）
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

GRANULARITIES = ("per_token", "per_channel", "per_tensor")


@dataclass
class QuantizedKV:
    """量化后的 K/V + 记账信息。"""

    data: torch.Tensor           # int8 tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None  # None 表示对称量化
    bits: int
    granularity: str
    numel: int                   # 原始元素数

    def bytes_used(self) -> float:
        """KV 实际占用字节（数据 + scale/zero，scale 按 fp16 计）。"""
        data_bytes = self.numel * self.bits / 8
        scale_bytes = self.scale.numel() * 2
        zp_bytes = 0.0 if self.zero_point is None else self.zero_point.numel() * 2
        return data_bytes + scale_bytes + zp_bytes


class KVQuantizer:
    """对 K/V 张量做量化/反量化。

    输入 shape: [..., dim]（对最后一个维度做 token/通道统计）。
    """

    def __init__(
        self,
        bits: int = 8,
        granularity: str = "per_token",
        symmetric: bool | None = None,
    ) -> None:
        assert bits in (2, 4, 8), f"bits 只支持 2/4/8，得到 {bits}"
        assert granularity in GRANULARITIES, f"granularity 必须 ∈ {GRANULARITIES}"
        self.bits = bits
        self.granularity = granularity
        # 2-bit 对称只有 4 个电平(-2,-1,0,1)且正侧只有 1 级，默认非对称
        self.symmetric = (bits != 2) if symmetric is None else symmetric
        self.qmin = -2 ** (bits - 1) if self.symmetric else 0
        self.qmax = 2 ** (bits - 1) - 1 if self.symmetric else 2**bits - 1

    # ---- 统计维度 ----
    def _per_dim(self, x: torch.Tensor, keepdim: bool) -> torch.Tensor:
        """按 granularity 返回沿最后一个维度的分组统计（广播用）。"""
        if self.granularity == "per_token":
            return torch.ones(*x.shape[:-1], 1, device=x.device) if keepdim else torch.tensor(1.0)
        raise NotImplementedError("内部辅助，per_channel/per_tensor 用 _amax/_minmax 直接算")

    def _amax(self, x: torch.Tensor) -> torch.Tensor:
        """按粒度求最大绝对值，返回可广播到 x 的 shape。"""
        if self.granularity == "per_token":
            return x.abs().amax(dim=-1, keepdim=True)
        if self.granularity == "per_channel":
            dims = tuple(range(x.dim() - 1))
            return x.abs().amax(dim=dims, keepdim=True)
        return x.abs().amax()

    def _minmax(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """按粒度求 min/max，返回可广播的 shape。"""
        if self.granularity == "per_token":
            return x.amax(dim=-1, keepdim=True), x.amin(dim=-1, keepdim=True)
        if self.granularity == "per_channel":
            dims = tuple(range(x.dim() - 1))
            return x.amax(dim=dims, keepdim=True), x.amin(dim=dims, keepdim=True)
        return x.amax(), x.amin()

    # ---- 量化/反量化 ----
    def quantize(self, x: torch.Tensor) -> QuantizedKV:
        """量化 K/V。x: [..., dim]."""
        numel = x.numel()
        if self.symmetric:
            scale = self._amax(x) / self.qmax
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            q = torch.round(x / scale).clamp(self.qmin, self.qmax).to(torch.int8)
            return QuantizedKV(q, scale, None, self.bits, self.granularity, numel)
        # 非对称：min/max + zero_point
        amax, amin = self._minmax(x)
        scale = (amax - amin) / (self.qmax - self.qmin)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = torch.round(self.qmin - amin / scale).clamp(self.qmin, self.qmax)
        q = torch.round(x / scale + zero).clamp(self.qmin, self.qmax).to(torch.int8)
        return QuantizedKV(q, scale, zero, self.bits, self.granularity, numel)

    def dequantize(self, qt: QuantizedKV) -> torch.Tensor:
        """反量化回 fp16（供注意力计算）。"""
        if qt.zero_point is None:
            return (qt.data.float() * qt.scale).to(torch.float16)
        return ((qt.data.float() - qt.zero_point) * qt.scale).to(torch.float16)

    def __repr__(self) -> str:
        sym = "sym" if self.symmetric else "asym"
        return f"KVQuantizer(bits={self.bits}, {self.granularity}, {sym})"
