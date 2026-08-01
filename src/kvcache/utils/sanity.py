"""正确性门禁：任何论文表格产生之前必须通过的检查。

- roundtrip 位精确：quantize(dequantize(x)) 与 x 的误差在容差内
- KV 重建 MSE：量化 KV 重建后与 fp16 KV 的重建误差
- baseline 已知答案：复现 H2O/SnapKV/KIVI 数字对照发表值
"""
from __future__ import annotations

import torch

_DEFAULT_TOL = 1e-4


def roundtrip_error(x: torch.Tensor, quantizer) -> float:
    """quantize -> dequantize 的数值误差（量化器正确性的第一道闸）。"""
    xr = quantizer.dequantize(quantizer.quantize(x))
    return float((xr - x).abs().max().item())


def kv_reconstruction_mse(k_fp16: torch.Tensor, k_quant: torch.Tensor) -> float:
    """量化 KV 相对 fp16 的重建 MSE。"""
    return float(((k_quant - k_fp16) ** 2).mean().item())


def assert_sanity(
    x: torch.Tensor,
    quantizer,
    *,
    tol: float = _DEFAULT_TOL,
) -> None:
    """组合门禁：roundtrip 误差超容差直接抛错。"""
    err = roundtrip_error(x, quantizer)
    if err > tol:
        raise AssertionError(f"roundtrip error {err:.2e} exceeds tol {tol:.2e}")
