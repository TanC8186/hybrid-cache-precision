"""H2O（Heavy-Hitter Oracle）式驱逐策略。

核心思想：注意力分数高的 token 是"重头戏"（heavy hitters），对生成质量影响最大；
驱逐时保留累计注意力分数 top-k 的 token + 始终保留最近 window 个 token，其余丢弃。

接口设计供 harness 使用：
- `keep_mask(scores, budget, window)`：给定累计分数，返回保留掩码
- `AttentionScoreAccumulator`：逐 token 步累计注意力分数
"""
from __future__ import annotations

import torch


def keep_mask(
    scores: torch.Tensor,
    budget: int,
    window: int = 64,
) -> torch.Tensor:
    """H2O 保留掩码。

    Args:
        scores: [T] 累计注意力分数（越高越重要）
        budget: 允许保留的最大 token 数
        window: 始终保留的最近 token 数

    Returns:
        [T] bool 掩码（True = 保留）
    """
    t = scores.shape[0]
    if t <= budget:
        return torch.ones(t, dtype=torch.bool, device=scores.device)
    # 始终保留最近 window 个 token
    recent = torch.zeros(t, dtype=torch.bool, device=scores.device)
    recent[-window:] = True
    rest = ~recent
    k = budget - window
    if k <= 0:
        return recent
    # 在非最近 token 里按累计分数保留 top-k
    topk_idx = scores[rest].topk(k).indices
    rest_pos = torch.nonzero(rest).squeeze(1)
    mask = recent.clone()
    mask[rest_pos[topk_idx]] = True
    return mask


class AttentionScoreAccumulator:
    """逐 token 步累计每个缓存 token 的注意力分数（H2O 的 running scores）。"""

    def __init__(self, device: torch.device) -> None:
        self.scores = torch.zeros(0, dtype=torch.float32, device=device)

    def accumulate(self, attn_weights: torch.Tensor) -> None:
        """attn_weights: [1, num_heads, 1, T]（最后一行的 softmax 权重）。

        T = 当前缓存长度。累计到 self.scores；若缓存被驱逐过，attn_weights 的
        列对应的是当前保留的 token（按序），与 self.scores 长度一致。
        """
        w = attn_weights.mean(dim=(0, 1, 2))  # [T] 跨头平均
        assert w.shape[0] == self.scores.shape[0], (
            f"attn 列数 {w.shape[0]} != 缓存长度 {self.scores.shape[0]}"
        )
        self.scores += w

    def after_evict(self, mask: torch.Tensor) -> None:
        """驱逐后收缩 scores（只保留 mask 对应的 token）。"""
        self.scores = self.scores[mask]

    def reset(self) -> None:
        self.scores = torch.zeros(0, device=self.scores.device)
