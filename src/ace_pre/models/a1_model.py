from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_group_count(num_channels: int, requested_groups: int = 8) -> int:
    g = min(requested_groups, num_channels)
    while g > 1:
        if num_channels % g == 0:
            return g
        g -= 1
    return 1


def masked_mean_2d(x: torch.Tensor, pair_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x: [B,C,H,W]
    pair_mask: [B,1,H,W]
    return: [B,C]
    """
    masked_x = x * pair_mask
    denom = pair_mask.sum(dim=(-1, -2)).clamp_min(eps)  # [B,1]
    pooled = masked_x.sum(dim=(-1, -2)) / denom         # [B,C]
    return pooled


class MaskedSEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = masked_mean_2d(x, pair_mask)                  # [B,C]
        gate = self.fc2(F.gelu(self.fc1(pooled)))              # [B,C]
        gate = torch.sigmoid(gate)                             # [B,C]
        out = x * gate.unsqueeze(-1).unsqueeze(-1)             # [B,C,H,W]
        out = out * pair_mask
        return out, gate


class LocalRelationBlock(nn.Module):
    """
    1x1 + 3x3 + residual + masked SE + mask
    input/output: [B,d,5,5]
    """
    def __init__(self, hidden_dim: int, requested_groups: int = 8, se_reduction: int = 4) -> None:
        super().__init__()
        num_groups = _choose_group_count(hidden_dim, requested_groups)

        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim)
        self.se = MaskedSEBlock(hidden_dim, reduction=se_reduction)

    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.gelu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.gelu(out)

        out = out + residual
        out, gate = self.se(out, pair_mask)
        out = out * pair_mask
        return out, gate


class RelationAttentionPooling(nn.Module):
    """
    tokens: [B,T,D]
    token_mask: [B,T]
    return:
      z_global: [B,D]
      attn_weights: [B,T]
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = math.sqrt(tokens.size(-1))
        logits = (tokens * self.query.unsqueeze(0).unsqueeze(0)).sum(dim=-1) / scale  # [B,T]

        invalid = ~token_mask.bool()
        logits = logits.masked_fill(invalid, float("-inf"))

        attn = torch.softmax(logits, dim=-1)  # [B,T]
        attn = self.dropout(attn)

        z_global = torch.bmm(attn.unsqueeze(1), tokens).squeeze(1)  # [B,D]
        return z_global, attn


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@dataclass
class A1ForwardOutput:
    y_hat: torch.Tensor
    hidden_map: torch.Tensor
    tokens: torch.Tensor
    token_mask: torch.Tensor
    attn_weights: torch.Tensor
    gate_weights: list[torch.Tensor]
    z_global: torch.Tensor
    z_local: torch.Tensor
    z_fused: torch.Tensor


class A1ACERegressor(nn.Module):
    """
    输入:
      x_hand: [B,C,5,5]
      pair_mask: [B,1,5,5]

    输出:
      y_hat: [B,1]
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1,
        se_reduction: int = 4,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        stem_groups = _choose_group_count(hidden_dim, 8)

        self.stem_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.stem_norm = nn.GroupNorm(num_groups=stem_groups, num_channels=hidden_dim)

        self.blocks = nn.ModuleList([
            LocalRelationBlock(
                hidden_dim=hidden_dim,
                requested_groups=8,
                se_reduction=se_reduction,
            )
            for _ in range(num_blocks)
        ])

        self.attn_pool = RelationAttentionPooling(hidden_dim=hidden_dim, dropout=attention_dropout)
        self.reg_head = RegressionHead(in_dim=hidden_dim * 2, hidden_dim=hidden_dim, dropout=dropout)

    def _tokenize(self, h: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, d, h_sp, w_sp = h.shape
        tokens = h.permute(0, 2, 3, 1).reshape(b, h_sp * w_sp, d)           # [B,25,D]
        token_mask = pair_mask.reshape(b, h_sp * w_sp) > 0                  # [B,25]
        return tokens, token_mask

    def forward(
        self,
        x_hand: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> A1ForwardOutput:
        if x_hand.dim() != 4:
            raise ValueError(f"x_hand must be [B,C,H,W], got {tuple(x_hand.shape)}")
        if pair_mask.dim() != 4:
            raise ValueError(f"pair_mask must be [B,1,H,W], got {tuple(pair_mask.shape)}")

        h = self.stem_conv(x_hand)
        h = self.stem_norm(h)
        h = F.gelu(h)
        h = h * pair_mask

        gate_weights = []
        for block in self.blocks:
            h, gate = block(h, pair_mask)
            gate_weights.append(gate)

        tokens, token_mask = self._tokenize(h, pair_mask)
        z_global, attn_weights = self.attn_pool(tokens, token_mask)
        z_local = masked_mean_2d(h, pair_mask)
        z_fused = torch.cat([z_global, z_local], dim=-1)  # [B,2D]

        y_hat = self.reg_head(z_fused)  # [B,1]

        return A1ForwardOutput(
            y_hat=y_hat,
            hidden_map=h,
            tokens=tokens,
            token_mask=token_mask,
            attn_weights=attn_weights,
            gate_weights=gate_weights,
            z_global=z_global,
            z_local=z_local,
            z_fused=z_fused,
        )