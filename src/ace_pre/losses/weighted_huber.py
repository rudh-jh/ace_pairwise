from __future__ import annotations

import torch
import torch.nn as nn


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta: float = 0.5) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred: [B,1]
        target: [B,1]
        sample_weight: [B]
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
        if sample_weight.dim() != 1 or sample_weight.size(0) != pred.size(0):
            raise ValueError("sample_weight must be [B]")

        error = pred - target
        abs_error = torch.abs(error)

        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=pred.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear   # [B,1]

        weight = sample_weight.unsqueeze(-1)                # [B,1]
        weighted = loss * weight
        return weighted.sum() / weight.sum().clamp_min(1e-8)