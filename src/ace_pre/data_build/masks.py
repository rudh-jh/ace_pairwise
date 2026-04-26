from __future__ import annotations

from typing import Optional

import torch


def build_residue_mask(
    length: int,
    max_len: int = 5,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not (0 <= length <= max_len):
        raise ValueError(f"length must be in [0, {max_len}], got {length}")

    mask = torch.zeros(max_len, dtype=dtype, device=device)
    mask[:length] = 1.0
    return mask


def build_pair_mask(
    length: int,
    max_len: int = 5,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    residue_mask = build_residue_mask(length, max_len=max_len, device=device, dtype=dtype)
    pair_mask = torch.outer(residue_mask, residue_mask).unsqueeze(0)
    return pair_mask


def apply_2d_mask(x: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    if x.dim() not in (3, 4):
        raise ValueError(f"x must be 3D or 4D, got shape={tuple(x.shape)}")
    if pair_mask.dim() not in (3, 4):
        raise ValueError(f"pair_mask must be 3D or 4D, got shape={tuple(pair_mask.shape)}")

    if x.dim() == 3:
        if pair_mask.dim() != 3:
            raise ValueError("For x [C,H,W], pair_mask must be [1,H,W]")
        if x.shape[-2:] != pair_mask.shape[-2:]:
            raise ValueError(f"Spatial mismatch: x={tuple(x.shape)}, mask={tuple(pair_mask.shape)}")
        return x * pair_mask

    if pair_mask.dim() == 3:
        pair_mask = pair_mask.unsqueeze(0)
    if x.shape[0] != pair_mask.shape[0] and pair_mask.shape[0] != 1:
        raise ValueError(f"Batch mismatch: x={tuple(x.shape)}, mask={tuple(pair_mask.shape)}")
    if x.shape[-2:] != pair_mask.shape[-2:]:
        raise ValueError(f"Spatial mismatch: x={tuple(x.shape)}, mask={tuple(pair_mask.shape)}")

    return x * pair_mask


def masked_mean_2d(
    x: torch.Tensor,
    pair_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"x must be [B,C,H,W], got shape={tuple(x.shape)}")

    if pair_mask.dim() == 3:
        pair_mask = pair_mask.unsqueeze(0)
    if pair_mask.dim() != 4:
        raise ValueError(f"pair_mask must be [B,1,H,W] or [1,H,W], got shape={tuple(pair_mask.shape)}")

    if pair_mask.shape[-2:] != x.shape[-2:]:
        raise ValueError(f"Spatial mismatch: x={tuple(x.shape)}, mask={tuple(pair_mask.shape)}")

    if pair_mask.shape[0] == 1 and x.shape[0] > 1:
        pair_mask = pair_mask.expand(x.shape[0], -1, -1, -1)

    masked_x = x * pair_mask
    denom = pair_mask.sum(dim=(-1, -2)).clamp_min(eps)
    pooled = masked_x.sum(dim=(-1, -2)) / denom
    return pooled


def flatten_pair_mask(pair_mask: torch.Tensor) -> torch.Tensor:
    if pair_mask.dim() == 3:
        pair_mask = pair_mask.unsqueeze(0)
    if pair_mask.dim() != 4:
        raise ValueError(f"pair_mask must be [B,1,H,W] or [1,H,W], got shape={tuple(pair_mask.shape)}")

    b, c, h, w = pair_mask.shape
    if c != 1:
        raise ValueError(f"pair_mask channel dim must be 1, got {c}")

    return pair_mask.view(b, h * w)