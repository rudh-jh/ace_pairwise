from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[1]
src_root = project_root / "src"

for p in [project_root, src_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ace_pre.data_build.tensor_builder import RelationTensorBuilder
from ace_pre.data_build.a1_dataset import A1SequenceDataset, a1_collate_fn
from ace_pre.losses.weighted_huber import WeightedHuberLoss


SEEDS = [42, 52, 62]


VARIANTS = {
    "DiagBranchOnly": {
        "pair_mode": "none",
        "hidden_dim": 64,
        "dropout": 0.15,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "residual_scale": 0.0,
        "use_gate": False,
    },
    "DiagCorePairResidualGate": {
        "pair_mode": "core",
        "hidden_dim": 64,
        "dropout": 0.20,
        "lr": 1e-3,
        "weight_decay": 2e-4,
        "residual_scale": 0.50,
        "use_gate": True,
    },
    "DiagCorePairResidualSmallGate": {
        "pair_mode": "core",
        "hidden_dim": 64,
        "dropout": 0.20,
        "lr": 1e-3,
        "weight_decay": 2e-4,
        "residual_scale": 0.25,
        "use_gate": True,
    },
    "DiagAllPairResidualGate": {
        "pair_mode": "all",
        "hidden_dim": 64,
        "dropout": 0.25,
        "lr": 8e-4,
        "weight_decay": 3e-4,
        "residual_scale": 0.30,
        "use_gate": True,
    },
}


CORE_PAIR_KEYWORDS = [
    "pair_hydrophobicity",
    "pair_charge",
    "pair_polarity",
    "pair_volume",
    "pair_aromatic_pair",
    "pair_donor_acceptor",
]


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, H]
        # mask:   [B, N]
        scores = (tokens * self.query.view(1, 1, -1)).sum(dim=-1)
        scores = scores / (tokens.shape[-1] ** 0.5)

        scores = scores.masked_fill(mask < 0.5, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return pooled


class TokenEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.net(tokens)
        z = self.pool(h, mask)
        return z


class DiagPairResidualRegressor(nn.Module):
    def __init__(
        self,
        diag_dim: int,
        pair_dim: int,
        hidden_dim: int,
        dropout: float,
        residual_scale: float,
        use_gate: bool,
    ) -> None:
        super().__init__()

        self.use_pair = pair_dim > 0 and residual_scale > 0
        self.residual_scale = float(residual_scale)
        self.use_gate = bool(use_gate)

        self.diag_encoder = TokenEncoder(
            input_dim=diag_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.diag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        if self.use_pair:
            self.pair_encoder = TokenEncoder(
                input_dim=pair_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

            self.pair_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

            if self.use_gate:
                self.gate_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )
            else:
                self.gate_head = None
        else:
            self.pair_encoder = None
            self.pair_head = None
            self.gate_head = None

    def forward(
        self,
        diag_tokens: torch.Tensor,
        diag_mask: torch.Tensor,
        pair_tokens: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_diag = self.diag_encoder(diag_tokens, diag_mask)
        y_diag = self.diag_head(z_diag).squeeze(-1)

        if not self.use_pair:
            return y_diag

        if pair_tokens is None or pair_mask is None:
            return y_diag

        z_pair = self.pair_encoder(pair_tokens, pair_mask)
        delta = self.pair_head(z_pair).squeeze(-1)

        if self.use_gate:
            gate = self.gate_head(z_diag).squeeze(-1)
        else:
            gate = torch.ones_like(delta)

        y = y_diag + self.residual_scale * gate * delta
        return y


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("配置为 auto，但没有检测到 CUDA。")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置为 cuda，但 torch.cuda.is_available() 为 False。")

    return torch.device(device_name)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    val = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(val) if pd.notna(val) else 0.0


def get_channel_names() -> list[str]:
    builder = RelationTensorBuilder(max_len=5)
    return list(builder.channel_names)


def is_core_pair_channel(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in CORE_PAIR_KEYWORDS)


def resolve_channel_indices(
    channel_names: list[str],
    pair_mode: str,
) -> tuple[list[int], list[int]]:
    diag_indices = [
        i for i, name in enumerate(channel_names)
        if name.lower().startswith("diag_")
    ]

    if pair_mode == "none":
        pair_indices = []

    elif pair_mode == "core":
        pair_indices = [
            i for i, name in enumerate(channel_names)
            if name.lower().startswith("pair_") and is_core_pair_channel(name)
        ]

    elif pair_mode == "all":
        pair_indices = [
            i for i, name in enumerate(channel_names)
            if name.lower().startswith("pair_")
        ]

    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")

    if not diag_indices:
        raise ValueError("No diag_ channels found.")

    return diag_indices, pair_indices


def build_loader(
    csv_path: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    builder = RelationTensorBuilder(max_len=5)
    dataset = A1SequenceDataset(csv_path=csv_path, tensor_builder=builder)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=a1_collate_fn,
    )


def move_batch_to_device_except_x(batch: dict, device: torch.device) -> dict:
    out = {}

    for k, v in batch.items():
        if k in {"x_hand", "pair_mask"}:
            out[k] = v
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v

    return out


def extract_diag_pair_inputs(
    batch: dict,
    diag_indices: list[int],
    pair_indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    x = batch["x_hand"].to(device)
    pair_mask_2d = batch["pair_mask"].to(device)  # [B, 1, 5, 5]

    B, _, L, _ = x.shape

    # diag tokens: [B, L, C_diag]
    x_diag = x[:, diag_indices, :, :]
    diag_tokens = torch.diagonal(x_diag, dim1=-2, dim2=-1).permute(0, 2, 1).contiguous()

    diag_mask = torch.diagonal(pair_mask_2d[:, 0, :, :], dim1=-2, dim2=-1).contiguous()

    if not pair_indices:
        return diag_tokens, diag_mask, None, None

    # pair tokens: [B, L*L, C_pair]
    x_pair = x[:, pair_indices, :, :]
    pair_tokens = x_pair.permute(0, 2, 3, 1).reshape(B, L * L, len(pair_indices)).contiguous()

    full_pair_mask = pair_mask_2d[:, 0, :, :]  # [B, L, L]
    eye = torch.eye(L, dtype=full_pair_mask.dtype, device=device).view(1, L, L)
    offdiag_mask = full_pair_mask * (1.0 - eye)
    pair_mask = offdiag_mask.reshape(B, L * L).contiguous()

    return diag_tokens, diag_mask, pair_tokens, pair_mask


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: WeightedHuberLoss,
    device: torch.device,
    diag_indices: list[int],
    pair_indices: list[int],
) -> float:
    model.train()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device_except_x(batch, device)

        diag_tokens, diag_mask, pair_tokens, pair_mask = extract_diag_pair_inputs(
            batch=batch,
            diag_indices=diag_indices,
            pair_indices=pair_indices,
            device=device,
        )

        y = batch["label_pIC50"].view(-1)
        sample_weight = batch["sample_weight"].view(-1)

        optimizer.zero_grad()

        pred = model(
            diag_tokens=diag_tokens,
            diag_mask=diag_mask,
            pair_tokens=pair_tokens,
            pair_mask=pair_mask,
        )

        loss = criterion(
            pred=pred,
            target=y,
            sample_weight=sample_weight,
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: WeightedHuberLoss,
    device: torch.device,
    diag_indices: list[int],
    pair_indices: list[int],
    save_predictions: bool = False,
) -> dict:
    model.eval()

    total_loss = 0.0
    n_batches = 0

    preds = []
    trues = []

    sample_ids = []
    sequences = []
    lengths = []
    task_roles = []

    for batch in loader:
        batch = move_batch_to_device_except_x(batch, device)

        diag_tokens, diag_mask, pair_tokens, pair_mask = extract_diag_pair_inputs(
            batch=batch,
            diag_indices=diag_indices,
            pair_indices=pair_indices,
            device=device,
        )

        y = batch["label_pIC50"].view(-1)
        sample_weight = batch["sample_weight"].view(-1)

        pred = model(
            diag_tokens=diag_tokens,
            diag_mask=diag_mask,
            pair_tokens=pair_tokens,
            pair_mask=pair_mask,
        )

        loss = criterion(
            pred=pred,
            target=y,
            sample_weight=sample_weight,
        )

        total_loss += float(loss.item())
        n_batches += 1

        pred_np = pred.detach().cpu().numpy().reshape(-1)
        true_np = y.detach().cpu().numpy().reshape(-1)

        preds.append(pred_np)
        trues.append(true_np)

        if save_predictions:
            sample_ids.extend(batch["sample_id"])
            sequences.extend(batch["sequence"])
            lengths.extend(batch["length"].detach().cpu().numpy().reshape(-1).tolist())
            task_roles.extend(batch["task_role"])

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    result = {
        "loss": total_loss / max(1, n_batches),
        "rmse": rmse_score(y_true, y_pred),
        "mae": mae_score(y_true, y_pred),
        "spearman": spearman_score(y_true, y_pred),
    }

    if save_predictions:
        result["pred_df"] = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "sequence": sequences,
                "length": lengths,
                "task_role": task_roles,
                "y_true": y_true,
                "y_pred": y_pred,
                "abs_error": np.abs(y_true - y_pred),
            }
        )

    return result


def train_one_variant(
    cfg: dict,
    seed: int,
    variant_name: str,
    variant_cfg: dict,
    channel_names: list[str],
    artifacts_dir: Path,
) -> dict:
    set_seed(seed)
    ensure_dir(artifacts_dir)

    device = choose_device(cfg["train"]["device"])

    split_dir = project_root / cfg["paths"]["split_dir"]

    train_joint_csv = split_dir / "train_joint.csv"
    val_main_csv = split_dir / "val_main.csv"
    test_main_csv = split_dir / "test_main.csv"

    train_loader = build_loader(
        train_joint_csv,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        shuffle=True,
    )

    val_loader = build_loader(
        val_main_csv,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        shuffle=False,
    )

    test_loader = build_loader(
        test_main_csv,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        shuffle=False,
    )

    diag_indices, pair_indices = resolve_channel_indices(
        channel_names=channel_names,
        pair_mode=variant_cfg["pair_mode"],
    )

    diag_names = [channel_names[i] for i in diag_indices]
    pair_names = [channel_names[i] for i in pair_indices]

    model = DiagPairResidualRegressor(
        diag_dim=len(diag_indices),
        pair_dim=len(pair_indices),
        hidden_dim=int(variant_cfg["hidden_dim"]),
        dropout=float(variant_cfg["dropout"]),
        residual_scale=float(variant_cfg["residual_scale"]),
        use_gate=bool(variant_cfg["use_gate"]),
    ).to(device)

    criterion = WeightedHuberLoss(delta=float(cfg["train"]["huber_delta"]))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant_cfg["lr"]),
        weight_decay=float(variant_cfg["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    monitor = cfg["selection"]["monitor"]
    select_mode = cfg["selection"]["mode"]

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    patience = int(early_cfg.get("patience", 10))
    min_delta = float(early_cfg.get("min_delta", 0.0))

    best_metric = float("inf") if select_mode == "min" else float("-inf")
    best_epoch = -1
    epochs_no_improve = 0

    best_ckpt = artifacts_dir / "best_model.pt"
    history = []

    with open(artifacts_dir / "channel_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "variant_name": variant_name,
                "variant_cfg": variant_cfg,
                "diag_indices": diag_indices,
                "diag_names": diag_names,
                "pair_indices": pair_indices,
                "pair_names": pair_names,
                "n_diag_channels": len(diag_indices),
                "n_pair_channels": len(pair_indices),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "-" * 80)
    print(f"Seed={seed} | Variant={variant_name}")
    print(f"Device: {device}")
    print(f"Diag channels: {len(diag_indices)}")
    print(f"Pair channels: {len(pair_indices)}")
    print(f"Variant cfg: {variant_cfg}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            diag_indices=diag_indices,
            pair_indices=pair_indices,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            diag_indices=diag_indices,
            pair_indices=pair_indices,
            save_predictions=False,
        )

        row = {
            "seed": seed,
            "variant": variant_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_spearman": val_metrics["spearman"],
        }

        history.append(row)

        current = row[monitor]

        if select_mode == "min":
            improved = current < (best_metric - min_delta)
        else:
            improved = current > (best_metric + min_delta)

        if improved:
            best_metric = current
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "variant_name": variant_name,
                    "variant_cfg": variant_cfg,
                    "diag_indices": diag_indices,
                    "pair_indices": pair_indices,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                },
                best_ckpt,
            )
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_rmse={row['val_rmse']:.4f} | "
            f"val_mae={row['val_mae']:.4f} | "
            f"val_spearman={row['val_spearman']:.4f}"
        )

        if early_enabled and epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    pd.DataFrame(history).to_csv(
        artifacts_dir / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_best = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        diag_indices=diag_indices,
        pair_indices=pair_indices,
        save_predictions=True,
    )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        diag_indices=diag_indices,
        pair_indices=pair_indices,
        save_predictions=True,
    )

    val_pred_df = val_best.pop("pred_df")
    test_pred_df = test_metrics.pop("pred_df")

    val_pred_df.to_csv(
        artifacts_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_pred_df.to_csv(
        artifacts_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "seed": seed,
        "variant": variant_name,
        "pair_mode": variant_cfg["pair_mode"],
        "n_diag_channels": len(diag_indices),
        "n_pair_channels": len(pair_indices),
        "hidden_dim": variant_cfg["hidden_dim"],
        "dropout": variant_cfg["dropout"],
        "lr": variant_cfg["lr"],
        "weight_decay": variant_cfg["weight_decay"],
        "residual_scale": variant_cfg["residual_scale"],
        "use_gate": variant_cfg["use_gate"],
        "best_epoch": best_epoch,
        "best_monitor": monitor,
        "best_metric": best_metric,
        "val_loss": val_best["loss"],
        "val_rmse": val_best["rmse"],
        "val_mae": val_best["mae"],
        "val_spearman": val_best["spearman"],
        "test_loss": test_metrics["loss"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_spearman": test_metrics["spearman"],
    }

    with open(artifacts_dir / "best_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDiag + Pair residual experiment finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df
        .groupby("variant", as_index=False)
        .agg(
            pair_mode=("pair_mode", "first"),
            n_diag_channels=("n_diag_channels", "first"),
            n_pair_channels=("n_pair_channels", "first"),
            residual_scale=("residual_scale", "first"),
            use_gate=("use_gate", "first"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_spearman_mean=("test_spearman", "mean"),
            test_spearman_std=("test_spearman", "std"),
            val_rmse_mean=("val_rmse", "mean"),
            val_rmse_std=("val_rmse", "std"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .sort_values(["test_rmse_mean", "test_mae_mean"], ascending=[True, True])
    )

    return summary


def main() -> None:
    base_cfg_path = project_root / "configs" / "train_a1_v1.yaml"
    base_cfg = load_yaml(base_cfg_path)

    summary_dir = project_root / "artifacts" / "a1_diag_pair_residual_summary"
    ensure_dir(summary_dir)

    channel_names = get_channel_names()

    all_rows = []

    for seed in SEEDS:
        for variant_name, variant_cfg in VARIANTS.items():
            cfg = copy.deepcopy(base_cfg)

            cfg["split"]["seed"] = seed
            cfg["paths"]["split_dir"] = f"data/final/splits/seed_{seed}"

            artifacts_dir = (
                project_root
                / "artifacts"
                / "a1_diag_pair_residual"
                / f"seed_{seed}"
                / variant_name
            )

            summary = train_one_variant(
                cfg=cfg,
                seed=seed,
                variant_name=variant_name,
                variant_cfg=variant_cfg,
                channel_names=channel_names,
                artifacts_dir=artifacts_dir,
            )

            all_rows.append(summary)

            pd.DataFrame(all_rows).to_csv(
                summary_dir / "a1_diag_pair_residual_raw_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    raw_df = pd.DataFrame(all_rows)

    raw_df.to_csv(
        summary_dir / "a1_diag_pair_residual_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = summarize_results(raw_df)

    summary_df.to_csv(
        summary_dir / "a1_diag_pair_residual_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("All Diag + Pair residual experiments finished.")
    print(f"Raw results saved to: {summary_dir / 'a1_diag_pair_residual_raw.csv'}")
    print(f"Summary saved to: {summary_dir / 'a1_diag_pair_residual_summary.csv'}")
    print("=" * 80)

    print("\nDiag + Pair residual summary:")
    print(
        summary_df[
            [
                "variant",
                "pair_mode",
                "n_diag_channels",
                "n_pair_channels",
                "residual_scale",
                "use_gate",
                "test_rmse_mean",
                "test_rmse_std",
                "test_mae_mean",
                "test_mae_std",
                "test_spearman_mean",
                "test_spearman_std",
            ]
        ]
    )


if __name__ == "__main__":
    main()