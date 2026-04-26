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


SPLIT_SEEDS = [42, 52, 62]

# 每个 split 内部训练 5 次，然后做 ensemble。
# 如果你想先快速测试，可以改成 [0, 1, 2]。
RUN_SEEDS = [0, 1, 2, 3, 4]


VARIANTS = {
    "DiagTokenLenHead": {
        "hidden_dim": 64,
        "num_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-3,
        "weight_decay": 2e-4,
        "mse_alpha": 0.10,
        "rank_lambda": 0.00,
        "select_by": "val_rmse",
    },
    "DiagTokenLenHeadRank": {
        "hidden_dim": 64,
        "num_blocks": 2,
        "dropout": 0.18,
        "lr": 1e-3,
        "weight_decay": 3e-4,
        "mse_alpha": 0.05,
        "rank_lambda": 0.02,
        "select_by": "combined",
    },
}


class ResidualTokenBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ffn(self.norm(x))
        x = x + h
        x = x * mask.unsqueeze(-1)
        return x


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = (tokens * self.query.view(1, 1, -1)).sum(dim=-1)
        scores = scores / (tokens.shape[-1] ** 0.5)

        scores = scores.masked_fill(mask < 0.5, -1e9)

        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return pooled


class A1DiagTokenLenHeadRegressor(nn.Module):
    """
    输入：
        diag_tokens: [B, 5, C_diag]
        diag_mask:   [B, 5]
        length:      [B]

    逻辑：
        1. 只处理 5 个对角线 residue tokens；
        2. 加 position embedding 和 length embedding；
        3. attention pooling；
        4. 按长度 2/3/4/5 使用不同 regression head。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
        max_len: int = 5,
    ) -> None:
        super().__init__()

        self.max_len = max_len

        self.input_norm = nn.LayerNorm(input_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_emb = nn.Parameter(torch.randn(max_len, hidden_dim) * 0.02)

        # length 只考虑 2/3/4/5，映射到 0/1/2/3
        self.len_emb = nn.Embedding(4, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                ResidualTokenBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        self.pool = MaskedAttentionPool(hidden_dim=hidden_dim)

        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2aa、3aa、4aa、5aa 各一个 head
        self.length_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(4)
            ]
        )

    def forward(
        self,
        diag_tokens: torch.Tensor,
        diag_mask: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        # diag_tokens: [B, 5, C]
        # diag_mask: [B, 5]
        # length: [B]

        B, L, _ = diag_tokens.shape

        length = length.view(-1).long()
        length_idx = torch.clamp(length - 2, min=0, max=3)

        x = self.input_norm(diag_tokens)
        x = self.input_proj(x)

        pos_emb = self.pos_emb[:L, :].view(1, L, -1)
        len_emb = self.len_emb(length_idx).view(B, 1, -1)

        x = x + pos_emb + len_emb
        x = x * diag_mask.unsqueeze(-1)

        for block in self.blocks:
            x = block(x, diag_mask)

        z = self.pool(x, diag_mask)
        z = self.shared(z)

        all_preds = torch.cat(
            [head(z) for head in self.length_heads],
            dim=1,
        )  # [B, 4]

        pred = all_preds.gather(1, length_idx.view(-1, 1)).squeeze(1)
        return pred


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def evaluate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse_score(y_true, y_pred),
        "mae": mae_score(y_true, y_pred),
        "spearman": spearman_score(y_true, y_pred),
    }


def get_diag_channel_indices() -> tuple[list[str], list[int]]:
    builder = RelationTensorBuilder(max_len=5)
    channel_names = list(builder.channel_names)

    diag_indices = [
        i for i, name in enumerate(channel_names)
        if name.lower().startswith("diag_")
    ]

    if not diag_indices:
        raise ValueError("No diag_ channels found.")

    return channel_names, diag_indices


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


def extract_diag_inputs(
    batch: dict,
    diag_indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x_hand"].to(device)
    pair_mask_2d = batch["pair_mask"].to(device)

    x_diag = x[:, diag_indices, :, :]  # [B, C_diag, 5, 5]

    # torch.diagonal -> [B, C_diag, 5]
    diag_tokens = (
        torch.diagonal(x_diag, dim1=-2, dim2=-1)
        .permute(0, 2, 1)
        .contiguous()
    )  # [B, 5, C_diag]

    diag_mask = torch.diagonal(
        pair_mask_2d[:, 0, :, :],
        dim1=-2,
        dim2=-1,
    ).contiguous()  # [B, 5]

    length = batch["length"].to(device).view(-1).long()

    return diag_tokens, diag_mask, length


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    sample_weight = sample_weight.view(-1)

    loss = (pred - target) ** 2
    loss = loss * sample_weight
    return loss.sum() / (sample_weight.sum() + 1e-8)


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    min_label_diff: float = 0.20,
) -> torch.Tensor:
    """
    小权重排序损失。
    目的不是替代回归，而是略微改善 Spearman / Top-K。
    """
    pred = pred.view(-1)
    target = target.view(-1)
    sample_weight = sample_weight.view(-1)

    n = pred.shape[0]

    if n < 2:
        return pred.new_tensor(0.0)

    diff_y = target.view(n, 1) - target.view(1, n)
    diff_p = pred.view(n, 1) - pred.view(1, n)

    mask = torch.abs(diff_y) >= min_label_diff

    if mask.sum() == 0:
        return pred.new_tensor(0.0)

    sign = torch.sign(diff_y)

    # 希望 sign * diff_p > 0
    loss = torch.nn.functional.softplus(-sign * diff_p)

    w_pair = sample_weight.view(n, 1) * sample_weight.view(1, n)
    loss = loss * w_pair

    return loss[mask].mean()


def compute_train_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    huber_criterion: WeightedHuberLoss,
    mse_alpha: float,
    rank_lambda: float,
) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    sample_weight = sample_weight.view(-1)

    huber = huber_criterion(
        pred=pred,
        target=target,
        sample_weight=sample_weight,
    )

    if mse_alpha > 0:
        mse = weighted_mse_loss(pred, target, sample_weight)
        loss = (1.0 - mse_alpha) * huber + mse_alpha * mse
    else:
        loss = huber

    if rank_lambda > 0:
        rank_loss = pairwise_ranking_loss(
            pred=pred,
            target=target,
            sample_weight=sample_weight,
        )
        loss = loss + rank_lambda * rank_loss

    return loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    huber_criterion: WeightedHuberLoss,
    device: torch.device,
    diag_indices: list[int],
    mse_alpha: float,
    rank_lambda: float,
) -> float:
    model.train()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device_except_x(batch, device)

        diag_tokens, diag_mask, length = extract_diag_inputs(
            batch=batch,
            diag_indices=diag_indices,
            device=device,
        )

        y = batch["label_pIC50"].view(-1)
        sample_weight = batch["sample_weight"].view(-1)

        optimizer.zero_grad()

        pred = model(
            diag_tokens=diag_tokens,
            diag_mask=diag_mask,
            length=length,
        )

        loss = compute_train_loss(
            pred=pred,
            target=y,
            sample_weight=sample_weight,
            huber_criterion=huber_criterion,
            mse_alpha=mse_alpha,
            rank_lambda=rank_lambda,
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
    huber_criterion: WeightedHuberLoss,
    device: torch.device,
    diag_indices: list[int],
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

        diag_tokens, diag_mask, length = extract_diag_inputs(
            batch=batch,
            diag_indices=diag_indices,
            device=device,
        )

        y = batch["label_pIC50"].view(-1)
        sample_weight = batch["sample_weight"].view(-1)

        pred = model(
            diag_tokens=diag_tokens,
            diag_mask=diag_mask,
            length=length,
        )

        loss = huber_criterion(
            pred=pred.view(-1),
            target=y.view(-1),
            sample_weight=sample_weight.view(-1),
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
            lengths.extend(length.detach().cpu().numpy().reshape(-1).tolist())
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


def get_selection_score(val_metrics: dict, select_by: str) -> float:
    if select_by == "val_rmse":
        return float(val_metrics["rmse"])

    if select_by == "combined":
        # 越小越好。
        # 以 RMSE 为主，同时轻微考虑 MAE 和 Spearman。
        return float(
            val_metrics["rmse"]
            + 0.25 * val_metrics["mae"]
            - 0.15 * val_metrics["spearman"]
        )

    raise ValueError(f"Unknown select_by: {select_by}")


def train_one_run(
    cfg: dict,
    split_seed: int,
    run_seed: int,
    variant_name: str,
    variant_cfg: dict,
    diag_indices: list[int],
    diag_names: list[str],
    artifacts_dir: Path,
) -> dict:
    effective_seed = split_seed * 1000 + run_seed
    set_seed(effective_seed)
    ensure_dir(artifacts_dir)

    device = choose_device(cfg["train"]["device"])

    split_dir = project_root / cfg["paths"]["split_dir"]

    train_loader = build_loader(
        split_dir / "train_joint.csv",
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        shuffle=True,
    )

    val_loader = build_loader(
        split_dir / "val_main.csv",
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        shuffle=False,
    )

    test_loader = build_loader(
        split_dir / "test_main.csv",
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        shuffle=False,
    )

    model = A1DiagTokenLenHeadRegressor(
        input_dim=len(diag_indices),
        hidden_dim=int(variant_cfg["hidden_dim"]),
        num_blocks=int(variant_cfg["num_blocks"]),
        dropout=float(variant_cfg["dropout"]),
        max_len=5,
    ).to(device)

    huber_criterion = WeightedHuberLoss(delta=float(cfg["train"]["huber_delta"]))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant_cfg["lr"]),
        weight_decay=float(variant_cfg["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    patience = int(early_cfg.get("patience", 10))
    min_delta = float(early_cfg.get("min_delta", 0.0))

    mse_alpha = float(variant_cfg["mse_alpha"])
    rank_lambda = float(variant_cfg["rank_lambda"])
    select_by = str(variant_cfg["select_by"])

    best_score = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_ckpt = artifacts_dir / "best_model.pt"

    history = []

    save_json(
        {
            "split_seed": split_seed,
            "run_seed": run_seed,
            "effective_seed": effective_seed,
            "variant_name": variant_name,
            "variant_cfg": variant_cfg,
            "n_diag_channels": len(diag_indices),
            "diag_indices": diag_indices,
            "diag_names": diag_names,
        },
        artifacts_dir / "run_config.json",
    )

    print("\n" + "-" * 80)
    print(f"SplitSeed={split_seed} | RunSeed={run_seed} | Variant={variant_name}")
    print(f"Device: {device}")
    print(f"DiagToken dim: {len(diag_indices)}")
    print(f"Variant cfg: {variant_cfg}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            huber_criterion=huber_criterion,
            device=device,
            diag_indices=diag_indices,
            mse_alpha=mse_alpha,
            rank_lambda=rank_lambda,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            huber_criterion=huber_criterion,
            device=device,
            diag_indices=diag_indices,
            save_predictions=False,
        )

        score = get_selection_score(val_metrics, select_by=select_by)

        row = {
            "split_seed": split_seed,
            "run_seed": run_seed,
            "variant": variant_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_spearman": val_metrics["spearman"],
            "selection_score": score,
        }
        history.append(row)

        improved = score < (best_score - min_delta)

        if improved:
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "split_seed": split_seed,
                    "run_seed": run_seed,
                    "variant_name": variant_name,
                    "variant_cfg": variant_cfg,
                    "diag_indices": diag_indices,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                },
                best_ckpt,
            )
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"val_spearman={val_metrics['spearman']:.4f} | "
            f"score={score:.4f}"
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
        huber_criterion=huber_criterion,
        device=device,
        diag_indices=diag_indices,
        save_predictions=True,
    )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        huber_criterion=huber_criterion,
        device=device,
        diag_indices=diag_indices,
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
        "split_seed": split_seed,
        "run_seed": run_seed,
        "effective_seed": effective_seed,
        "variant": variant_name,
        "n_diag_channels": len(diag_indices),
        "hidden_dim": variant_cfg["hidden_dim"],
        "num_blocks": variant_cfg["num_blocks"],
        "dropout": variant_cfg["dropout"],
        "lr": variant_cfg["lr"],
        "weight_decay": variant_cfg["weight_decay"],
        "mse_alpha": variant_cfg["mse_alpha"],
        "rank_lambda": variant_cfg["rank_lambda"],
        "select_by": variant_cfg["select_by"],
        "best_epoch": best_epoch,
        "best_score": best_score,
        "val_loss": val_best["loss"],
        "val_rmse": val_best["rmse"],
        "val_mae": val_best["mae"],
        "val_spearman": val_best["spearman"],
        "test_loss": test_metrics["loss"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_spearman": test_metrics["spearman"],
    }

    save_json(summary, artifacts_dir / "best_summary.json")

    print("\nDiagToken run finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def build_ensemble_for_variant_split(
    split_seed: int,
    variant_name: str,
    run_summaries: list[dict],
    run_dirs: list[Path],
    out_dir: Path,
) -> list[dict]:
    ensure_dir(out_dir)

    val_preds = []
    test_preds = []
    val_rmses = []

    for summary, run_dir in zip(run_summaries, run_dirs):
        val_df = pd.read_csv(run_dir / "val_predictions.csv")
        test_df = pd.read_csv(run_dir / "test_predictions.csv")

        val_preds.append(val_df["y_pred"].values.astype(float))
        test_preds.append(test_df["y_pred"].values.astype(float))
        val_rmses.append(float(summary["val_rmse"]))

    base_val_df = pd.read_csv(run_dirs[0] / "val_predictions.csv")
    base_test_df = pd.read_csv(run_dirs[0] / "test_predictions.csv")

    y_val = base_val_df["y_true"].values.astype(float)
    y_test = base_test_df["y_true"].values.astype(float)

    val_pred_mat = np.vstack(val_preds)
    test_pred_mat = np.vstack(test_preds)

    weights = 1.0 / (np.asarray(val_rmses, dtype=float) ** 2 + 1e-8)
    weights = weights / weights.sum()

    ensemble_outputs = []

    for ensemble_name, val_pred, test_pred in [
        (
            f"{variant_name}-EnsMean",
            val_pred_mat.mean(axis=0),
            test_pred_mat.mean(axis=0),
        ),
        (
            f"{variant_name}-EnsWeighted",
            np.average(val_pred_mat, axis=0, weights=weights),
            np.average(test_pred_mat, axis=0, weights=weights),
        ),
    ]:
        val_metrics = evaluate_arrays(y_val, val_pred)
        test_metrics = evaluate_arrays(y_test, test_pred)

        val_out = base_val_df.copy()
        val_out["y_pred"] = val_pred
        val_out["abs_error"] = np.abs(val_out["y_true"].values.astype(float) - val_pred)

        test_out = base_test_df.copy()
        test_out["y_pred"] = test_pred
        test_out["abs_error"] = np.abs(test_out["y_true"].values.astype(float) - test_pred)

        safe_name = ensemble_name.replace("/", "_")

        val_out.to_csv(
            out_dir / f"{safe_name}_val_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        test_out.to_csv(
            out_dir / f"{safe_name}_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        row = {
            "split_seed": split_seed,
            "variant": variant_name,
            "model": ensemble_name,
            "ensemble_type": "mean" if ensemble_name.endswith("EnsMean") else "weighted",
            "n_members": len(run_dirs),
            "member_val_rmses": json.dumps(val_rmses, ensure_ascii=False),
            "member_weights": json.dumps(weights.tolist(), ensure_ascii=False),
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_spearman": val_metrics["spearman"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_spearman": test_metrics["spearman"],
        }

        ensemble_outputs.append(row)

    return ensemble_outputs


def summarize_single_runs(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df
        .groupby("variant", as_index=False)
        .agg(
            n_diag_channels=("n_diag_channels", "first"),
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


def summarize_ensemble(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df
        .groupby("model", as_index=False)
        .agg(
            variant=("variant", "first"),
            ensemble_type=("ensemble_type", "first"),
            n_members=("n_members", "first"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_spearman_mean=("test_spearman", "mean"),
            test_spearman_std=("test_spearman", "std"),
            val_rmse_mean=("val_rmse", "mean"),
            val_rmse_std=("val_rmse", "std"),
        )
        .sort_values(["test_rmse_mean", "test_mae_mean"], ascending=[True, True])
    )


def main() -> None:
    base_cfg_path = project_root / "configs" / "train_a1_v1.yaml"
    base_cfg = load_yaml(base_cfg_path)

    summary_dir = project_root / "artifacts" / "a1_diag_token_ensemble_summary"
    ensure_dir(summary_dir)

    channel_names, diag_indices = get_diag_channel_indices()
    diag_names = [channel_names[i] for i in diag_indices]

    pd.DataFrame(
        {
            "diag_channel_index": diag_indices,
            "diag_channel_name": diag_names,
        }
    ).to_csv(
        summary_dir / "diag_token_channels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_run_rows = []
    all_ensemble_rows = []

    for split_seed in SPLIT_SEEDS:
        for variant_name, variant_cfg in VARIANTS.items():
            cfg = copy.deepcopy(base_cfg)
            cfg["split"]["seed"] = split_seed
            cfg["paths"]["split_dir"] = f"data/final/splits/seed_{split_seed}"

            run_summaries = []
            run_dirs = []

            for run_seed in RUN_SEEDS:
                run_dir = (
                    project_root
                    / "artifacts"
                    / "a1_diag_token_ensemble"
                    / f"split_seed_{split_seed}"
                    / variant_name
                    / f"run_{run_seed}"
                )

                summary = train_one_run(
                    cfg=cfg,
                    split_seed=split_seed,
                    run_seed=run_seed,
                    variant_name=variant_name,
                    variant_cfg=variant_cfg,
                    diag_indices=diag_indices,
                    diag_names=diag_names,
                    artifacts_dir=run_dir,
                )

                run_summaries.append(summary)
                run_dirs.append(run_dir)
                all_run_rows.append(summary)

                pd.DataFrame(all_run_rows).to_csv(
                    summary_dir / "a1_diag_token_single_runs_raw_partial.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            ens_dir = (
                project_root
                / "artifacts"
                / "a1_diag_token_ensemble"
                / f"split_seed_{split_seed}"
                / variant_name
                / "ensemble"
            )

            ensemble_rows = build_ensemble_for_variant_split(
                split_seed=split_seed,
                variant_name=variant_name,
                run_summaries=run_summaries,
                run_dirs=run_dirs,
                out_dir=ens_dir,
            )

            all_ensemble_rows.extend(ensemble_rows)

            pd.DataFrame(all_ensemble_rows).to_csv(
                summary_dir / "a1_diag_token_ensemble_raw_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    single_raw = pd.DataFrame(all_run_rows)
    ensemble_raw = pd.DataFrame(all_ensemble_rows)

    single_raw.to_csv(
        summary_dir / "a1_diag_token_single_runs_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ensemble_raw.to_csv(
        summary_dir / "a1_diag_token_ensemble_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    single_summary = summarize_single_runs(single_raw)
    ensemble_summary = summarize_ensemble(ensemble_raw)

    single_summary.to_csv(
        summary_dir / "a1_diag_token_single_runs_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ensemble_summary.to_csv(
        summary_dir / "a1_diag_token_ensemble_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("All A1-DiagToken ensemble experiments finished.")
    print(f"Single-run raw saved to: {summary_dir / 'a1_diag_token_single_runs_raw.csv'}")
    print(f"Single-run summary saved to: {summary_dir / 'a1_diag_token_single_runs_summary.csv'}")
    print(f"Ensemble raw saved to: {summary_dir / 'a1_diag_token_ensemble_raw.csv'}")
    print(f"Ensemble summary saved to: {summary_dir / 'a1_diag_token_ensemble_summary.csv'}")
    print("=" * 80)

    print("\nSingle-run summary:")
    print(
        single_summary[
            [
                "variant",
                "n_diag_channels",
                "test_rmse_mean",
                "test_rmse_std",
                "test_mae_mean",
                "test_mae_std",
                "test_spearman_mean",
                "test_spearman_std",
            ]
        ]
    )

    print("\nEnsemble summary:")
    print(
        ensemble_summary[
            [
                "model",
                "variant",
                "ensemble_type",
                "n_members",
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