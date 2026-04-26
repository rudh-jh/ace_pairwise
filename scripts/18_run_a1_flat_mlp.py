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
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parents[1]
src_root = project_root / "src"

for p in [project_root, src_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ace_pre.baselines.tensor_flat_features import build_a1_flat_feature_frame
from ace_pre.losses.weighted_huber import WeightedHuberLoss


SEEDS = [42, 52, 62]


MLP_VARIANTS = {
    "A1FlatMLP-small": {
        "hidden_dims": [128, 64],
        "dropout": 0.20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "A1FlatMLP-tiny": {
        "hidden_dims": [64, 32],
        "dropout": 0.10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "A1FlatMLP-wide": {
        "hidden_dims": [256, 128],
        "dropout": 0.30,
        "lr": 8e-4,
        "weight_decay": 2e-4,
    },
}


class FlatDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sample_weight = torch.tensor(sample_weight, dtype=torch.float32)
        self.meta_df = meta_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict:
        row = self.meta_df.iloc[idx]

        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "sample_weight": self.sample_weight[idx],
            "sample_id": str(row.get("sample_id", idx)),
            "sequence": str(row.get("sequence", "")),
            "length": int(row.get("length", -1)),
            "task_role": str(row.get("task_role", "")),
        }


class A1FlatMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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


def make_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_train_df = build_a1_flat_feature_frame(train_df, sequence_col="sequence")
    X_val_df = build_a1_flat_feature_frame(val_df, sequence_col="sequence")
    X_test_df = build_a1_flat_feature_frame(test_df, sequence_col="sequence")

    X_val_df = X_val_df.reindex(columns=X_train_df.columns, fill_value=0.0)
    X_test_df = X_test_df.reindex(columns=X_train_df.columns, fill_value=0.0)

    feature_names = list(X_train_df.columns)

    # 只用训练集统计量做填补和标准化，避免泄漏。
    med = X_train_df.median(axis=0)
    X_train_df = X_train_df.fillna(med)
    X_val_df = X_val_df.fillna(med)
    X_test_df = X_test_df.fillna(med)

    mean = X_train_df.mean(axis=0)
    std = X_train_df.std(axis=0).replace(0, 1.0)

    X_train = ((X_train_df - mean) / std).values.astype(np.float32)
    X_val = ((X_val_df - mean) / std).values.astype(np.float32)
    X_test = ((X_test_df - mean) / std).values.astype(np.float32)

    return X_train, X_val, X_test, feature_names


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    meta_df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = FlatDataset(
        X=X,
        y=y,
        sample_weight=sample_weight,
        meta_df=meta_df,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: WeightedHuberLoss,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        sample_weight = batch["sample_weight"].to(device)

        optimizer.zero_grad()

        pred = model(x)

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
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        sample_weight = batch["sample_weight"].to(device)

        pred = model(x)

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
    artifacts_dir: Path,
) -> dict:
    set_seed(seed)
    ensure_dir(artifacts_dir)

    device = choose_device(cfg["train"]["device"])

    split_dir = project_root / cfg["paths"]["split_dir"]

    train_df = pd.read_csv(split_dir / "train_joint.csv")
    val_df = pd.read_csv(split_dir / "val_main.csv")
    test_df = pd.read_csv(split_dir / "test_main.csv")

    X_train, X_val, X_test, feature_names = make_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    y_train = train_df["label_pIC50"].astype(float).values
    y_val = val_df["label_pIC50"].astype(float).values
    y_test = test_df["label_pIC50"].astype(float).values

    w_train = train_df["sample_weight"].astype(float).values
    w_val = np.ones_like(y_val, dtype=float)
    w_test = np.ones_like(y_test, dtype=float)

    batch_size = int(cfg["train"]["batch_size"])

    train_loader = make_loader(
        X_train,
        y_train,
        w_train,
        train_df,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        X_val,
        y_val,
        w_val,
        val_df,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        X_test,
        y_test,
        w_test,
        test_df,
        batch_size=batch_size,
        shuffle=False,
    )

    input_dim = X_train.shape[1]

    model = A1FlatMLP(
        input_dim=input_dim,
        hidden_dims=variant_cfg["hidden_dims"],
        dropout=float(variant_cfg["dropout"]),
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

    with open(artifacts_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    with open(artifacts_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "variant_name": variant_name,
                "variant_cfg": variant_cfg,
                "input_dim": input_dim,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "-" * 80)
    print(f"Seed={seed} | Variant={variant_name}")
    print(f"Device: {device}")
    print(f"Input dim: {input_dim}")
    print(f"Hidden dims: {variant_cfg['hidden_dims']}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
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
                    "input_dim": input_dim,
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
        save_predictions=True,
    )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
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
        "input_dim": input_dim,
        "hidden_dims": json.dumps(variant_cfg["hidden_dims"], ensure_ascii=False),
        "dropout": variant_cfg["dropout"],
        "lr": variant_cfg["lr"],
        "weight_decay": variant_cfg["weight_decay"],
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

    print("\nA1FlatMLP finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df
        .groupby("variant", as_index=False)
        .agg(
            input_dim=("input_dim", "first"),
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

    summary_dir = project_root / "artifacts" / "a1_flat_mlp_summary"
    ensure_dir(summary_dir)

    all_rows = []

    for seed in SEEDS:
        for variant_name, variant_cfg in MLP_VARIANTS.items():
            cfg = copy.deepcopy(base_cfg)

            cfg["split"]["seed"] = seed
            cfg["paths"]["split_dir"] = f"data/final/splits/seed_{seed}"

            artifacts_dir = (
                project_root
                / "artifacts"
                / "a1_flat_mlp"
                / f"seed_{seed}"
                / variant_name
            )

            summary = train_one_variant(
                cfg=cfg,
                seed=seed,
                variant_name=variant_name,
                variant_cfg=variant_cfg,
                artifacts_dir=artifacts_dir,
            )

            all_rows.append(summary)

            pd.DataFrame(all_rows).to_csv(
                summary_dir / "a1_flat_mlp_raw_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    raw_df = pd.DataFrame(all_rows)

    raw_df.to_csv(
        summary_dir / "a1_flat_mlp_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = summarize_results(raw_df)

    summary_df.to_csv(
        summary_dir / "a1_flat_mlp_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("All A1FlatMLP experiments finished.")
    print(f"Raw results saved to: {summary_dir / 'a1_flat_mlp_raw.csv'}")
    print(f"Summary saved to: {summary_dir / 'a1_flat_mlp_summary.csv'}")
    print("=" * 80)

    print("\nA1FlatMLP summary:")
    print(
        summary_df[
            [
                "variant",
                "input_dim",
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