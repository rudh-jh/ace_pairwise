from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from ace_pre.models.a1_model import A1ACERegressor


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_path(project_root: Path) -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return project_root / "configs" / "train_a1_v1.yaml"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s_true = pd.Series(y_true)
    s_pred = pd.Series(y_pred)
    val = s_true.corr(s_pred, method="spearman")
    return float(val) if pd.notna(val) else 0.0


def build_loader(csv_path: Path, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    builder = RelationTensorBuilder(max_len=5)
    dataset = A1SequenceDataset(csv_path=csv_path, tensor_builder=builder)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=a1_collate_fn,
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model: A1ACERegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: WeightedHuberLoss,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        out = model(
            x_hand=batch["x_hand"],
            pair_mask=batch["pair_mask"],
        )
        loss = criterion(
            pred=out.y_hat,
            target=batch["label_pIC50"],
            sample_weight=batch["sample_weight"],
        )
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

    return running_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: A1ACERegressor,
    loader: DataLoader,
    criterion: WeightedHuberLoss,
    device: torch.device,
) -> dict:
    model.eval()

    running_loss = 0.0
    n_batches = 0

    preds = []
    trues = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        out = model(
            x_hand=batch["x_hand"],
            pair_mask=batch["pair_mask"],
        )
        loss = criterion(
            pred=out.y_hat,
            target=batch["label_pIC50"],
            sample_weight=batch["sample_weight"],
        )

        running_loss += float(loss.item())
        n_batches += 1

        preds.append(out.y_hat.detach().cpu().numpy().reshape(-1))
        trues.append(batch["label_pIC50"].detach().cpu().numpy().reshape(-1))

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    return {
        "loss": running_loss / max(1, n_batches),
        "rmse": rmse_score(y_true, y_pred),
        "mae": mae_score(y_true, y_pred),
        "spearman": spearman_score(y_true, y_pred),
    }


def main() -> None:
    config_path = get_config_path(project_root)
    cfg = load_yaml(config_path)

    split_dir = project_root / cfg["paths"]["split_dir"]
    artifacts_dir = project_root / cfg["paths"]["artifacts_dir"]
    ensure_dir(artifacts_dir)

    seed = int(cfg["split"]["seed"])
    set_seed(seed)

    device = choose_device(cfg["train"]["device"])
    print(f"Using device: {device}")

    train_joint_csv = split_dir / "train_joint.csv"
    val_main_csv = split_dir / "val_main.csv"
    test_main_csv = split_dir / "test_main.csv"

    if not train_joint_csv.exists():
        raise FileNotFoundError(f"Missing file: {train_joint_csv}")
    if not val_main_csv.exists():
        raise FileNotFoundError(f"Missing file: {val_main_csv}")
    if not test_main_csv.exists():
        raise FileNotFoundError(f"Missing file: {test_main_csv}")

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])

    train_loader = build_loader(train_joint_csv, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = build_loader(val_main_csv, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = build_loader(test_main_csv, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    sample_batch = next(iter(train_loader))
    in_channels = int(sample_batch["x_hand"].shape[1])

    model = A1ACERegressor(
        in_channels=in_channels,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_blocks=int(cfg["model"]["num_blocks"]),
        dropout=float(cfg["model"]["dropout"]),
        se_reduction=int(cfg["model"]["se_reduction"]),
        attention_dropout=float(cfg["model"]["attention_dropout"]),
    ).to(device)

    criterion = WeightedHuberLoss(delta=float(cfg["train"]["huber_delta"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    monitor = cfg["selection"]["monitor"]
    mode = cfg["selection"]["mode"]

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    patience = int(early_cfg.get("patience", 10))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    epochs_no_improve = 0

    history = []
    best_metric = float("inf") if mode == "min" else float("-inf")
    best_epoch = -1
    best_ckpt = artifacts_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_spearman": val_metrics["spearman"],
        }
        history.append(row)

        current = row[monitor]
        if mode == "min":
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
                    "config": cfg,
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "in_channels": in_channels,
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

    history_df = pd.DataFrame(history)
    history_df.to_csv(artifacts_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    summary = {
        "best_epoch": best_epoch,
        "best_monitor": monitor,
        "best_metric": best_metric,
        "test_loss": test_metrics["loss"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_spearman": test_metrics["spearman"],
    }

    with open(artifacts_dir / "best_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nTraining finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()