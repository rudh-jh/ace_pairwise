from __future__ import annotations

import copy
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


SEEDS = [42, 52, 62]

LITE_MODES = [
    "diag_lite",
    "diag_terminal_pair_lite",
    "diag_core_pair_lite",
    "diag_core_terminal_pair_lite",
    "diag_pair_no_heur",
    "full",
]


MODE_DESCRIPTIONS = {
    "diag_lite": "Only keep diagonal residue-level physicochemical and terminal channels.",
    "diag_terminal_pair_lite": "Keep diagonal channels plus terminal/order pairwise structural channels.",
    "diag_core_pair_lite": "Keep diagonal channels plus selected core pairwise physicochemical channels.",
    "diag_core_terminal_pair_lite": "Keep diagonal channels plus selected core pairwise and terminal/order pairwise channels.",
    "diag_pair_no_heur": "Keep all diagonal and all pairwise channels, but remove ACE heuristic channels.",
    "full": "Use all A1 relation tensor channels.",
}


TERMINAL_ORDER_PAIR_KEYWORDS = [
    "pair_terminal_interaction",
    "pair_end_to_end",
    "pair_end_to_internal",
    "pair_forward_order",
    "pair_reverse_order",
]


CORE_PAIR_PHYS_KEYWORDS = [
    "pair_hydrophobicity",
    "pair_charge_product",
    "pair_charge_directional",
    "pair_polarity",
    "pair_volume",
    "pair_aromatic_pair",
    "pair_donor_acceptor",
]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


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
        raise RuntimeError(
            "当前配置为 auto，但没有检测到 CUDA。请安装 CUDA 版 PyTorch，"
            "或者把 configs/train_a1_v1.yaml 里的 train.device 改回 cpu。"
        )

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "你指定了 train.device=cuda，但 torch.cuda.is_available() 为 False。"
            "请检查是否安装了 CUDA 版 PyTorch。"
        )

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


def get_channel_names() -> list[str]:
    builder = RelationTensorBuilder(max_len=5)
    return list(builder.channel_names)


def is_terminal_order_pair_channel(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in TERMINAL_ORDER_PAIR_KEYWORDS)


def is_core_pair_phys_channel(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in CORE_PAIR_PHYS_KEYWORDS)


def resolve_channel_indices(
    channel_names: list[str],
    mode: str,
) -> list[int]:
    keep = []

    for name in channel_names:
        lower = name.lower()

        is_diag = lower.startswith("diag_")
        is_pair = lower.startswith("pair_")
        is_terminal_order_pair = is_terminal_order_pair_channel(lower)
        is_core_pair_phys = is_core_pair_phys_channel(lower)

        if mode == "diag_lite":
            flag = is_diag

        elif mode == "diag_terminal_pair_lite":
            flag = is_diag or is_terminal_order_pair

        elif mode == "diag_core_pair_lite":
            flag = is_diag or is_core_pair_phys

        elif mode == "diag_core_terminal_pair_lite":
            flag = is_diag or is_core_pair_phys or is_terminal_order_pair

        elif mode == "diag_pair_no_heur":
            flag = is_diag or is_pair

        elif mode == "full":
            flag = True

        else:
            raise ValueError(f"Unknown lite mode: {mode}")

        keep.append(flag)

    indices = [i for i, flag in enumerate(keep) if flag]

    if not indices:
        raise ValueError(f"Lite mode '{mode}' selected zero channels.")

    return indices


def make_spatial_mask_modifier(mode: str, max_len: int = 5) -> torch.Tensor | None:
    """
    diag_lite 只使用对角线残基属性，所以 pooling 时只允许对角线位置参与。
    其他模式同时包含 diag 和 pair 通道，继续使用原始 pair_mask。
    """
    if mode == "diag_lite":
        eye = torch.eye(max_len, dtype=torch.float32).view(1, 1, max_len, max_len)
        return eye

    return None


def apply_channel_set_to_batch(
    batch: dict,
    channel_indices: list[int],
    mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_hand = batch["x_hand"][:, channel_indices, :, :].to(device)
    pair_mask = batch["pair_mask"].to(device)

    spatial_modifier = make_spatial_mask_modifier(mode, max_len=x_hand.shape[-1])

    if spatial_modifier is not None:
        spatial_modifier = spatial_modifier.to(device)
        pair_mask = pair_mask * spatial_modifier

    return x_hand, pair_mask


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


def train_one_epoch(
    model: A1ACERegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: WeightedHuberLoss,
    device: torch.device,
    channel_indices: list[int],
    lite_mode: str,
) -> float:
    model.train()

    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device_except_x(batch, device)

        x_hand, pair_mask = apply_channel_set_to_batch(
            batch=batch,
            channel_indices=channel_indices,
            mode=lite_mode,
            device=device,
        )

        optimizer.zero_grad()

        out = model(
            x_hand=x_hand,
            pair_mask=pair_mask,
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
    channel_indices: list[int],
    lite_mode: str,
    save_predictions: bool = False,
) -> dict:
    model.eval()

    running_loss = 0.0
    n_batches = 0

    preds = []
    trues = []

    sample_ids = []
    sequences = []
    lengths = []
    task_roles = []

    for batch in loader:
        batch = move_batch_to_device_except_x(batch, device)

        x_hand, pair_mask = apply_channel_set_to_batch(
            batch=batch,
            channel_indices=channel_indices,
            mode=lite_mode,
            device=device,
        )

        out = model(
            x_hand=x_hand,
            pair_mask=pair_mask,
        )

        loss = criterion(
            pred=out.y_hat,
            target=batch["label_pIC50"],
            sample_weight=batch["sample_weight"],
        )

        running_loss += float(loss.item())
        n_batches += 1

        pred_np = out.y_hat.detach().cpu().numpy().reshape(-1)
        true_np = batch["label_pIC50"].detach().cpu().numpy().reshape(-1)

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
        "loss": running_loss / max(1, n_batches),
        "rmse": rmse_score(y_true, y_pred),
        "mae": mae_score(y_true, y_pred),
        "spearman": spearman_score(y_true, y_pred),
    }

    if save_predictions:
        pred_df = pd.DataFrame({
            "sample_id": sample_ids,
            "sequence": sequences,
            "length": lengths,
            "task_role": task_roles,
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": np.abs(y_true - y_pred),
        })
        result["pred_df"] = pred_df

    return result


def train_one_lite_mode(
    cfg: dict,
    seed: int,
    lite_mode: str,
    channel_names: list[str],
    channel_indices: list[int],
    artifacts_dir: Path,
) -> dict:
    set_seed(seed)

    split_dir = project_root / cfg["paths"]["split_dir"]

    train_joint_csv = split_dir / "train_joint.csv"
    val_main_csv = split_dir / "val_main.csv"
    test_main_csv = split_dir / "test_main.csv"

    if not train_joint_csv.exists():
        raise FileNotFoundError(f"Missing file: {train_joint_csv}")
    if not val_main_csv.exists():
        raise FileNotFoundError(f"Missing file: {val_main_csv}")
    if not test_main_csv.exists():
        raise FileNotFoundError(f"Missing file: {test_main_csv}")

    ensure_dir(artifacts_dir)

    device = choose_device(cfg["train"]["device"])

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])

    train_loader = build_loader(
        train_joint_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_loader = build_loader(
        val_main_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    test_loader = build_loader(
        test_main_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    selected_channel_names = [channel_names[i] for i in channel_indices]
    in_channels = len(channel_indices)

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
    select_mode = cfg["selection"]["mode"]

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    patience = int(early_cfg.get("patience", 10))
    min_delta = float(early_cfg.get("min_delta", 0.0))

    epochs_no_improve = 0
    history = []

    best_metric = float("inf") if select_mode == "min" else float("-inf")
    best_epoch = -1
    best_ckpt = artifacts_dir / "best_model.pt"

    channel_info = {
        "seed": seed,
        "lite_mode": lite_mode,
        "description": MODE_DESCRIPTIONS.get(lite_mode, ""),
        "n_channels": in_channels,
        "channel_indices": channel_indices,
        "channel_names": selected_channel_names,
    }

    with open(artifacts_dir / "channel_info.json", "w", encoding="utf-8") as f:
        json.dump(channel_info, f, ensure_ascii=False, indent=2)

    print("\n" + "-" * 80)
    print(f"Seed={seed} | LiteMode={lite_mode}")
    print(f"Device: {device}")
    print(f"Channels: {in_channels}/{len(channel_names)}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            channel_indices=channel_indices,
            lite_mode=lite_mode,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            channel_indices=channel_indices,
            lite_mode=lite_mode,
            save_predictions=False,
        )

        row = {
            "seed": seed,
            "lite_mode": lite_mode,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_spearman": val_metrics["spearman"],
            "n_channels": in_channels,
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
                    "config": cfg,
                    "seed": seed,
                    "lite_mode": lite_mode,
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "in_channels": in_channels,
                    "channel_indices": channel_indices,
                    "channel_names": selected_channel_names,
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
    history_df.to_csv(
        artifacts_dir / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_best_metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        channel_indices=channel_indices,
        lite_mode=lite_mode,
        save_predictions=True,
    )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        channel_indices=channel_indices,
        lite_mode=lite_mode,
        save_predictions=True,
    )

    val_pred_df = val_best_metrics.pop("pred_df")
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
        "lite_mode": lite_mode,
        "description": MODE_DESCRIPTIONS.get(lite_mode, ""),
        "n_channels": in_channels,
        "best_epoch": best_epoch,
        "best_monitor": monitor,
        "best_metric": best_metric,
        "val_loss": val_best_metrics["loss"],
        "val_rmse": val_best_metrics["rmse"],
        "val_mae": val_best_metrics["mae"],
        "val_spearman": val_best_metrics["spearman"],
        "test_loss": test_metrics["loss"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_spearman": test_metrics["spearman"],
    }

    with open(artifacts_dir / "best_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nLite channel set finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def summarize_lite_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df
        .groupby(["lite_mode", "description"], as_index=False)
        .agg(
            n_channels=("n_channels", "first"),
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

    summary_dir = project_root / "artifacts" / "a1_lite_channel_sets_summary"
    ensure_dir(summary_dir)

    cfg_dir = project_root / "artifacts" / "a1_lite_channel_sets_configs"
    ensure_dir(cfg_dir)

    channel_names = get_channel_names()

    channel_df = pd.DataFrame({
        "channel_index": list(range(len(channel_names))),
        "channel_name": channel_names,
    })

    channel_df.to_csv(
        summary_dir / "a1_all_channel_names.csv",
        index=False,
        encoding="utf-8-sig",
    )

    channel_set_rows = []

    for lite_mode in LITE_MODES:
        channel_indices = resolve_channel_indices(
            channel_names=channel_names,
            mode=lite_mode,
        )
        selected_names = [channel_names[i] for i in channel_indices]

        for name in selected_names:
            channel_set_rows.append({
                "lite_mode": lite_mode,
                "description": MODE_DESCRIPTIONS.get(lite_mode, ""),
                "channel_index": channel_names.index(name),
                "channel_name": name,
            })

    pd.DataFrame(channel_set_rows).to_csv(
        summary_dir / "a1_lite_channel_sets_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_rows = []

    for seed in SEEDS:
        for lite_mode in LITE_MODES:
            cfg = copy.deepcopy(base_cfg)

            cfg["split"]["seed"] = seed
            cfg["paths"]["split_dir"] = f"data/final/splits/seed_{seed}"

            cfg["paths"]["artifacts_dir"] = (
                f"artifacts/a1_lite_channel_sets/seed_{seed}/{lite_mode}"
            )

            seed_mode_cfg_path = (
                cfg_dir / f"train_a1_seed_{seed}_{lite_mode}.yaml"
            )
            save_yaml(cfg, seed_mode_cfg_path)

            channel_indices = resolve_channel_indices(
                channel_names=channel_names,
                mode=lite_mode,
            )

            artifacts_dir = project_root / cfg["paths"]["artifacts_dir"]

            summary = train_one_lite_mode(
                cfg=cfg,
                seed=seed,
                lite_mode=lite_mode,
                channel_names=channel_names,
                channel_indices=channel_indices,
                artifacts_dir=artifacts_dir,
            )

            all_rows.append(summary)

            raw_partial_df = pd.DataFrame(all_rows)
            raw_partial_df.to_csv(
                summary_dir / "a1_lite_channel_sets_raw_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    raw_df = pd.DataFrame(all_rows)

    raw_df.to_csv(
        summary_dir / "a1_lite_channel_sets_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = summarize_lite_results(raw_df)

    summary_df.to_csv(
        summary_dir / "a1_lite_channel_sets_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("All A1-Lite channel set experiments finished.")
    print(f"Raw results saved to: {summary_dir / 'a1_lite_channel_sets_raw.csv'}")
    print(f"Summary saved to: {summary_dir / 'a1_lite_channel_sets_summary.csv'}")
    print("=" * 80)

    print("\nA1-Lite channel set summary:")
    print(
        summary_df[
            [
                "lite_mode",
                "n_channels",
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