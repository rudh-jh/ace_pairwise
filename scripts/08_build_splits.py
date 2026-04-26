from __future__ import annotations

import sys
from pathlib import Path
import yaml
import pandas as pd

def get_config_path(project_root: Path) -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return project_root / "configs" / "train_a1_v1.yaml"

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    # cfg = load_yaml(project_root / "configs" / "train_a1_v1.yaml")
    config_path = get_config_path(project_root)
    cfg = load_yaml(config_path)

    main_csv = project_root / cfg["paths"]["main_csv"]
    aux_csv = project_root / cfg["paths"]["aux_csv"]
    split_dir = project_root / cfg["paths"]["split_dir"]

    seed = int(cfg["split"]["seed"])
    val_ratio = float(cfg["split"]["val_ratio"])
    test_ratio = float(cfg["split"]["test_ratio"])

    ensure_dir(split_dir)

    main_df = pd.read_csv(main_csv)
    aux_df = pd.read_csv(aux_csv)

    main_df = main_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(main_df)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError("Not enough main samples to create train/val/test split.")

    train_main = main_df.iloc[:n_train].copy()
    val_main = main_df.iloc[n_train:n_train + n_val].copy()
    test_main = main_df.iloc[n_train + n_val:].copy()

    train_joint = pd.concat([train_main, aux_df], ignore_index=True)
    train_joint = train_joint.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    save_df(train_main, split_dir / "train_main.csv")
    save_df(val_main, split_dir / "val_main.csv")
    save_df(test_main, split_dir / "test_main.csv")
    save_df(train_joint, split_dir / "train_joint.csv")
    save_df(aux_df, split_dir / "aux_full.csv")

    summary = pd.DataFrame([
        {"metric": "train_main_rows", "value": len(train_main)},
        {"metric": "val_main_rows", "value": len(val_main)},
        {"metric": "test_main_rows", "value": len(test_main)},
        {"metric": "aux_rows", "value": len(aux_df)},
        {"metric": "train_joint_rows", "value": len(train_joint)},
    ])
    save_df(summary, split_dir / "split_summary.csv")

    print("Done.")
    print(f"Split dir: {split_dir}")


if __name__ == "__main__":
    main()