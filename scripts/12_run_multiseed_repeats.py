from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


SEEDS = [42, 52, 62]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_script(python_exe: str, script_path: Path, config_path: Path) -> None:
    cmd = [python_exe, str(script_path), str(config_path)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    base_cfg_path = project_root / "configs" / "train_a1_v1.yaml"
    base_cfg = load_yaml(base_cfg_path)

    cfg_dir = project_root / "artifacts" / "multiseed_configs"
    summary_dir = project_root / "artifacts" / "multiseed_summary"
    ensure_dir(cfg_dir)
    ensure_dir(summary_dir)

    a1_rows = []
    baseline_rows = []
    a1flat_rows = []

    for seed in SEEDS:
        cfg = copy.deepcopy(base_cfg)

        cfg["split"]["seed"] = seed
        cfg["paths"]["split_dir"] = f"data/final/splits/seed_{seed}"
        cfg["paths"]["artifacts_dir"] = f"artifacts/a1_seed_{seed}"
        cfg["paths"]["baseline_artifacts_dir"] = f"artifacts/baselines_seed_{seed}"
        cfg["paths"]["a1flat_artifacts_dir"] = f"artifacts/a1flat_baselines_seed_{seed}"

        seed_cfg_path = cfg_dir / f"train_a1_seed_{seed}.yaml"
        save_yaml(cfg, seed_cfg_path)

        run_script(python_exe, project_root / "scripts" / "08_build_splits.py", seed_cfg_path)
        run_script(python_exe, project_root / "scripts" / "09_train_a1.py", seed_cfg_path)
        run_script(python_exe, project_root / "scripts" / "10_run_tabular_baselines.py", seed_cfg_path)
        run_script(python_exe, project_root / "scripts" / "11_run_a1flat_baselines.py", seed_cfg_path)

        # 汇总 A1
        a1_summary_path = project_root / cfg["paths"]["artifacts_dir"] / "best_summary.json"
        with open(a1_summary_path, "r", encoding="utf-8") as f:
            a1_summary = json.load(f)

        a1_rows.append({
            "seed": seed,
            "best_epoch": a1_summary["best_epoch"],
            "val_monitor": a1_summary["best_metric"],
            "test_rmse": a1_summary["test_rmse"],
            "test_mae": a1_summary["test_mae"],
            "test_spearman": a1_summary["test_spearman"],
        })

        # 汇总简单 baseline
        baseline_path = project_root / cfg["paths"]["baseline_artifacts_dir"] / "baseline_results.csv"
        baseline_df = pd.read_csv(baseline_path)
        baseline_df["seed"] = seed
        baseline_rows.append(baseline_df)

        # 汇总 A1Flat baseline
        a1flat_path = project_root / cfg["paths"]["a1flat_artifacts_dir"] / "a1flat_baseline_results.csv"
        a1flat_df = pd.read_csv(a1flat_path)
        a1flat_df["seed"] = seed
        a1flat_rows.append(a1flat_df)

    # 保存逐 seed 原始结果
    a1_df = pd.DataFrame(a1_rows)
    a1_df.to_csv(summary_dir / "a1_multiseed_raw.csv", index=False, encoding="utf-8-sig")

    baseline_all = pd.concat(baseline_rows, ignore_index=True)
    baseline_all.to_csv(summary_dir / "baseline_multiseed_raw.csv", index=False, encoding="utf-8-sig")

    a1flat_all = pd.concat(a1flat_rows, ignore_index=True)
    a1flat_all.to_csv(summary_dir / "a1flat_multiseed_raw.csv", index=False, encoding="utf-8-sig")

    # 汇总均值与标准差
    a1_stats = pd.DataFrame([
        {
            "model": "A1",
            "test_rmse_mean": a1_df["test_rmse"].mean(),
            "test_rmse_std": a1_df["test_rmse"].std(ddof=1),
            "test_mae_mean": a1_df["test_mae"].mean(),
            "test_mae_std": a1_df["test_mae"].std(ddof=1),
            "test_spearman_mean": a1_df["test_spearman"].mean(),
            "test_spearman_std": a1_df["test_spearman"].std(ddof=1),
        }
    ])

    baseline_stats = (
        baseline_all[baseline_all["split"] == "test"]
        .groupby("model", as_index=False)
        .agg(
            test_rmse_mean=("rmse", "mean"),
            test_rmse_std=("rmse", "std"),
            test_mae_mean=("mae", "mean"),
            test_mae_std=("mae", "std"),
            test_spearman_mean=("spearman", "mean"),
            test_spearman_std=("spearman", "std"),
        )
    )

    a1flat_stats = (
        a1flat_all[a1flat_all["split"] == "test"]
        .groupby("model", as_index=False)
        .agg(
            test_rmse_mean=("rmse", "mean"),
            test_rmse_std=("rmse", "std"),
            test_mae_mean=("mae", "mean"),
            test_mae_std=("mae", "std"),
            test_spearman_mean=("spearman", "mean"),
            test_spearman_std=("spearman", "std"),
        )
    )

    combined = pd.concat([a1_stats, baseline_stats, a1flat_stats], ignore_index=True)
    combined.to_csv(summary_dir / "multiseed_comparison_summary.csv", index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved to: {summary_dir / 'multiseed_comparison_summary.csv'}")


if __name__ == "__main__":
    main()