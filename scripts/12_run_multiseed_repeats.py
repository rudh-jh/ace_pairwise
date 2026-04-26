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


def summarize_test_results(
    df: pd.DataFrame,
    group_cols: list[str],
    experiment_group: str,
) -> pd.DataFrame:
    """
    对 test split 的结果做 mean/std 汇总。
    df 需要包含：
    split, rmse, mae, spearman, 以及 group_cols 中指定的列。
    """
    test_df = df[df["split"] == "test"].copy()

    summary = (
        test_df
        .groupby(group_cols, as_index=False)
        .agg(
            test_rmse_mean=("rmse", "mean"),
            test_rmse_std=("rmse", "std"),
            test_mae_mean=("mae", "mean"),
            test_mae_std=("mae", "std"),
            test_spearman_mean=("spearman", "mean"),
            test_spearman_std=("spearman", "std"),
        )
    )

    summary.insert(0, "experiment_group", experiment_group)
    return summary


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
    fair_rows = []

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print(f"Running seed: {seed}")
        print("=" * 80)

        cfg = copy.deepcopy(base_cfg)

        cfg["split"]["seed"] = seed
        cfg["paths"]["split_dir"] = f"data/final/splits/seed_{seed}"

        # A1 主模型输出目录
        cfg["paths"]["artifacts_dir"] = f"artifacts/a1_seed_{seed}"

        # 普通 tabular baseline 输出目录
        cfg["paths"]["baseline_artifacts_dir"] = f"artifacts/baselines_seed_{seed}"

        # A1 张量展平 baseline 输出目录
        cfg["paths"]["a1flat_artifacts_dir"] = f"artifacts/a1flat_baselines_seed_{seed}"

        # fair feature baseline 输出目录
        cfg["paths"]["fair_baseline_artifacts_dir"] = f"artifacts/fair_baselines_seed_{seed}"

        seed_cfg_path = cfg_dir / f"train_a1_seed_{seed}.yaml"
        save_yaml(cfg, seed_cfg_path)

        # 1. 构建 split
        run_script(
            python_exe,
            project_root / "scripts" / "08_build_splits.py",
            seed_cfg_path,
        )

        # 2. 训练 A1
        run_script(
            python_exe,
            project_root / "scripts" / "09_train_a1.py",
            seed_cfg_path,
        )

        # 3. 普通 tabular baseline
        run_script(
            python_exe,
            project_root / "scripts" / "10_run_tabular_baselines.py",
            seed_cfg_path,
        )

        # 4. A1Flat baseline
        run_script(
            python_exe,
            project_root / "scripts" / "11_run_a1flat_baselines.py",
            seed_cfg_path,
        )

        # 5. Fair feature baseline
        # 包括：
        # MeanPredictor
        # SeqOnly-Ridge/SVR/CatBoost
        # PhysChemOnly-Ridge/SVR/CatBoost
        # Descriptor-Ridge/SVR/CatBoost
        # A1FlatFull-Ridge/SVR/CatBoost
        run_script(
            python_exe,
            project_root / "scripts" / "13_run_fair_feature_baselines.py",
            seed_cfg_path,
        )

        # ------------------------------------------------------------------
        # 汇总 A1
        # ------------------------------------------------------------------
        a1_summary_path = (
            project_root
            / cfg["paths"]["artifacts_dir"]
            / "best_summary.json"
        )

        if not a1_summary_path.exists():
            raise FileNotFoundError(f"A1 summary not found: {a1_summary_path}")

        with open(a1_summary_path, "r", encoding="utf-8") as f:
            a1_summary = json.load(f)

        a1_rows.append({
            "seed": seed,
            "model": "A1",
            "best_epoch": a1_summary["best_epoch"],
            "val_monitor": a1_summary["best_metric"],
            "test_rmse": a1_summary["test_rmse"],
            "test_mae": a1_summary["test_mae"],
            "test_spearman": a1_summary["test_spearman"],
        })

        # ------------------------------------------------------------------
        # 汇总普通 baseline
        # ------------------------------------------------------------------
        baseline_path = (
            project_root
            / cfg["paths"]["baseline_artifacts_dir"]
            / "baseline_results.csv"
        )

        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline results not found: {baseline_path}")

        baseline_df = pd.read_csv(baseline_path)
        baseline_df["seed"] = seed
        baseline_rows.append(baseline_df)

        # ------------------------------------------------------------------
        # 汇总 A1Flat baseline
        # ------------------------------------------------------------------
        a1flat_path = (
            project_root
            / cfg["paths"]["a1flat_artifacts_dir"]
            / "a1flat_baseline_results.csv"
        )

        if not a1flat_path.exists():
            raise FileNotFoundError(f"A1Flat results not found: {a1flat_path}")

        a1flat_df = pd.read_csv(a1flat_path)
        a1flat_df["seed"] = seed
        a1flat_rows.append(a1flat_df)

        # ------------------------------------------------------------------
        # 汇总 fair feature baseline
        # ------------------------------------------------------------------
        fair_path = (
            project_root
            / cfg["paths"]["fair_baseline_artifacts_dir"]
            / "fair_feature_baseline_results.csv"
        )

        if not fair_path.exists():
            raise FileNotFoundError(f"Fair feature results not found: {fair_path}")

        fair_df = pd.read_csv(fair_path)
        fair_df["seed"] = seed
        fair_rows.append(fair_df)

    # ======================================================================
    # 保存逐 seed 原始结果
    # ======================================================================
    a1_df = pd.DataFrame(a1_rows)
    a1_df.to_csv(
        summary_dir / "a1_multiseed_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_all = pd.concat(baseline_rows, ignore_index=True)
    baseline_all.to_csv(
        summary_dir / "baseline_multiseed_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    a1flat_all = pd.concat(a1flat_rows, ignore_index=True)
    a1flat_all.to_csv(
        summary_dir / "a1flat_multiseed_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fair_all = pd.concat(fair_rows, ignore_index=True)
    fair_all.to_csv(
        summary_dir / "fair_feature_multiseed_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ======================================================================
    # 汇总 A1 mean/std
    # ======================================================================
    a1_stats = pd.DataFrame([
        {
            "experiment_group": "a1_model",
            "model": "A1",
            "feature_set": "A1StructuredTensor",
            "n_features": pd.NA,
            "test_rmse_mean": a1_df["test_rmse"].mean(),
            "test_rmse_std": a1_df["test_rmse"].std(ddof=1),
            "test_mae_mean": a1_df["test_mae"].mean(),
            "test_mae_std": a1_df["test_mae"].std(ddof=1),
            "test_spearman_mean": a1_df["test_spearman"].mean(),
            "test_spearman_std": a1_df["test_spearman"].std(ddof=1),
        }
    ])

    # ======================================================================
    # 汇总普通 baseline mean/std
    # ======================================================================
    baseline_stats = summarize_test_results(
        df=baseline_all,
        group_cols=["model"],
        experiment_group="tabular_baseline",
    )
    baseline_stats["feature_set"] = "OriginalDescriptor"
    baseline_stats["n_features"] = pd.NA

    # 调整列顺序
    baseline_stats = baseline_stats[
        [
            "experiment_group",
            "model",
            "feature_set",
            "n_features",
            "test_rmse_mean",
            "test_rmse_std",
            "test_mae_mean",
            "test_mae_std",
            "test_spearman_mean",
            "test_spearman_std",
        ]
    ]

    # ======================================================================
    # 汇总 A1Flat baseline mean/std
    # ======================================================================
    a1flat_stats = summarize_test_results(
        df=a1flat_all,
        group_cols=["model"],
        experiment_group="a1flat_baseline",
    )
    a1flat_stats["feature_set"] = "A1FlatOriginal"
    a1flat_stats["n_features"] = pd.NA

    # 调整列顺序
    a1flat_stats = a1flat_stats[
        [
            "experiment_group",
            "model",
            "feature_set",
            "n_features",
            "test_rmse_mean",
            "test_rmse_std",
            "test_mae_mean",
            "test_mae_std",
            "test_spearman_mean",
            "test_spearman_std",
        ]
    ]

    # ======================================================================
    # 汇总 fair feature baseline mean/std
    # ======================================================================
    # fair feature baseline 里面有 feature_set 和 n_features，
    # 所以需要按 model + feature_set 汇总。
    fair_test = fair_all[fair_all["split"] == "test"].copy()

    fair_stats = (
        fair_test
        .groupby(["model", "feature_set"], as_index=False)
        .agg(
            n_features=("n_features", "first"),
            test_rmse_mean=("rmse", "mean"),
            test_rmse_std=("rmse", "std"),
            test_mae_mean=("mae", "mean"),
            test_mae_std=("mae", "std"),
            test_spearman_mean=("spearman", "mean"),
            test_spearman_std=("spearman", "std"),
        )
    )

    fair_stats.insert(0, "experiment_group", "fair_feature_baseline")

    fair_stats = fair_stats[
        [
            "experiment_group",
            "model",
            "feature_set",
            "n_features",
            "test_rmse_mean",
            "test_rmse_std",
            "test_mae_mean",
            "test_mae_std",
            "test_spearman_mean",
            "test_spearman_std",
        ]
    ]

    fair_stats = fair_stats.sort_values(
        ["test_rmse_mean", "test_mae_mean"],
        ascending=[True, True],
    )

    fair_stats.to_csv(
        summary_dir / "fair_feature_multiseed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ======================================================================
    # 保存原来的总表：A1 + 普通 baseline + A1Flat
    # ======================================================================
    original_combined = pd.concat(
        [
            a1_stats,
            baseline_stats,
            a1flat_stats,
        ],
        ignore_index=True,
    )

    original_combined = original_combined.sort_values(
        ["test_rmse_mean", "test_mae_mean"],
        ascending=[True, True],
    )

    original_combined.to_csv(
        summary_dir / "multiseed_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ======================================================================
    # 保存新的完整总表：A1 + 普通 baseline + A1Flat + fair feature baseline
    # ======================================================================
    full_combined = pd.concat(
        [
            a1_stats,
            baseline_stats,
            a1flat_stats,
            fair_stats,
        ],
        ignore_index=True,
    )

    full_combined = full_combined.sort_values(
        ["test_rmse_mean", "test_mae_mean"],
        ascending=[True, True],
    )

    full_combined.to_csv(
        summary_dir / "multiseed_comparison_summary_with_fair.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDone.")
    print(f"Saved original summary to: {summary_dir / 'multiseed_comparison_summary.csv'}")
    print(f"Saved fair feature summary to: {summary_dir / 'fair_feature_multiseed_summary.csv'}")
    print(f"Saved full summary to: {summary_dir / 'multiseed_comparison_summary_with_fair.csv'}")

    print("\nFair feature baseline summary:")
    print(
        fair_stats[
            [
                "model",
                "feature_set",
                "n_features",
                "test_rmse_mean",
                "test_rmse_std",
                "test_mae_mean",
                "test_mae_std",
                "test_spearman_mean",
                "test_spearman_std",
            ]
        ]
    )

    print("\nFull comparison summary:")
    print(
        full_combined[
            [
                "experiment_group",
                "model",
                "feature_set",
                "n_features",
                "test_rmse_mean",
                "test_mae_mean",
                "test_spearman_mean",
            ]
        ]
    )


if __name__ == "__main__":
    main()