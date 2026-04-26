from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEEDS = [42, 52, 62]

FAIR_MODELS = [
    "A1FlatFull-SVR",
    "SeqOnly-SVR",
    "SeqOnly-CatBoost",
    "PhysChemOnly-SVR",
    "Descriptor-SVR",
    "Descriptor-CatBoost",
    "MeanPredictor",
]

LITE_MODELS = {
    "A1-DiagLite": "diag_lite",
    "A1-DiagCorePairLite": "diag_core_pair_lite",
}

TOP_KS = [5, 10, 20]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    val = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(val) if pd.notna(val) else np.nan


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "bias_mean": float(np.mean(y_pred - y_true)),
        "abs_error_median": float(np.median(np.abs(y_pred - y_true))),
    }


def activity_bin(y: float) -> str:
    if y >= 6.0:
        return "strong_pIC50_ge_6"
    if y >= 5.0:
        return "medium_5_to_6"
    return "weak_lt_5"


def load_seed_predictions(seed: int) -> pd.DataFrame:
    split_path = PROJECT_ROOT / "data" / "final" / "splits" / f"seed_{seed}" / "test_main.csv"

    if not split_path.exists():
        raise FileNotFoundError(f"Missing test split: {split_path}")

    base_df = pd.read_csv(split_path).reset_index(drop=True)

    if "sample_id" not in base_df.columns:
        base_df["sample_id"] = [f"seed{seed}_row{i}" for i in range(len(base_df))]

    if "label_pIC50" not in base_df.columns:
        raise ValueError(f"test_main.csv must contain label_pIC50: {split_path}")

    out = pd.DataFrame({
        "seed": seed,
        "row_index": np.arange(len(base_df)),
        "sample_id": base_df["sample_id"].astype(str),
        "sequence": base_df["sequence"].astype(str),
        "length": base_df["length"].astype(int),
        "y_true": base_df["label_pIC50"].astype(float),
    })

    # 尽量保留一些后续分析可能用到的元信息
    for col in [
        "source_db",
        "source_count",
        "citation_ids",
        "quality_tier",
        "quality_flag",
        "conflict_flag",
        "sample_weight",
        "task_role",
    ]:
        if col in base_df.columns:
            out[col] = base_df[col]

    # 读取 fair baseline 预测
    fair_pred_path = (
        PROJECT_ROOT
        / "artifacts"
        / f"fair_baselines_seed_{seed}"
        / "fair_feature_test_predictions.csv"
    )

    if not fair_pred_path.exists():
        raise FileNotFoundError(f"Missing fair predictions: {fair_pred_path}")

    fair_pred_df = pd.read_csv(fair_pred_path).reset_index(drop=True)

    if len(fair_pred_df) != len(out):
        raise ValueError(
            f"Row mismatch for seed {seed}: "
            f"test_main={len(out)}, fair_pred={len(fair_pred_df)}"
        )

    for model in FAIR_MODELS:
        if model in fair_pred_df.columns:
            out[model] = fair_pred_df[model].astype(float).values
        else:
            print(f"[WARN] Model column not found in fair predictions: {model}")

    # 读取 A1-Lite 系列预测
    for model_name, lite_mode in LITE_MODELS.items():
        lite_pred_path = (
            PROJECT_ROOT
            / "artifacts"
            / "a1_lite_channel_sets"
            / f"seed_{seed}"
            / lite_mode
            / "test_predictions.csv"
        )

        if not lite_pred_path.exists():
            print(f"[WARN] Missing lite prediction file: {lite_pred_path}")
            continue

        lite_pred_df = pd.read_csv(lite_pred_path).reset_index(drop=True)

        if len(lite_pred_df) != len(out):
            raise ValueError(
                f"Row mismatch for seed {seed}, {model_name}: "
                f"test_main={len(out)}, lite_pred={len(lite_pred_df)}"
            )

        out[model_name] = lite_pred_df["y_pred"].astype(float).values

    return out


def build_long_predictions(wide_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        c for c in wide_df.columns
        if c not in FAIR_MODELS and c not in LITE_MODELS.keys()
    ]

    model_cols = [c for c in FAIR_MODELS if c in wide_df.columns]
    model_cols += [c for c in LITE_MODELS.keys() if c in wide_df.columns]

    rows = []

    for model in model_cols:
        tmp = wide_df[meta_cols].copy()
        tmp["model"] = model
        tmp["y_pred"] = wide_df[model].astype(float).values
        tmp["error"] = tmp["y_pred"] - tmp["y_true"]
        tmp["abs_error"] = np.abs(tmp["error"])
        tmp["activity_bin"] = tmp["y_true"].apply(activity_bin)
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def compute_overall_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (seed, model), g in long_df.groupby(["seed", "model"]):
        metrics = regression_metrics(
            g["y_true"].values.astype(float),
            g["y_pred"].values.astype(float),
        )

        rows.append({
            "seed": seed,
            "model": model,
            "n": len(g),
            **metrics,
        })

    return pd.DataFrame(rows)


def compute_group_metrics(long_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []

    for (seed, model, group_value), g in long_df.groupby(["seed", "model", group_col]):
        if len(g) == 0:
            continue

        metrics = regression_metrics(
            g["y_true"].values.astype(float),
            g["y_pred"].values.astype(float),
        )

        rows.append({
            "seed": seed,
            "model": model,
            "group_col": group_col,
            "group_value": group_value,
            "n": len(g),
            **metrics,
        })

    return pd.DataFrame(rows)


def compute_topk_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (seed, model), g in long_df.groupby(["seed", "model"]):
        g = g.copy().reset_index(drop=True)

        for k in TOP_KS:
            if len(g) < k:
                continue

            true_top_idx = set(
                g.sort_values("y_true", ascending=False)
                .head(k)
                .index
                .tolist()
            )

            pred_top = (
                g.sort_values("y_pred", ascending=False)
                .head(k)
                .copy()
            )

            pred_top_idx = set(pred_top.index.tolist())

            hit_count = len(true_top_idx & pred_top_idx)
            hit_rate = hit_count / k

            rows.append({
                "seed": seed,
                "model": model,
                "k": k,
                "hit_count": hit_count,
                "hit_rate": hit_rate,
                "pred_topk_true_mean": float(pred_top["y_true"].mean()),
                "pred_topk_true_median": float(pred_top["y_true"].median()),
                "pred_topk_true_min": float(pred_top["y_true"].min()),
                "true_topk_true_mean": float(
                    g.loc[list(true_top_idx), "y_true"].mean()
                ),
                "test_set_true_mean": float(g["y_true"].mean()),
                "enrichment_over_test_mean": float(
                    pred_top["y_true"].mean() - g["y_true"].mean()
                ),
            })

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [
        c for c in df.columns
        if c not in group_cols
        and c not in {"seed"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    agg_dict = {}

    for col in numeric_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")

    summary = (
        df
        .groupby(group_cols, as_index=False)
        .agg(**agg_dict)
    )

    return summary


def build_worst_predictions(long_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    rows = []

    for (seed, model), g in long_df.groupby(["seed", "model"]):
        worst = (
            g.sort_values("abs_error", ascending=False)
            .head(top_n)
            .copy()
        )
        worst["rank_abs_error"] = np.arange(1, len(worst) + 1)
        rows.append(worst)

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    out_dir = PROJECT_ROOT / "artifacts" / "topk_error_analysis"
    ensure_dir(out_dir)

    seed_dfs = []

    for seed in SEEDS:
        print(f"Loading predictions for seed {seed}...")
        seed_dfs.append(load_seed_predictions(seed))

    wide_df = pd.concat(seed_dfs, ignore_index=True)
    wide_df.to_csv(
        out_dir / "final_predictions_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    long_df = build_long_predictions(wide_df)
    long_df.to_csv(
        out_dir / "final_predictions_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 1. 总体指标
    overall_raw = compute_overall_metrics(long_df)
    overall_raw.to_csv(
        out_dir / "overall_metrics_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_summary = summarize_metrics(overall_raw, ["model"])
    overall_summary = overall_summary.sort_values(
        ["rmse_mean", "mae_mean"],
        ascending=[True, True],
    )
    overall_summary.to_csv(
        out_dir / "overall_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2. 按长度分组
    length_raw = compute_group_metrics(long_df, "length")
    length_raw.to_csv(
        out_dir / "length_group_metrics_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    length_summary = summarize_metrics(length_raw, ["model", "group_value"])
    length_summary.to_csv(
        out_dir / "length_group_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 3. 按活性强弱分组
    activity_raw = compute_group_metrics(long_df, "activity_bin")
    activity_raw.to_csv(
        out_dir / "activity_group_metrics_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    activity_summary = summarize_metrics(activity_raw, ["model", "group_value"])
    activity_summary.to_csv(
        out_dir / "activity_group_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 4. Top-K 分析
    topk_raw = compute_topk_metrics(long_df)
    topk_raw.to_csv(
        out_dir / "topk_metrics_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topk_summary = summarize_metrics(topk_raw, ["model", "k"])
    topk_summary.to_csv(
        out_dir / "topk_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 5. 最差预测样本
    worst_df = build_worst_predictions(long_df, top_n=20)
    worst_df.to_csv(
        out_dir / "worst_predictions_top20_each_model_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDone.")
    print(f"Saved to: {out_dir}")

    print("\nOverall summary:")
    print(
        overall_summary[
            [
                "model",
                "rmse_mean",
                "rmse_std",
                "mae_mean",
                "mae_std",
                "spearman_mean",
                "spearman_std",
                "bias_mean_mean",
            ]
        ]
    )

    print("\nTop-K summary:")
    print(
        topk_summary[
            [
                "model",
                "k",
                "hit_rate_mean",
                "hit_rate_std",
                "pred_topk_true_mean_mean",
                "enrichment_over_test_mean_mean",
            ]
        ].sort_values(["k", "hit_rate_mean"], ascending=[True, False])
    )


if __name__ == "__main__":
    main()