from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAIR_RAW_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "multiseed_summary"
    / "fair_feature_multiseed_raw.csv"
)

LITE_RAW_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "a1_lite_channel_sets_summary"
    / "a1_lite_channel_sets_raw.csv"
)

OUT_DIR = PROJECT_ROOT / "artifacts" / "final_model_comparison"


SELECTED_FAIR_MODELS = [
    "MeanPredictor",
    "SeqOnly-SVR",
    "SeqOnly-CatBoost",
    "PhysChemOnly-SVR",
    "PhysChemOnly-CatBoost",
    "Descriptor-SVR",
    "Descriptor-CatBoost",
    "A1FlatFull-SVR",
    "A1FlatFull-CatBoost",
]

SELECTED_LITE_MODES = [
    "diag_lite",
    "diag_core_pair_lite",
    "diag_pair_no_heur",
    "full",
]


LITE_NAME_MAP = {
    "diag_lite": "A1-DiagLite",
    "diag_core_pair_lite": "A1-DiagCorePairLite",
    "diag_pair_no_heur": "A1-DiagPairNoHeur",
    "full": "A1-Full",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_fair_results() -> pd.DataFrame:
    if not FAIR_RAW_PATH.exists():
        raise FileNotFoundError(f"Missing fair baseline raw file: {FAIR_RAW_PATH}")

    df = pd.read_csv(FAIR_RAW_PATH)

    df = df[df["split"] == "test"].copy()
    df = df[df["model"].isin(SELECTED_FAIR_MODELS)].copy()

    out = pd.DataFrame({
        "seed": df["seed"],
        "model": df["model"],
        "model_group": "traditional_or_flat_baseline",
        "feature_set": df["feature_set"],
        "n_features": df["n_features"],
        "test_rmse": df["rmse"],
        "test_mae": df["mae"],
        "test_spearman": df["spearman"],
    })

    return out


def load_lite_results() -> pd.DataFrame:
    if not LITE_RAW_PATH.exists():
        raise FileNotFoundError(f"Missing A1-Lite raw file: {LITE_RAW_PATH}")

    df = pd.read_csv(LITE_RAW_PATH)

    df = df[df["lite_mode"].isin(SELECTED_LITE_MODES)].copy()

    out = pd.DataFrame({
        "seed": df["seed"],
        "model": df["lite_mode"].map(LITE_NAME_MAP),
        "model_group": "neural_a1_lite",
        "feature_set": df["lite_mode"],
        "n_features": df["n_channels"],
        "test_rmse": df["test_rmse"],
        "test_mae": df["test_mae"],
        "test_spearman": df["test_spearman"],
    })

    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df
        .groupby(["model", "model_group", "feature_set"], as_index=False)
        .agg(
            n_features=("n_features", "first"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_mae_mean=("test_mae", "mean"),
            test_mae_std=("test_mae", "std"),
            test_spearman_mean=("test_spearman", "mean"),
            test_spearman_std=("test_spearman", "std"),
        )
        .sort_values(["test_rmse_mean", "test_mae_mean"], ascending=[True, True])
    )

    return summary


def build_delta_vs_diaglite(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个模型相对 A1-DiagLite 的差值。

    delta_rmse = other_model_rmse - A1_DiagLite_rmse
    delta_mae = other_model_mae - A1_DiagLite_mae
    delta_spearman = other_model_spearman - A1_DiagLite_spearman

    解释：
    - delta_rmse > 0：A1-DiagLite 的 RMSE 更低，也就是 A1-DiagLite 更好。
    - delta_mae > 0：A1-DiagLite 的 MAE 更低，也就是 A1-DiagLite 更好。
    - delta_spearman < 0：A1-DiagLite 的 Spearman 更高，也就是 A1-DiagLite 更好。
    """
    diag = raw_df[raw_df["model"] == "A1-DiagLite"].copy()

    if diag.empty:
        raise ValueError("No A1-DiagLite rows found.")

    diag = diag[
        [
            "seed",
            "test_rmse",
            "test_mae",
            "test_spearman",
        ]
    ].rename(
        columns={
            "test_rmse": "diaglite_rmse",
            "test_mae": "diaglite_mae",
            "test_spearman": "diaglite_spearman",
        }
    )

    merged = raw_df.merge(diag, on="seed", how="left")

    merged["delta_rmse_vs_diaglite"] = (
        merged["test_rmse"] - merged["diaglite_rmse"]
    )
    merged["delta_mae_vs_diaglite"] = (
        merged["test_mae"] - merged["diaglite_mae"]
    )
    merged["delta_spearman_vs_diaglite"] = (
        merged["test_spearman"] - merged["diaglite_spearman"]
    )

    delta_summary = (
        merged[merged["model"] != "A1-DiagLite"]
        .groupby(["model", "model_group", "feature_set"], as_index=False)
        .agg(
            delta_rmse_mean=("delta_rmse_vs_diaglite", "mean"),
            delta_rmse_std=("delta_rmse_vs_diaglite", "std"),
            delta_mae_mean=("delta_mae_vs_diaglite", "mean"),
            delta_mae_std=("delta_mae_vs_diaglite", "std"),
            delta_spearman_mean=("delta_spearman_vs_diaglite", "mean"),
            delta_spearman_std=("delta_spearman_vs_diaglite", "std"),
        )
        .sort_values("delta_rmse_mean", ascending=False)
    )

    return merged, delta_summary


def main() -> None:
    ensure_dir(OUT_DIR)

    fair_df = load_fair_results()
    lite_df = load_lite_results()

    raw_df = pd.concat([fair_df, lite_df], ignore_index=True)

    raw_df = raw_df.sort_values(
        ["seed", "test_rmse", "test_mae"],
        ascending=[True, True, True],
    )

    raw_df.to_csv(
        OUT_DIR / "final_model_comparison_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = summarize(raw_df)

    summary_df.to_csv(
        OUT_DIR / "final_model_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_raw, delta_summary = build_delta_vs_diaglite(raw_df)

    delta_raw.to_csv(
        OUT_DIR / "final_model_delta_vs_diaglite_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_summary.to_csv(
        OUT_DIR / "final_model_delta_vs_diaglite_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Done.")
    print(f"Saved to: {OUT_DIR}")

    print("\nFinal model comparison summary:")
    print(
        summary_df[
            [
                "model",
                "model_group",
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

    print("\nDelta vs A1-DiagLite:")
    print(
        delta_summary[
            [
                "model",
                "delta_rmse_mean",
                "delta_mae_mean",
                "delta_spearman_mean",
            ]
        ]
    )


if __name__ == "__main__":
    main()