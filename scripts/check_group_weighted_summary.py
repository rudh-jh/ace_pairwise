from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "artifacts" / "a1flat_group_weighted_svr" / "a1flat_group_weighted_raw.csv"
OUT_DIR = PROJECT_ROOT / "artifacts" / "a1flat_group_weighted_svr"

df = pd.read_csv(RAW_PATH)

metric_cols = [
    "test_rmse",
    "test_mae",
    "test_spearman",
    "test_top5_hit_rate",
    "test_top10_hit_rate",
    "test_top20_hit_rate",
    "test_strong_mae",
    "test_strong_rmse",
    "test_strong_bias",
    "val_rmse",
]

# 1. 原始粒度：model + step + feature_set
summary_full = (
    df.groupby(["model", "step", "feature_set"], as_index=False)
    .agg(
        n_runs=("seed", "count"),
        seeds=("seed", lambda x: ",".join(map(str, sorted(x.unique())))),
        n_features=("n_features", "first"),
        test_rmse_mean=("test_rmse", "mean"),
        test_rmse_std=("test_rmse", "std"),
        test_mae_mean=("test_mae", "mean"),
        test_spearman_mean=("test_spearman", "mean"),
        test_top5_hit_rate_mean=("test_top5_hit_rate", "mean"),
        test_top10_hit_rate_mean=("test_top10_hit_rate", "mean"),
        test_top20_hit_rate_mean=("test_top20_hit_rate", "mean"),
        test_strong_mae_mean=("test_strong_mae", "mean"),
        test_strong_bias_mean=("test_strong_bias", "mean"),
        val_rmse_mean=("val_rmse", "mean"),
    )
    .sort_values(["test_rmse_mean", "test_mae_mean"], ascending=[True, True])
)

summary_full.to_csv(
    OUT_DIR / "a1flat_group_weighted_summary_with_nruns.csv",
    index=False,
    encoding="utf-8-sig",
)

# 2. 只按 model 聚合：解决 A1FlatSelected-TunedSVR 被 k200/k400 拆开的问题
summary_by_model = (
    df.groupby(["model"], as_index=False)
    .agg(
        n_runs=("seed", "count"),
        seeds=("seed", lambda x: ",".join(map(str, sorted(x.unique())))),
        n_features_min=("n_features", "min"),
        n_features_max=("n_features", "max"),
        test_rmse_mean=("test_rmse", "mean"),
        test_rmse_std=("test_rmse", "std"),
        test_mae_mean=("test_mae", "mean"),
        test_mae_std=("test_mae", "std"),
        test_spearman_mean=("test_spearman", "mean"),
        test_spearman_std=("test_spearman", "std"),
        test_top5_hit_rate_mean=("test_top5_hit_rate", "mean"),
        test_top10_hit_rate_mean=("test_top10_hit_rate", "mean"),
        test_top20_hit_rate_mean=("test_top20_hit_rate", "mean"),
        test_strong_mae_mean=("test_strong_mae", "mean"),
        test_strong_bias_mean=("test_strong_bias", "mean"),
        val_rmse_mean=("val_rmse", "mean"),
    )
    .sort_values(["test_rmse_mean", "test_mae_mean"], ascending=[True, True])
)

summary_by_model.to_csv(
    OUT_DIR / "a1flat_group_weighted_summary_by_model.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nTop 30 by model-level test RMSE:")
cols = [
    "model",
    "n_runs",
    "seeds",
    "n_features_min",
    "n_features_max",
    "test_rmse_mean",
    "test_mae_mean",
    "test_spearman_mean",
    "test_top5_hit_rate_mean",
    "test_top10_hit_rate_mean",
    "test_strong_mae_mean",
    "test_strong_bias_mean",
]
print(summary_by_model[cols].head(30).to_string(index=False))

print("\nSaved:")
print(OUT_DIR / "a1flat_group_weighted_summary_with_nruns.csv")
print(OUT_DIR / "a1flat_group_weighted_summary_by_model.csv")