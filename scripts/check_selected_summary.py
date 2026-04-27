from pathlib import Path
import pandas as pd

def find_project_root(start: Path) -> Path:
    """
    从当前文件位置往上找项目根目录。
    判断标准：存在 artifacts 文件夹和 configs 文件夹。
    """
    cur = start.resolve()

    for parent in [cur] + list(cur.parents):
        if (parent / "artifacts").exists() and (parent / "configs").exists():
            return parent

    raise FileNotFoundError("没有找到项目根目录，请检查文件位置。")


project_root = find_project_root(Path(__file__).parent)

p = project_root / "artifacts" / "a1flat_svr_improvement_suite" / "a1flat_svr_improvement_raw.csv"

print(f"项目根目录: {project_root}")
print(f"读取文件: {p}")

if not p.exists():
    raise FileNotFoundError(f"找不到文件: {p}")

df = pd.read_csv(p)

selected = df[df["model"] == "A1FlatSelected-TunedSVR"].copy()

print("\nSelected rows:")
print(
    selected[
        [
            "seed",
            "feature_set",
            "n_features",
            "test_rmse",
            "test_mae",
            "test_spearman",
            "test_top10_hit_rate",
            "test_top20_hit_rate",
            "test_strong_mae",
            "test_strong_bias",
        ]
    ].to_string(index=False)
)

summary = selected.groupby("model", as_index=False).agg(
    n_seeds=("seed", "nunique"),
    test_rmse_mean=("test_rmse", "mean"),
    test_rmse_std=("test_rmse", "std"),
    test_mae_mean=("test_mae", "mean"),
    test_mae_std=("test_mae", "std"),
    test_spearman_mean=("test_spearman", "mean"),
    test_spearman_std=("test_spearman", "std"),
    top10_mean=("test_top10_hit_rate", "mean"),
    top20_mean=("test_top20_hit_rate", "mean"),
    strong_mae_mean=("test_strong_mae", "mean"),
    strong_bias_mean=("test_strong_bias", "mean"),
)

print("\nCombined selected summary:")
print(summary.to_string(index=False))