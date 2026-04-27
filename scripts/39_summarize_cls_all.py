from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_summary(path: Path, dataset: str, experiment_group: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] missing: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["dataset"] = dataset
    df["experiment_group"] = experiment_group

    if "feature_set" not in df.columns:
        if experiment_group == "pLM4ACE-like":
            df["feature_set"] = "ESM2"
        elif experiment_group == "GRU4ACE-like":
            df["feature_set"] = "Token/GRU"
        else:
            df["feature_set"] = "Unknown"

    if "model" not in df.columns:
        raise ValueError(f"No model column in {path}")

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "dataset",
        "experiment_group",
        "feature_set",
        "model",
        "n_seeds",
        "n_features_mean",
        "extra_dim_mean",
        "test_acc_mean",
        "test_acc_std",
        "test_bacc_mean",
        "test_bacc_std",
        "test_auc_mean",
        "test_auc_std",
        "test_auprc_mean",
        "test_auprc_std",
        "test_f1_mean",
        "test_f1_std",
        "test_mcc_mean",
        "test_mcc_std",
        "test_precision_mean",
        "test_precision_std",
        "test_sensitivity_mean",
        "test_sensitivity_std",
        "test_specificity_mean",
        "test_specificity_std",
    ]

    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    out = df[keep_cols].copy()

    metric_cols = [c for c in out.columns if c.startswith("test_")]
    for c in metric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def add_method_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def label(row):
        group = str(row["experiment_group"])
        model = str(row["model"])
        feature = str(row["feature_set"])

        if group == "pLM4ACE-like":
            return "pLM4ACE-like"

        if group == "GRU4ACE-like":
            return "GRU4ACE-like"

        if group == "Fusion":
            return "A1/Descriptor + ESM2 fusion"

        if "A1Flat" in feature or "A1Flat" in model:
            return "Ours: A1 pairwise representation"

        if "Descriptor" in feature or "Descriptor" in model or "PhysChem" in feature:
            return "Descriptor baseline"

        if "SeqOnly" in feature or "SeqOnly" in model:
            return "Sequence baseline"

        return "Other"

    out["method_family"] = out.apply(label, axis=1)
    return out


def rank_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out_list = []

    for dataset, g in df.groupby("dataset"):
        g2 = g.sort_values(
            ["test_mcc_mean", "test_auc_mean", "test_bacc_mean"],
            ascending=[False, False, False],
        ).copy()
        g2.insert(0, "rank_by_mcc", range(1, len(g2) + 1))
        out_list.append(g2)

    return pd.concat(out_list, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact_root",
        default="artifacts/classification",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/classification/final_summary",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    artifact_root = project_root / args.artifact_root
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

    datasets = ["acepair_cls_2_3", "acepair_cls_2_5"]

    all_parts = []

    for dataset in datasets:
        paths = [
            (
                "FeatureModelGrid",
                artifact_root / "feature_model_grid" / dataset / "cls_grid_summary.csv",
            ),
            (
                "pLM4ACE-like",
                artifact_root / "plm4ace_like" / dataset / "plm4ace_like_cls_summary.csv",
            ),
            (
                "Fusion",
                artifact_root / "fusion_models" / dataset / "fusion_cls_summary.csv",
            ),
            (
                "GRU4ACE-like",
                artifact_root / "gru4ace_like" / dataset / "gru4ace_like_cls_summary.csv",
            ),
        ]

        for group, path in paths:
            part = read_summary(path, dataset, group)
            if len(part) > 0:
                all_parts.append(part)

    if not all_parts:
        raise FileNotFoundError("No summary files found.")

    all_df = pd.concat(all_parts, ignore_index=True)
    all_df = normalize_columns(all_df)
    all_df = add_method_label(all_df)
    ranked = rank_by_dataset(all_df)

    all_path = out_dir / "all_cls_model_comparison_ranked.csv"
    ranked.to_csv(all_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Saved:", all_path)

    for dataset in datasets:
        sub = ranked[ranked["dataset"] == dataset].copy()

        top_path = out_dir / f"{dataset}_top20_by_mcc.csv"
        sub.head(20).to_csv(top_path, index=False, encoding="utf-8-sig")

        print("=" * 80)
        print(dataset)
        print(sub.head(20)[
            [
                "rank_by_mcc",
                "method_family",
                "experiment_group",
                "feature_set",
                "model",
                "test_mcc_mean",
                "test_auc_mean",
                "test_bacc_mean",
                "test_f1_mean",
            ]
        ].to_string(index=False))
        print("Saved:", top_path)

    # 额外输出每个方法家族的最好模型
    best_family_rows = []

    for (dataset, family), g in ranked.groupby(["dataset", "method_family"]):
        best_family_rows.append(g.sort_values("rank_by_mcc").iloc[0])

    best_family = pd.DataFrame(best_family_rows)
    best_family = best_family.sort_values(
        ["dataset", "test_mcc_mean", "test_auc_mean"],
        ascending=[True, False, False],
    )

    best_family_path = out_dir / "best_model_per_method_family.csv"
    best_family.to_csv(best_family_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Best model per method family:")
    print(best_family[
        [
            "dataset",
            "method_family",
            "experiment_group",
            "feature_set",
            "model",
            "test_mcc_mean",
            "test_auc_mean",
            "test_bacc_mean",
            "test_f1_mean",
        ]
    ].to_string(index=False))
    print("Saved:", best_family_path)


if __name__ == "__main__":
    main()