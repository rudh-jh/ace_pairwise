from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Cannot find required column from candidates: {candidates}")
    return None


def normalize_sequence(seq: object) -> str:
    return str(seq).strip().upper()


def build_one_dataset(
    df: pd.DataFrame,
    length_min: int,
    length_max: int,
    ic50_threshold_uM: float,
) -> pd.DataFrame:
    seq_col = find_col(df, ["sequence", "Sequence", "peptide", "Peptide"])
    length_col = find_col(df, ["length", "Length"], required=False)
    ic50_col = find_col(df, ["label_ic50_uM", "ic50_uM", "IC50_uM", "ic50"], required=False)
    pic50_col = find_col(df, ["label_pIC50", "pIC50", "pic50"], required=False)

    out = df.copy()
    out[seq_col] = out[seq_col].map(normalize_sequence)

    if length_col is None:
        out["length"] = out[seq_col].str.len()
        length_col = "length"
    else:
        out[length_col] = pd.to_numeric(out[length_col], errors="coerce")
        out["length"] = out[length_col].fillna(out[seq_col].str.len()).astype(int)

    out = out[(out["length"] >= length_min) & (out["length"] <= length_max)].copy()

    if ic50_col is not None:
        out[ic50_col] = pd.to_numeric(out[ic50_col], errors="coerce")
        out = out.dropna(subset=[ic50_col]).copy()
        out["cls_label"] = (out[ic50_col] <= ic50_threshold_uM).astype(int)
        out["cls_source"] = f"{ic50_col} <= {ic50_threshold_uM} uM"
    elif pic50_col is not None:
        out[pic50_col] = pd.to_numeric(out[pic50_col], errors="coerce")
        out = out.dropna(subset=[pic50_col]).copy()
        pic50_threshold = -np.log10(ic50_threshold_uM * 1e-6)
        out["cls_label"] = (out[pic50_col] >= pic50_threshold).astype(int)
        out["cls_source"] = f"{pic50_col} >= {pic50_threshold:.6f}"
    else:
        raise ValueError("Need either IC50 column or pIC50 column to build classification label.")

    # 你的分类标签方向：1 = high activity, 0 = low/weak activity
    out["cls_label_name"] = out["cls_label"].map(
        {
            1: "high_activity",
            0: "low_or_weak_activity",
        }
    )

    # pLM4ACE 兼容标签方向：0 = positive/high activity, 1 = negative/low activity
    out["plm4ace_label"] = out["cls_label"].map({1: 0, 0: 1}).astype(int)

    out["cls_threshold_ic50_uM"] = float(ic50_threshold_uM)
    out["cls_length_range"] = f"{length_min}_{length_max}"

    preferred = [
        "sequence",
        "length",
        "task_role",
        "is_main_23",
        "is_aux_45",
        "label_ic50_uM",
        "label_pIC50",
        "cls_label",
        "cls_label_name",
        "plm4ace_label",
        "cls_source",
        "cls_threshold_ic50_uM",
        "cls_length_range",
        "sample_weight",
        "weight_len",
        "weight_quality",
        "source_count",
        "record_count_total",
        "high_confidence_flag",
        "conflict_flag",
        "tier_23",
        "ic50_uM_min",
        "ic50_uM_max",
        "ic50_uM_median",
        "ic50_uM_mean",
        "ic50_spread_ratio_max_min",
        "source_db_set",
        "origin_source_set",
        "citation_ids",
        "notes",
    ]
    cols = [c for c in preferred if c in out.columns]
    others = [c for c in out.columns if c not in cols]
    out = out[cols + others].reset_index(drop=True)

    return out


def write_summary(df: pd.DataFrame, path: Path, dataset_name: str) -> None:
    rows = []

    rows.append({"dataset": dataset_name, "metric": "n_total", "value": len(df)})

    for k, v in df["cls_label"].value_counts().sort_index().items():
        rows.append({"dataset": dataset_name, "metric": f"cls_label_{k}", "value": int(v)})

    for k, v in df["cls_label_name"].value_counts().items():
        rows.append({"dataset": dataset_name, "metric": f"label_{k}", "value": int(v)})

    for k, v in df["length"].value_counts().sort_index().items():
        rows.append({"dataset": dataset_name, "metric": f"length_{k}", "value": int(v)})

    if "conflict_flag" in df.columns:
        for k, v in df["conflict_flag"].astype(str).value_counts().items():
            rows.append({"dataset": dataset_name, "metric": f"conflict_flag_{k}", "value": int(v)})

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/final/ace_exact_core_2_5_master_v1.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="data/final/classification",
    )
    parser.add_argument(
        "--ic50_threshold_uM",
        type=float,
        default=50.0,
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / args.input
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

    df = pd.read_csv(input_path)

    datasets = {
        "acepair_cls_2_3": build_one_dataset(df, 2, 3, args.ic50_threshold_uM),
        "acepair_cls_2_5": build_one_dataset(df, 2, 5, args.ic50_threshold_uM),
    }

    for name, sub in datasets.items():
        out_path = out_dir / f"{name}.csv"
        summary_path = out_dir / f"{name}_summary.csv"

        sub.to_csv(out_path, index=False, encoding="utf-8-sig")
        write_summary(sub, summary_path, name)

        print("=" * 80)
        print(name)
        print("saved:", out_path)
        print("n =", len(sub))
        print("label distribution:")
        print(sub["cls_label_name"].value_counts())
        print("length distribution:")
        print(sub["length"].value_counts().sort_index())

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()