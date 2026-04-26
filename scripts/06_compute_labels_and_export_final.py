from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def compute_pIC50_from_uM(ic50_uM: float) -> float:
    """
    pIC50 = 6 - log10(IC50 in uM)
    因为:
        1 uM = 1e-6 M
        pIC50 = -log10(IC50_M)
              = -log10(IC50_uM * 1e-6)
              = 6 - log10(IC50_uM)
    """
    if pd.isna(ic50_uM):
        raise ValueError("ic50_uM is NaN")
    ic50_uM = float(ic50_uM)
    if ic50_uM <= 0:
        raise ValueError(f"ic50_uM must be > 0, got {ic50_uM}")
    return float(6.0 - np.log10(ic50_uM))


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "dataset_exact_core_v1.yaml"
    cfg = load_config(config_path)

    dedup_dir = project_root / cfg["outputs"]["interim_dir"] / "deduped"
    final_dir = project_root / cfg["outputs"]["final_dir"]
    report_dir = project_root / cfg["outputs"]["report_dir"]

    ensure_dir(final_dir)
    ensure_dir(report_dir)

    weighted_master_path = dedup_dir / "ace_exact_core_2_5_weighted_v1.csv"
    trace_path = dedup_dir / "ace_exact_trace_links_prelabel_v1.csv"

    if not weighted_master_path.exists():
        raise FileNotFoundError(f"Missing weighted master file: {weighted_master_path}")
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing trace file: {trace_path}")

    df = safe_read_csv(weighted_master_path)
    trace_df = safe_read_csv(trace_path)

    required_cols = [
        "sequence",
        "length",
        "ic50_uM_median",
        "task_role",
        "is_main_23",
        "is_aux_45",
        "weight_len",
        "weight_quality",
        "sample_weight",
    ]
    validate_required_columns(df, required_cols)

    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    df["length"] = df["length"].astype(int)
    df["ic50_uM_median"] = df["ic50_uM_median"].astype(float)

    bad_ic50 = df["ic50_uM_median"].isna() | (df["ic50_uM_median"] <= 0)
    if bad_ic50.any():
        bad_rows = int(bad_ic50.sum())
        raise ValueError(f"Found {bad_rows} rows with invalid ic50_uM_median <= 0 or NaN")

    # 正式标签
    df["label_ic50_uM"] = df["ic50_uM_median"]
    df["label_pIC50"] = df["label_ic50_uM"].map(compute_pIC50_from_uM)

    # 统一列顺序：把建模最常用的列放前面
    preferred_front_cols = [
        "sequence",
        "length",
        "task_role",
        "is_main_23",
        "is_aux_45",
        "label_ic50_uM",
        "label_pIC50",
        "sample_weight",
        "weight_len",
        "weight_quality",
        "dedup_strategy",
        "label_source_mode",
        "is_from_merged_backbone",
        "contains_merged",
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
        "source_table_set",
        "source_file_set",
        "origin_source_set",
        "raw_record_ids",
        "citation_ids",
        "notes",
    ]
    existing_front_cols = [c for c in preferred_front_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_front_cols]
    df = df[existing_front_cols + remaining_cols].copy()

    # 分层导出
    master_df = df.copy()
    main_df = df[df["is_main_23"].astype(bool)].copy()
    aux_df = df[df["is_aux_45"].astype(bool)].copy()

    master_out = final_dir / "ace_exact_core_2_5_master_v1.csv"
    main_out = final_dir / "ace_exact_main_2_3_v1.csv"
    aux_out = final_dir / "ace_exact_aux_4_5_v1.csv"
    trace_out = final_dir / "ace_exact_trace_links_v1.csv"

    save_df(master_df, master_out)
    save_df(main_df, main_out)
    save_df(aux_df, aux_out)
    save_df(trace_df, trace_out)

    # 汇总报告
    summary_rows = [
        {"metric": "master_rows", "value": len(master_df)},
        {"metric": "main_2_3_rows", "value": len(main_df)},
        {"metric": "aux_4_5_rows", "value": len(aux_df)},
        {"metric": "trace_rows", "value": len(trace_df)},
        {"metric": "unique_sequence_master", "value": int(master_df["sequence"].nunique())},
        {"metric": "unique_sequence_main", "value": int(main_df["sequence"].nunique())},
        {"metric": "unique_sequence_aux", "value": int(aux_df["sequence"].nunique())},
        {"metric": "pIC50_min_master", "value": float(master_df["label_pIC50"].min())},
        {"metric": "pIC50_max_master", "value": float(master_df["label_pIC50"].max())},
        {"metric": "pIC50_mean_master", "value": float(master_df["label_pIC50"].mean())},
        {"metric": "weight_min_master", "value": float(master_df["sample_weight"].min())},
        {"metric": "weight_max_master", "value": float(master_df["sample_weight"].max())},
        {"metric": "weight_mean_master", "value": float(master_df["sample_weight"].mean())},
        {"metric": "merged_backbone_rows", "value": int(master_df["is_from_merged_backbone"].fillna(False).sum())},
        {"metric": "conflict_flag_rows", "value": int(master_df["conflict_flag"].fillna(False).sum())},
        {"metric": "multi_source_rows", "value": int((master_df["source_count"].fillna(0).astype(int) > 1).sum())},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_out = report_dir / "final_dataset_summary_v1.csv"
    save_df(summary_df, summary_out)

    # 长度分布
    length_dist = (
        master_df["length"]
        .value_counts()
        .sort_index()
        .rename_axis("length")
        .reset_index(name="count")
    )
    save_df(length_dist, report_dir / "final_dataset_length_distribution_v1.csv")

    print("\nDone.")
    print(f"Final master saved to: {master_out}")
    print(f"Final main 2-3 saved to: {main_out}")
    print(f"Final aux 4-5 saved to: {aux_out}")
    print(f"Final trace saved to: {trace_out}")
    print(f"Final summary saved to: {summary_out}")


if __name__ == "__main__":
    main()