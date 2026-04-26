from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


MODEL_INPUT_COLUMNS = [
    "sample_id",
    "sequence",
    "length",
    "task_role",
    "is_main_23",
    "is_aux_45",

    "label_pIC50",
    "label_ic50_uM",

    "sample_weight",
    "weight_len",
    "weight_quality",

    "source_count",
    "is_from_merged_backbone",
    "contains_merged",
    "high_confidence_flag",
    "conflict_flag",
    "tier_23",

    "dedup_strategy",
    "label_source_mode",

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


def build_sample_id(df: pd.DataFrame, prefix: str) -> pd.Series:
    return pd.Series([f"{prefix}_{i:04d}" for i in range(1, len(df) + 1)], index=df.index)


def validate_table(df: pd.DataFrame, table_name: str) -> None:
    required_cols = [
        "sequence",
        "length",
        "label_pIC50",
        "label_ic50_uM",
        "sample_weight",
        "task_role",
        "is_main_23",
        "is_aux_45",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")

    if df["sequence"].isna().any():
        raise ValueError(f"{table_name} has NaN sequence")
    if (df["length"] <= 0).any():
        raise ValueError(f"{table_name} has non-positive length")
    if (df["label_ic50_uM"] <= 0).any():
        raise ValueError(f"{table_name} has non-positive label_ic50_uM")
    if df["label_pIC50"].isna().any():
        raise ValueError(f"{table_name} has NaN label_pIC50")


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in MODEL_INPUT_COLUMNS if c in df.columns]
    others = [c for c in df.columns if c not in existing]
    return df[existing + others].copy()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "dataset_exact_core_v1.yaml"
    cfg = load_config(config_path)

    final_dir = project_root / cfg["outputs"]["final_dir"]
    report_dir = project_root / cfg["outputs"]["report_dir"]
    model_input_dir = final_dir / "model_input"

    ensure_dir(model_input_dir)
    ensure_dir(report_dir)

    master_path = final_dir / "ace_exact_core_2_5_master_v1.csv"
    main_path = final_dir / "ace_exact_main_2_3_v1.csv"
    aux_path = final_dir / "ace_exact_aux_4_5_v1.csv"

    if not master_path.exists():
        raise FileNotFoundError(f"Missing file: {master_path}")
    if not main_path.exists():
        raise FileNotFoundError(f"Missing file: {main_path}")
    if not aux_path.exists():
        raise FileNotFoundError(f"Missing file: {aux_path}")

    master_df = safe_read_csv(master_path)
    main_df = safe_read_csv(main_path)
    aux_df = safe_read_csv(aux_path)

    # 基础清洗
    for df in [master_df, main_df, aux_df]:
        df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
        df["length"] = df["length"].astype(int)

    # 构造 sample_id
    master_df["sample_id"] = build_sample_id(master_df, "ACE")
    main_df["sample_id"] = build_sample_id(main_df, "MAIN")
    aux_df["sample_id"] = build_sample_id(aux_df, "AUX")

    validate_table(master_df, "master_df")
    validate_table(main_df, "main_df")
    validate_table(aux_df, "aux_df")

    # 统一列顺序
    master_df = reorder_columns(master_df)
    main_df = reorder_columns(main_df)
    aux_df = reorder_columns(aux_df)

    # joint 表：用于联合训练
    joint_df = pd.concat([main_df, aux_df], ignore_index=True)
    joint_df["sample_id"] = build_sample_id(joint_df, "JOINT")
    joint_df = reorder_columns(joint_df)

    # 输出
    main_out = model_input_dir / "a1_main_2_3_model_input_v1.csv"
    aux_out = model_input_dir / "a1_aux_4_5_model_input_v1.csv"
    joint_out = model_input_dir / "a1_joint_2_5_model_input_v1.csv"

    save_df(main_df, main_out)
    save_df(aux_df, aux_out)
    save_df(joint_df, joint_out)

    summary = pd.DataFrame([
        {"metric": "main_rows", "value": len(main_df)},
        {"metric": "aux_rows", "value": len(aux_df)},
        {"metric": "joint_rows", "value": len(joint_df)},
        {"metric": "main_unique_seq", "value": int(main_df["sequence"].nunique())},
        {"metric": "aux_unique_seq", "value": int(aux_df["sequence"].nunique())},
        {"metric": "joint_unique_seq", "value": int(joint_df["sequence"].nunique())},
        {"metric": "main_pIC50_min", "value": float(main_df["label_pIC50"].min())},
        {"metric": "main_pIC50_max", "value": float(main_df["label_pIC50"].max())},
        {"metric": "aux_pIC50_min", "value": float(aux_df["label_pIC50"].min())},
        {"metric": "aux_pIC50_max", "value": float(aux_df["label_pIC50"].max())},
        {"metric": "joint_weight_min", "value": float(joint_df["sample_weight"].min())},
        {"metric": "joint_weight_max", "value": float(joint_df["sample_weight"].max())},
        {"metric": "joint_weight_mean", "value": float(joint_df["sample_weight"].mean())},
    ])
    save_df(summary, report_dir / "a1_model_input_summary_v1.csv")

    print("\nDone.")
    print(f"Main model input saved to: {main_out}")
    print(f"Aux model input saved to: {aux_out}")
    print(f"Joint model input saved to: {joint_out}")
    print(f"Summary saved to: {report_dir / 'a1_model_input_summary_v1.csv'}")


if __name__ == "__main__":
    main()