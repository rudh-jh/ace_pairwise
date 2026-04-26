from __future__ import annotations

from pathlib import Path
from typing import Any

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


def build_weight_len(length: int) -> float:
    if length in {2, 3}:
        return 1.0
    if length == 4:
        return 0.6
    if length == 5:
        return 0.5
    return 0.0


def build_weight_quality(row: pd.Series) -> float:
    length = int(row["length"])
    tier = norm_text(row.get("tier_23", ""))

    if length in {2, 3}:
        if tier == "Tier A":
            return 1.00
        if tier == "Tier B":
            return 0.95
        if tier == "Tier C":
            return 0.85

        if bool(row.get("high_confidence_flag", False)):
            return 0.95

        if bool(row.get("is_from_merged_backbone", False)):
            return 0.80

        return 0.75

    # 4–5 aa 辅助层
    if bool(row.get("conflict_flag", False)):
        return 0.50
    if int(row.get("source_count", 1)) > 1:
        return 0.80
    return 0.70


def build_task_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_main_23"] = out["length"].isin([2, 3])
    out["is_aux_45"] = out["length"].isin([4, 5])
    out["task_role"] = out["is_main_23"].map(lambda x: "main" if x else "aux")
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "dataset_exact_core_v1.yaml"
    cfg = load_config(config_path)

    dedup_dir = project_root / cfg["outputs"]["interim_dir"] / "deduped"
    report_dir = project_root / cfg["outputs"]["report_dir"]

    ensure_dir(report_dir)

    master_path = dedup_dir / "ace_exact_core_2_5_master_prelabel_v1.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master file: {master_path}")

    df = safe_read_csv(master_path)
    df["length"] = df["length"].astype(int)

    df = build_task_flags(df)
    df["weight_len"] = df["length"].map(build_weight_len)
    df["weight_quality"] = df.apply(build_weight_quality, axis=1)
    df["sample_weight"] = df["weight_len"] * df["weight_quality"]

    # 分表导出
    master_out = dedup_dir / "ace_exact_core_2_5_weighted_v1.csv"
    main_out = dedup_dir / "ace_exact_main_2_3_weighted_v1.csv"
    aux_out = dedup_dir / "ace_exact_aux_4_5_weighted_v1.csv"

    save_df(df, master_out)
    save_df(df[df["is_main_23"]].copy(), main_out)
    save_df(df[df["is_aux_45"]].copy(), aux_out)

    summary = pd.DataFrame([
        {"metric": "master_rows", "value": len(df)},
        {"metric": "main_rows", "value": int(df["is_main_23"].sum())},
        {"metric": "aux_rows", "value": int(df["is_aux_45"].sum())},
        {"metric": "sample_weight_min", "value": float(df["sample_weight"].min())},
        {"metric": "sample_weight_max", "value": float(df["sample_weight"].max())},
        {"metric": "sample_weight_mean", "value": float(df["sample_weight"].mean())},
    ])
    save_df(summary, report_dir / "weighted_dataset_summary.csv")

    print("\nDone.")
    print(f"Weighted master saved to: {master_out}")
    print(f"Weighted main saved to: {main_out}")
    print(f"Weighted aux saved to: {aux_out}")
    print(f"Summary saved to: {report_dir / 'weighted_dataset_summary.csv'}")


if __name__ == "__main__":
    main()