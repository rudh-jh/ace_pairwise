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


def norm_upper(x: Any) -> str:
    return norm_text(x).upper()


def norm_lower(x: Any) -> str:
    return norm_text(x).lower()


def to_bool(x: Any) -> bool | None:
    if pd.isna(x):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def safe_float(x: Any) -> float | None:
    if pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> int | None:
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def split_multi_text(value: Any) -> list[str]:
    s = norm_text(value)
    if not s:
        return []
    parts = [p.strip() for p in s.split("||")]
    return [p for p in parts if p]


def unique_join(values: list[Any]) -> str | None:
    bag = []
    seen = set()

    for v in values:
        for p in split_multi_text(v):
            if p and p not in seen:
                seen.add(p)
                bag.append(p)

    if not bag:
        return None
    return " || ".join(bag)


def compute_spread_ratio(vmin: float | None, vmax: float | None) -> float | None:
    if vmin is None or vmax is None or vmin <= 0:
        return None
    return float(vmax / vmin)


def prepare_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sequence"] = out["sequence"].astype(str).str.upper().str.strip()
    out["length"] = out["length"].astype(int)
    out["ic50_uM"] = out["ic50_uM"].astype(float)

    # 保证 filtered_source_name 存在
    if "filtered_source_name" not in out.columns:
        out["filtered_source_name"] = out["source_db"].astype(str)

    out["filtered_source_name_norm"] = out["filtered_source_name"].map(norm_lower)

    out["sequence_len_check"] = out["sequence"].str.len()
    out = out[out["length"] == out["sequence_len_check"]].copy()

    # 核心修复：直接用 filtered_source_name 判断 merged 来源
    out["is_merged_source"] = out["filtered_source_name_norm"].eq("merged_short23_tiered_standardized")

    # 后面用于统计“来源重叠”的唯一来源键，也直接用 filtered_source_name
    out["origin_source_key"] = out["filtered_source_name_norm"]

    out["high_confidence_flag"] = out["high_confidence_flag"].map(
        lambda x: bool(to_bool(x)) if to_bool(x) is not None else False
    )
    out["conflict_flag"] = out["conflict_flag"].map(
        lambda x: bool(to_bool(x)) if to_bool(x) is not None else False
    )

    return out


def build_master_row_from_merged(group: pd.DataFrame) -> dict[str, Any]:
    merged_rows = group[group["is_merged_source"]].copy()
    if merged_rows.empty:
        raise ValueError("No merged rows found in group.")

    merged_row = merged_rows.iloc[0]

    sequence = norm_upper(merged_row["sequence"])
    length = safe_int(merged_row["length"])

    ic50_median = safe_float(merged_row["ic50_uM_median"])
    ic50_min = safe_float(merged_row["ic50_uM_min"])
    ic50_max = safe_float(merged_row["ic50_uM_max"])

    source_count = int(group["origin_source_key"].nunique())
    total_record_count = int(group["record_count"].fillna(1).astype(int).sum())

    row = {
        "sequence": sequence,
        "length": length,
        "dedup_strategy": "merged_backbone",
        "task_role": "main" if length in {2, 3} else "aux",

        "label_source_mode": "merged_sequence_level_authoritative",
        "is_from_merged_backbone": True,
        "contains_merged": True,

        "ic50_uM_min": ic50_min,
        "ic50_uM_max": ic50_max,
        "ic50_uM_median": ic50_median,
        "ic50_uM_mean": safe_float(merged_row["ic50_uM"]),
        "ic50_spread_ratio_max_min": compute_spread_ratio(ic50_min, ic50_max),

        "source_count": source_count,
        "record_count_total": total_record_count,
        "row_count_after_filter": len(group),

        "source_db_set": unique_join(group["source_db"].tolist()),
        "source_table_set": unique_join(group["source_table"].tolist()),
        "source_file_set": unique_join(group["source_file"].tolist()),
        "origin_source_set": unique_join(group["filtered_source_name"].tolist()),

        "high_confidence_flag": bool(merged_row["high_confidence_flag"]),
        "conflict_flag": bool(group["conflict_flag"].fillna(False).any()),
        "tier_23": norm_text(merged_row["tier_23"]) or None,

        "raw_record_ids": unique_join(group["raw_record_ids"].tolist()),
        "citation_ids": unique_join(group["citation_ids"].tolist()),
        "notes": unique_join(group["notes"].tolist()),
    }
    return row


def build_master_row_by_aggregation(group: pd.DataFrame) -> dict[str, Any]:
    sequence = norm_upper(group["sequence"].iloc[0])
    length = safe_int(group["length"].iloc[0])

    ic50_values = group["ic50_uM"].dropna().astype(float).tolist()
    ic50_min = float(np.min(ic50_values)) if ic50_values else None
    ic50_max = float(np.max(ic50_values)) if ic50_values else None
    ic50_median = float(np.median(ic50_values)) if ic50_values else None
    ic50_mean = float(np.mean(ic50_values)) if ic50_values else None

    source_count = int(group["origin_source_key"].nunique())
    record_count_total = int(group["record_count"].fillna(1).astype(int).sum())
    task_role = "main" if length in {2, 3} else "aux"

    row = {
        "sequence": sequence,
        "length": length,
        "dedup_strategy": "cross_source_or_source_only_aggregation",
        "task_role": task_role,

        "label_source_mode": "aggregated_from_filtered_exact_candidates",
        "is_from_merged_backbone": False,
        "contains_merged": bool(group["is_merged_source"].any()),

        "ic50_uM_min": ic50_min,
        "ic50_uM_max": ic50_max,
        "ic50_uM_median": ic50_median,
        "ic50_uM_mean": ic50_mean,
        "ic50_spread_ratio_max_min": compute_spread_ratio(ic50_min, ic50_max),

        "source_count": source_count,
        "record_count_total": record_count_total,
        "row_count_after_filter": len(group),

        "source_db_set": unique_join(group["source_db"].tolist()),
        "source_table_set": unique_join(group["source_table"].tolist()),
        "source_file_set": unique_join(group["source_file"].tolist()),
        "origin_source_set": unique_join(group["filtered_source_name"].tolist()),

        "high_confidence_flag": False,
        "conflict_flag": bool(group["conflict_flag"].fillna(False).any()),
        "tier_23": None,

        "raw_record_ids": unique_join(group["raw_record_ids"].tolist()),
        "citation_ids": unique_join(group["citation_ids"].tolist()),
        "notes": unique_join(group["notes"].tolist()),
    }
    return row


def build_trace_rows(group: pd.DataFrame, master_sequence: str, master_length: int, dedup_strategy: str) -> list[dict[str, Any]]:
    rows = []

    authoritative_idx = None
    if dedup_strategy == "merged_backbone":
        merged_rows = group[group["is_merged_source"]]
        if not merged_rows.empty:
            authoritative_idx = merged_rows.index[0]

    for idx, row in group.iterrows():
        rows.append({
            "sequence": master_sequence,
            "length": master_length,
            "dedup_strategy": dedup_strategy,
            "source_db": row["source_db"],
            "source_table": row["source_table"],
            "source_file": row["source_file"],
            "filtered_source_name": row["filtered_source_name"],
            "source_row_id": row["source_row_id"],
            "is_authoritative_row": bool(idx == authoritative_idx) if authoritative_idx is not None else False,
            "is_merged_source": bool(row["is_merged_source"]),
            "ic50_uM": row["ic50_uM"],
            "label_type": row["label_type"],
            "label_is_exact": row["label_is_exact"],
            "raw_record_ids": row["raw_record_ids"],
            "citation_ids": row["citation_ids"],
            "notes": row["notes"],
        })
    return rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "dataset_exact_core_v1.yaml"
    cfg = load_config(config_path)

    filtered_dir = project_root / cfg["outputs"]["interim_dir"] / "filtered_exact_core"
    dedup_dir = project_root / cfg["outputs"]["interim_dir"] / "deduped"
    report_dir = project_root / cfg["outputs"]["report_dir"]

    ensure_dir(dedup_dir)
    ensure_dir(report_dir)

    candidates_path = filtered_dir / "all_exact_core_candidates.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing candidate file: {candidates_path}")

    df = safe_read_csv(candidates_path)
    df = prepare_candidates(df)

    master_rows = []
    trace_rows = []

    grouped = df.groupby(["sequence", "length"], sort=True)

    for (sequence, length), group in grouped:
        group = group.copy()

        use_merged_backbone = (int(length) in {2, 3}) and bool(group["is_merged_source"].any())

        if use_merged_backbone:
            master_row = build_master_row_from_merged(group)
        else:
            master_row = build_master_row_by_aggregation(group)

        master_rows.append(master_row)
        trace_rows.extend(
            build_trace_rows(
                group=group,
                master_sequence=master_row["sequence"],
                master_length=master_row["length"],
                dedup_strategy=master_row["dedup_strategy"],
            )
        )

    master_df = pd.DataFrame(master_rows).sort_values(["length", "sequence"]).reset_index(drop=True)
    trace_df = pd.DataFrame(trace_rows).sort_values(
        ["length", "sequence", "filtered_source_name", "source_row_id"]
    ).reset_index(drop=True)

    main_23_df = master_df[master_df["length"].isin([2, 3])].copy()
    aux_45_df = master_df[master_df["length"].isin([4, 5])].copy()

    save_df(master_df, dedup_dir / "ace_exact_core_2_5_master_prelabel_v1.csv")
    save_df(main_23_df, dedup_dir / "ace_exact_main_2_3_prelabel_v1.csv")
    save_df(aux_45_df, dedup_dir / "ace_exact_aux_4_5_prelabel_v1.csv")
    save_df(trace_df, dedup_dir / "ace_exact_trace_links_prelabel_v1.csv")

    summary = pd.DataFrame([
        {"metric": "master_total_rows", "value": len(master_df)},
        {"metric": "main_2_3_rows", "value": len(main_23_df)},
        {"metric": "aux_4_5_rows", "value": len(aux_45_df)},
        {"metric": "trace_rows", "value": len(trace_df)},
        {"metric": "merged_backbone_rows", "value": int((master_df["dedup_strategy"] == "merged_backbone").sum())},
        {"metric": "non_merged_aggregated_rows", "value": int((master_df["dedup_strategy"] != "merged_backbone").sum())},
        {"metric": "multi_source_rows", "value": int((master_df["source_count"].fillna(0) > 1).sum())},
        {"metric": "single_source_rows", "value": int((master_df["source_count"].fillna(0) == 1).sum())},
    ])
    save_df(summary, report_dir / "exact_core_dedup_summary.csv")

    length_dist = (
        master_df["length"]
        .value_counts()
        .sort_index()
        .rename_axis("length")
        .reset_index(name="count")
    )
    save_df(length_dist, report_dir / "exact_core_length_distribution_after_dedup.csv")

    source_combo_dist = (
        master_df["source_count"]
        .fillna(0)
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("source_count")
        .reset_index(name="count")
    )
    save_df(source_combo_dist, report_dir / "exact_core_source_count_distribution.csv")

    print("\nDone.")
    print(f"Master saved to: {dedup_dir / 'ace_exact_core_2_5_master_prelabel_v1.csv'}")
    print(f"Main 2-3 saved to: {dedup_dir / 'ace_exact_main_2_3_prelabel_v1.csv'}")
    print(f"Aux 4-5 saved to: {dedup_dir / 'ace_exact_aux_4_5_prelabel_v1.csv'}")
    print(f"Trace saved to: {dedup_dir / 'ace_exact_trace_links_prelabel_v1.csv'}")
    print(f"Summary saved to: {report_dir / 'exact_core_dedup_summary.csv'}")


if __name__ == "__main__":
    main()