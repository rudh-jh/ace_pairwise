from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


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


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_upper(x: Any) -> str:
    return normalize_text(x).upper()


def is_canonical_sequence(seq: Any) -> bool:
    s = normalize_upper(seq)
    if not s:
        return False
    return all(ch in STANDARD_AA for ch in s)


def is_exact_label(row: pd.Series) -> bool:
    """
    exact-only 判定逻辑。
    说明：
    1. 主要依赖 label_is_exact；
    2. 同时排除 threshold / range / inequality 等明显非精确标签；
    3. 对 merged 这种已经是 sequence-level 聚合且 label_is_exact=True 的表，允许保留。
    """
    label_is_exact = row.get("label_is_exact", False)
    if pd.isna(label_is_exact) or bool(label_is_exact) is not True:
        return False

    label_type = normalize_text(row.get("label_type", "")).lower()
    relation = normalize_text(row.get("ic50_relation_raw", "")).strip()

    bad_type_keywords = [
        "threshold",
        "range",
        "interval",
        "ec50",
        "inhibition_rate",
        "percent",
        "rescued",
    ]
    if any(k in label_type for k in bad_type_keywords):
        return False

    # exact continuous IC50 通常应避免明显不等式
    bad_relations = {"<", "<=", ">", ">=", "~"}
    if relation in bad_relations:
        return False

    return True


def classify_exclusion_reason(row: pd.Series, min_len: int, max_len: int) -> str | None:
    seq = normalize_upper(row.get("sequence"))
    if not seq:
        return "missing_sequence"

    seq_valid = row.get("sequence_valid", None)
    if pd.notna(seq_valid) and bool(seq_valid) is False:
        return "sequence_invalid"

    canonical_flag = row.get("canonical_aa_only", None)
    if pd.notna(canonical_flag):
        if bool(canonical_flag) is False:
            return "noncanonical_sequence"
    else:
        if not is_canonical_sequence(seq):
            return "noncanonical_sequence"

    length = row.get("length", None)
    if pd.isna(length):
        return "missing_length"
    try:
        length = int(length)
    except Exception:
        return "bad_length"

    if length < min_len or length > max_len:
        return "out_of_length_range"

    ic50_uM = row.get("ic50_uM", None)
    if pd.isna(ic50_uM):
        return "missing_ic50"
    try:
        ic50_uM = float(ic50_uM)
    except Exception:
        return "bad_ic50"
    if ic50_uM <= 0:
        return "nonpositive_ic50"

    if not is_exact_label(row):
        return "not_exact_continuous"

    return None


def filter_exact_core(df: pd.DataFrame, min_len: int, max_len: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()

    exclusion_reasons = []
    for _, row in work.iterrows():
        exclusion_reasons.append(classify_exclusion_reason(row, min_len=min_len, max_len=max_len))
    work["exclusion_reason"] = exclusion_reasons

    kept = work[work["exclusion_reason"].isna()].copy()
    dropped = work[work["exclusion_reason"].notna()].copy()

    # 再做一次稳妥清洗
    kept["sequence"] = kept["sequence"].astype(str).str.upper().str.strip()
    kept["length"] = kept["length"].astype(int)
    kept["ic50_uM"] = kept["ic50_uM"].astype(float)

    return kept, dropped


def build_filter_summary(
    source_name: str,
    input_df: pd.DataFrame,
    kept_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
) -> dict:
    return {
        "source_name": source_name,
        "input_rows": len(input_df),
        "kept_rows": len(kept_df),
        "dropped_rows": len(dropped_df),
        "kept_unique_sequence": int(kept_df["sequence"].nunique()) if not kept_df.empty else 0,
        "kept_min_len": int(kept_df["length"].min()) if not kept_df.empty else None,
        "kept_max_len": int(kept_df["length"].max()) if not kept_df.empty else None,
    }


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "dataset_exact_core_v1.yaml"
    cfg = load_config(config_path)

    min_len = int(cfg["filters"]["min_len"])
    max_len = int(cfg["filters"]["max_len"])

    standardized_dir = project_root / cfg["outputs"]["interim_dir"] / "standardized"
    filtered_dir = project_root / cfg["outputs"]["interim_dir"] / "filtered_exact_core"
    report_dir = project_root / cfg["outputs"]["report_dir"]

    ensure_dir(filtered_dir)
    ensure_dir(report_dir)

    source_files = {
        "merged_short23_tiered_standardized": standardized_dir / "merged_short23_tiered_standardized.csv",
        "biopep_core_standardized": standardized_dir / "biopep_core_standardized.csv",
        "mbpdb_core_standardized": standardized_dir / "mbpdb_core_standardized.csv",
        "ferm_strict_core_standardized": standardized_dir / "ferm_strict_core_standardized.csv",
        "ahtpdb_clean_um_standardized": standardized_dir / "ahtpdb_clean_um_standardized.csv",
    }

    kept_frames = []
    summary_rows = []
    drop_reason_rows = []

    for source_name, path in source_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing standardized file: {path}")

        df = safe_read_csv(path)
        kept_df, dropped_df = filter_exact_core(df, min_len=min_len, max_len=max_len)

        kept_df["filtered_source_name"] = source_name
        dropped_df["filtered_source_name"] = source_name

        kept_out = filtered_dir / f"{source_name}_exact_core_kept.csv"
        dropped_out = filtered_dir / f"{source_name}_exact_core_dropped.csv"

        save_df(kept_df, kept_out)
        save_df(dropped_df, dropped_out)

        kept_frames.append(kept_df)
        summary_rows.append(build_filter_summary(source_name, df, kept_df, dropped_df))

        if not dropped_df.empty:
            reason_count = (
                dropped_df["exclusion_reason"]
                .value_counts(dropna=False)
                .rename_axis("exclusion_reason")
                .reset_index(name="count")
            )
            reason_count["source_name"] = source_name
            drop_reason_rows.append(reason_count)

        print(f"[OK] {source_name}")
        print(f"     input={len(df)} kept={len(kept_df)} dropped={len(dropped_df)}")

    combined_kept = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
    combined_out = filtered_dir / "all_exact_core_candidates.csv"
    save_df(combined_kept, combined_out)

    summary_df = pd.DataFrame(summary_rows)
    save_df(summary_df, report_dir / "exact_core_filter_summary.csv")

    if drop_reason_rows:
        drop_reason_df = pd.concat(drop_reason_rows, ignore_index=True)
    else:
        drop_reason_df = pd.DataFrame(columns=["exclusion_reason", "count", "source_name"])
    save_df(drop_reason_df, report_dir / "exact_core_drop_reasons.csv")

    print("\nDone.")
    print(f"Combined kept candidates saved to: {combined_out}")
    print(f"Filter summary saved to: {report_dir / 'exact_core_filter_summary.csv'}")
    print(f"Drop reasons saved to: {report_dir / 'exact_core_drop_reasons.csv'}")


if __name__ == "__main__":
    main()