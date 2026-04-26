from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


COMMON_COLUMNS = [
    "source_db",
    "source_table",
    "source_file",
    "source_row_id",

    "sequence",
    "length",
    "sequence_valid",
    "canonical_aa_only",

    "ic50_uM",
    "ic50_uM_min",
    "ic50_uM_max",
    "ic50_uM_median",

    "ic50_relation_raw",
    "ic50_status_raw",
    "label_type",
    "label_is_exact",

    "is_sequence_level",
    "source_count",
    "record_count",

    "high_confidence_flag",
    "conflict_flag",
    "tier_23",

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


def resolve_input_paths(cfg: dict) -> dict[str, Path]:
    pepdb_root = Path(cfg["paths"]["pepdb_root"])
    return {name: pepdb_root / rel for name, rel in cfg["inputs"].items()}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_sequence_value(x: Any) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s == "" or s.lower() == "nan":
        return None
    return s


def is_canonical_sequence(seq: str | None) -> bool:
    if seq is None:
        return False
    return all(ch in STANDARD_AA for ch in seq)


def to_bool(x: Any) -> bool | None:
    if pd.isna(x):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


def to_int(x: Any) -> int | None:
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def to_float(x: Any) -> float | None:
    if pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def combine_text_fields(*values: Any) -> str | None:
    parts = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            parts.append(s)
    if not parts:
        return None
    return " || ".join(parts)


def finalize_common_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "sequence" in out.columns:
        out["sequence"] = out["sequence"].map(normalize_sequence_value)

    if "length" in out.columns:
        out["length"] = out["length"].map(to_int)

    if "sequence_valid" in out.columns:
        out["sequence_valid"] = out["sequence_valid"].map(to_bool)

    if "canonical_aa_only" in out.columns:
        out["canonical_aa_only"] = out["canonical_aa_only"].map(to_bool)

    for col in ["ic50_uM", "ic50_uM_min", "ic50_uM_max", "ic50_uM_median"]:
        if col in out.columns:
            out[col] = out[col].map(to_float)

    for col in ["label_is_exact", "is_sequence_level", "high_confidence_flag", "conflict_flag"]:
        if col in out.columns:
            out[col] = out[col].map(to_bool)

    for col in ["source_count", "record_count"]:
        if col in out.columns:
            out[col] = out[col].map(to_int)

    for col in COMMON_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out = out[COMMON_COLUMNS].copy()
    return out


def standardize_merged_short_23(merged_df: pd.DataFrame, tiers_df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    tiers_keep = [
        "sequence",
        "peptide_length",
        "consensus_tier",
        "high_confidence_flag",
        "conflict_flag",
        "consensus_note",
    ]
    merged = merged_df.merge(
        tiers_df[tiers_keep],
        on=["sequence", "peptide_length"],
        how="left",
        validate="1:1",
    ).copy()

    out = pd.DataFrame()
    out["source_db"] = "merged"
    out["source_table"] = "ace_short_2_3_merged_tiered"
    out["source_file"] = source_file
    out["source_row_id"] = merged.index.astype(str)

    out["sequence"] = merged["sequence"]
    out["length"] = merged["peptide_length"]
    out["sequence_valid"] = True
    out["canonical_aa_only"] = merged["sequence"].map(lambda x: is_canonical_sequence(normalize_sequence_value(x)))

    out["ic50_uM"] = merged["ic50_uM_median"]
    out["ic50_uM_min"] = merged["ic50_uM_min"]
    out["ic50_uM_max"] = merged["ic50_uM_max"]
    out["ic50_uM_median"] = merged["ic50_uM_median"]

    out["ic50_relation_raw"] = merged["ic50_relation_set"]
    out["ic50_status_raw"] = merged["ic50_status_set"]
    out["label_type"] = "aggregated_sequence_level"
    out["label_is_exact"] = True

    out["is_sequence_level"] = True
    out["source_count"] = merged["source_count"]
    out["record_count"] = merged["record_count_total"]

    out["high_confidence_flag"] = merged["high_confidence_flag"]
    out["conflict_flag"] = merged["conflict_flag"]
    out["tier_23"] = merged["consensus_tier"]

    out["raw_record_ids"] = combine_series(
        merged["merged_record_id_list"],
        merged["source_record_id_list"],
    )
    out["citation_ids"] = merged["doi_set"]
    out["notes"] = combine_series(
        merged["stability_note"],
        merged["consensus_note"],
        merged["source_name_set"],
        merged["database_reference_raw_set"],
    )

    return finalize_common_schema(out)


def standardize_biopep_core(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["source_db"] = "biopep_uwm"
    out["source_table"] = "biopep_uwm_ic50_core_sequence_level"
    out["source_file"] = source_file
    out["source_row_id"] = df.index.astype(str)

    out["sequence"] = df["sequence"]
    out["length"] = df["peptide_length"]
    out["sequence_valid"] = True
    out["canonical_aa_only"] = df["sequence"].map(lambda x: is_canonical_sequence(normalize_sequence_value(x)))

    out["ic50_uM"] = df["ic50_uM_median"]
    out["ic50_uM_min"] = df["ic50_uM_min"]
    out["ic50_uM_max"] = df["ic50_uM_max"]
    out["ic50_uM_median"] = df["ic50_uM_median"]

    out["ic50_relation_raw"] = df["ic50_relation_set"]
    out["ic50_status_raw"] = df["ic50_parse_status_set"]
    out["label_type"] = "sequence_level_core"
    out["label_is_exact"] = True

    out["is_sequence_level"] = True
    out["source_count"] = 1
    out["record_count"] = df["record_count"]

    out["high_confidence_flag"] = None
    out["conflict_flag"] = None
    out["tier_23"] = None

    out["raw_record_ids"] = combine_series(df["record_id_list"], df["source_record_id_list"])
    out["citation_ids"] = df["detail_report_url_list"]
    out["notes"] = combine_series(df["notes_set"], df["source_name_set"])

    return finalize_common_schema(out)


def standardize_mbpdb_core(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["source_db"] = "mbpdb"
    out["source_table"] = "mbpdb_ace_core_sequence_level"
    out["source_file"] = source_file
    out["source_row_id"] = df.index.astype(str)

    out["sequence"] = df["sequence"]
    out["length"] = df["peptide_length"]
    out["sequence_valid"] = True
    out["canonical_aa_only"] = df["sequence"].map(lambda x: is_canonical_sequence(normalize_sequence_value(x)))

    out["ic50_uM"] = df["ic50_uM_median"]
    out["ic50_uM_min"] = df["ic50_uM_min"]
    out["ic50_uM_max"] = df["ic50_uM_max"]
    out["ic50_uM_median"] = df["ic50_uM_median"]

    out["ic50_relation_raw"] = df["ic50_relation_set"]
    out["ic50_status_raw"] = df["ic50_parse_status_set"]
    out["label_type"] = "sequence_level_core"
    out["label_is_exact"] = True

    out["is_sequence_level"] = True
    out["source_count"] = 1
    out["record_count"] = df["record_count"]

    out["high_confidence_flag"] = None
    out["conflict_flag"] = None
    out["tier_23"] = None

    out["raw_record_ids"] = combine_series(df["record_id_list"], df["source_record_id_list"])
    out["citation_ids"] = df["doi_set"]
    out["notes"] = combine_series(df["title_set"], df["species_set"], df["protein_description_set"])

    return finalize_common_schema(out)


def standardize_ferm_strict_core(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["source_db"] = "fermfoodb"
    out["source_table"] = "fermfoodb_ace_core_strict_sequence_level"
    out["source_file"] = source_file
    out["source_row_id"] = df.index.astype(str)

    out["sequence"] = df["sequence"]
    out["length"] = df["peptide_length"]
    out["sequence_valid"] = True
    out["canonical_aa_only"] = df["sequence"].map(lambda x: is_canonical_sequence(normalize_sequence_value(x)))

    out["ic50_uM"] = df["ic50_uM_median"]
    out["ic50_uM_min"] = df["ic50_uM_min"]
    out["ic50_uM_max"] = df["ic50_uM_max"]
    out["ic50_uM_median"] = df["ic50_uM_median"]

    out["ic50_relation_raw"] = df["ic50_relation_set"]
    out["ic50_status_raw"] = df["ic50_status_set"]
    out["label_type"] = "sequence_level_core_strict"
    out["label_is_exact"] = True

    out["is_sequence_level"] = True
    out["source_count"] = 1
    out["record_count"] = df["record_count"]

    out["high_confidence_flag"] = None
    out["conflict_flag"] = df["stability_flag"].astype(str).str.contains("conflict", case=False, na=False)
    out["tier_23"] = None

    out["raw_record_ids"] = combine_series(df["record_id_list"], df["source_record_id_list"])
    out["citation_ids"] = df["pubmed_id_set"]
    out["notes"] = combine_series(
        df["stability_flag"],
        df["stability_note"],
        df["activity_label_set"],
        df["title_set"],
        df["assay_set"],
    )

    return finalize_common_schema(out)


def standardize_ahtpdb_clean_um(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["source_db"] = "ahtpdb"
    out["source_table"] = "ahtpdb_master_clean_um_expanded"
    out["source_file"] = source_file
    out["source_row_id"] = df["id"].astype(str)

    out["sequence"] = df["sequence_clean"]
    out["length"] = df["len_clean"]
    out["sequence_valid"] = df["sequence_valid"]
    out["canonical_aa_only"] = ~df["sequence_has_noncanonical_char"].map(lambda x: bool(to_bool(x)) if to_bool(x) is not None else False)

    out["ic50_uM"] = df["ic50_uM"]
    out["ic50_uM_min"] = df["ic50_uM"]
    out["ic50_uM_max"] = df["ic50_uM"]
    out["ic50_uM_median"] = df["ic50_uM"]

    out["ic50_relation_raw"] = df["ic50_relation"]
    out["ic50_status_raw"] = df["ic50_type"]
    out["label_type"] = df["ic50_type"]
    out["label_is_exact"] = df["ic50_exact_flag"]

    out["is_sequence_level"] = False
    out["source_count"] = 1
    out["record_count"] = 1

    out["high_confidence_flag"] = None
    out["conflict_flag"] = (
        df["exact_ic50_conflict_flag"].map(lambda x: bool(to_bool(x)) if to_bool(x) is not None else False)
        | df["ic50_merge_conflict_flag"].map(lambda x: bool(to_bool(x)) if to_bool(x) is not None else False)
    )
    out["tier_23"] = None

    out["raw_record_ids"] = df["id"].astype(str)
    out["citation_ids"] = df["source_raw"]
    out["notes"] = combine_series(
        df["table_from"],
        df["ic50_parse_note"],
        df["unit_conversion_note"],
        df["assay_raw"],
        df["method_raw"],
    )

    return finalize_common_schema(out)


def combine_series(*series_list: pd.Series) -> pd.Series:
    if not series_list:
        raise ValueError("At least one series is required.")

    length = len(series_list[0])
    for s in series_list:
        if len(s) != length:
            raise ValueError("All series passed to combine_series must have the same length.")

    rows = []
    for idx in range(length):
        values = [s.iloc[idx] for s in series_list]
        rows.append(combine_text_fields(*values))
    return pd.Series(rows)


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "dataset_exact_core_v1.yaml"
    cfg = load_config(config_path)

    input_paths = resolve_input_paths(cfg)
    interim_dir = project_root / cfg["outputs"]["interim_dir"] / "standardized"
    report_dir = project_root / cfg["outputs"]["report_dir"]

    ensure_dir(interim_dir)
    ensure_dir(report_dir)

    merged_short_23 = safe_read_csv(input_paths["merged_short_23"])
    merged_tiers = safe_read_csv(input_paths["merged_tiers"])
    biopep_core = safe_read_csv(input_paths["biopep_core"])
    mbpdb_core = safe_read_csv(input_paths["mbpdb_core"])
    ferm_strict_core = safe_read_csv(input_paths["ferm_strict_core"])
    ahtpdb_clean_um = safe_read_csv(input_paths["ahtpdb_clean_um"])

    standardized = {
        "merged_short23_tiered_standardized": standardize_merged_short_23(
            merged_short_23, merged_tiers, input_paths["merged_short_23"].name
        ),
        "biopep_core_standardized": standardize_biopep_core(
            biopep_core, input_paths["biopep_core"].name
        ),
        "mbpdb_core_standardized": standardize_mbpdb_core(
            mbpdb_core, input_paths["mbpdb_core"].name
        ),
        "ferm_strict_core_standardized": standardize_ferm_strict_core(
            ferm_strict_core, input_paths["ferm_strict_core"].name
        ),
        "ahtpdb_clean_um_standardized": standardize_ahtpdb_clean_um(
            ahtpdb_clean_um, input_paths["ahtpdb_clean_um"].name
        ),
    }

    summary_rows = []
    for name, df in standardized.items():
        out_path = interim_dir / f"{name}.csv"
        save_df(df, out_path)

        summary_rows.append({
            "dataset_name": name,
            "output_path": str(out_path),
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "non_null_sequence": int(df["sequence"].notna().sum()),
            "canonical_true": int(df["canonical_aa_only"].fillna(False).sum()),
            "label_is_exact_true": int(df["label_is_exact"].fillna(False).sum()),
            "min_length": df["length"].dropna().min() if df["length"].notna().any() else None,
            "max_length": df["length"].dropna().max() if df["length"].notna().any() else None,
        })

        print(f"[OK] saved: {out_path}")
        print(f"     rows={len(df)}, cols={len(df.columns)}")

    summary_df = build_summary(summary_rows)
    summary_path = report_dir / "standardized_schema_summary.csv"
    save_df(summary_df, summary_path)

    print("\nDone.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()