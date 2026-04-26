from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_input_paths(cfg: dict) -> dict[str, Path]:
    pepdb_root = Path(cfg["paths"]["pepdb_root"])
    inputs = cfg["inputs"]

    resolved = {}
    for name, rel_path in inputs.items():
        resolved[name] = pepdb_root / rel_path
    return resolved


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def guess_sequence_col(columns: list[str]) -> str | None:
    candidates = ["sequence", "seq", "peptide", "peptide_sequence", "canonical_sequence"]
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def guess_length_col(columns: list[str]) -> str | None:
    candidates = ["length", "peptide_length", "seq_len"]
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def build_audit_summary(name: str, path: Path, df: pd.DataFrame) -> dict:
    columns = list(df.columns)
    sequence_col = guess_sequence_col(columns)
    length_col = guess_length_col(columns)

    summary = {
        "source_name": name,
        "file_path": str(path),
        "exists": True,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "sequence_col_guess": sequence_col,
        "length_col_guess": length_col,
        "all_columns": " | ".join(columns),
    }

    if sequence_col is not None:
        summary["non_null_sequence"] = int(df[sequence_col].notna().sum())
        try:
            seq_series = df[sequence_col].dropna().astype(str).str.upper().str.strip()
            summary["unique_sequence_count"] = int(seq_series.nunique())
            summary["min_seq_len_observed"] = int(seq_series.str.len().min()) if not seq_series.empty else None
            summary["max_seq_len_observed"] = int(seq_series.str.len().max()) if not seq_series.empty else None
        except Exception:
            summary["unique_sequence_count"] = None
            summary["min_seq_len_observed"] = None
            summary["max_seq_len_observed"] = None
    else:
        summary["non_null_sequence"] = None
        summary["unique_sequence_count"] = None
        summary["min_seq_len_observed"] = None
        summary["max_seq_len_observed"] = None

    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "dataset_exact_core_v1.yaml"
    cfg = load_config(str(config_path))

    report_dir = project_root / cfg["outputs"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    input_paths = resolve_input_paths(cfg)

    audit_rows = []
    missing_rows = []

    for name, path in input_paths.items():
        print(f"\n[CHECK] {name}")
        print(path)

        if not path.exists():
            print("  -> MISSING")
            missing_rows.append({
                "source_name": name,
                "file_path": str(path),
                "exists": False,
            })
            continue

        df = safe_read_csv(path)
        row = build_audit_summary(name, path, df)
        audit_rows.append(row)

        print(f"  -> rows: {row['n_rows']}, cols: {row['n_cols']}")
        print(f"  -> sequence guess: {row['sequence_col_guess']}")
        print(f"  -> length guess: {row['length_col_guess']}")

    audit_df = pd.DataFrame(audit_rows)
    missing_df = pd.DataFrame(missing_rows)

    audit_csv = report_dir / "source_audit_summary.csv"
    missing_csv = report_dir / "missing_sources.csv"
    audit_md = report_dir / "source_audit_report.md"

    audit_df.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    missing_df.to_csv(missing_csv, index=False, encoding="utf-8-sig")

    with open(audit_md, "w", encoding="utf-8") as f:
        f.write("# Source Audit Report\n\n")
        f.write("## Existing Sources\n\n")
        if audit_df.empty:
            f.write("No readable sources found.\n")
        else:
            for _, row in audit_df.iterrows():
                f.write(f"### {row['source_name']}\n")
                f.write(f"- path: `{row['file_path']}`\n")
                f.write(f"- rows: {row['n_rows']}\n")
                f.write(f"- cols: {row['n_cols']}\n")
                f.write(f"- sequence_col_guess: `{row['sequence_col_guess']}`\n")
                f.write(f"- length_col_guess: `{row['length_col_guess']}`\n")
                f.write(f"- unique_sequence_count: {row['unique_sequence_count']}\n")
                f.write(f"- min_seq_len_observed: {row['min_seq_len_observed']}\n")
                f.write(f"- max_seq_len_observed: {row['max_seq_len_observed']}\n")
                f.write(f"- all_columns: {row['all_columns']}\n\n")

        f.write("## Missing Sources\n\n")
        if missing_df.empty:
            f.write("None.\n")
        else:
            for _, row in missing_df.iterrows():
                f.write(f"- {row['source_name']}: `{row['file_path']}`\n")

    print("\nDone.")
    print(f"Audit summary saved to: {audit_csv}")
    print(f"Markdown report saved to: {audit_md}")
    print(f"Missing sources saved to: {missing_csv}")


if __name__ == "__main__":
    main()