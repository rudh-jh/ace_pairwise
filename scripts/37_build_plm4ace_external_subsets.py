from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_sequence(seq: object) -> str:
    return str(seq).strip().upper()


def normalize_col_name(col: object) -> str:
    return str(col).strip().lower().replace("_", " ")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {normalize_col_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_col_name(cand)
        if key in norm_map:
            return norm_map[key]

    return None


def pick_sheet(xlsx_path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    sheets = pd.read_excel(xlsx_path, sheet_name=None, na_filter=False)

    print("Available sheets:", list(sheets.keys()))

    if sheet_name is not None:
        if sheet_name not in sheets:
            raise ValueError(
                f"Cannot find sheet={sheet_name}. Available sheets: {list(sheets.keys())}"
            )
        print("Selected sheet:", sheet_name)
        return sheets[sheet_name].copy()

    # pLM4ACE 里一般有 Before clean / After clean，优先用 After clean
    for preferred in ["After clean", "after clean", "Cleaned dataset", "cleaned dataset"]:
        if preferred in sheets:
            print("Selected sheet:", preferred)
            return sheets[preferred].copy()

    # 如果没有明确的 After clean，再自动找
    best_name = None
    best_df = None
    best_score = -1

    for name, df in sheets.items():
        seq_col = find_col(df, ["sequence", "sequences", "seq", "peptide", "peptide sequence"])
        label_col = find_col(df, ["label", "labels", "class", "activity", "y", "group"])
        ic50_col = find_col(df, ["IC50", "IC50 ", "ic50", "ic50 um", "IC50 uM"])

        score = 0
        if seq_col is not None:
            score += 10
        if label_col is not None:
            score += 8
        if ic50_col is not None:
            score += 8
        if "after" in str(name).lower():
            score += 20
        score += min(len(df), 2000) / 2000

        if score > best_score:
            best_score = score
            best_name = name
            best_df = df.copy()

    print("Selected sheet:", best_name)
    return best_df


def parse_group_to_cls_label(series: pd.Series) -> pd.Series:
    """
    pLM4ACE README 的标签方向：
    0 = high activity / positive
    1 = low or non-activity / negative

    本项目统一方向：
    cls_label = 1 表示 high activity
    cls_label = 0 表示 low/non activity
    """
    raw = series.astype(str).str.strip().str.lower()

    def map_one(x: str):
        if x in {"0", "0.0"}:
            return 1
        if x in {"1", "1.0"}:
            return 0

        if x in {"positive", "pos", "high", "high activity", "active", "p"}:
            return 1

        if x in {
            "negative",
            "neg",
            "low",
            "low activity",
            "non",
            "non activity",
            "non-activity",
            "inactive",
            "n",
        }:
            return 0

        return None

    return raw.map(map_one)


def standardize_plm4ace(df: pd.DataFrame, ic50_threshold_uM: float = 50.0) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    seq_col = find_col(df, ["sequence", "sequences", "seq", "peptide", "peptide sequence"])
    label_col = find_col(df, ["label", "labels", "class", "activity", "y"])
    group_col = find_col(df, ["group"])
    ic50_col = find_col(df, ["IC50", "ic50", "IC50 uM", "ic50 um", "IC50(μM)", "IC50 (μM)"])

    if seq_col is None:
        raise ValueError(f"Cannot find sequence column. Columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["sequence"] = df[seq_col].map(normalize_sequence)
    out = out[out["sequence"].str.len() > 0].copy()
    out["length"] = out["sequence"].str.len()

    if label_col is not None:
        raw_label = pd.to_numeric(df.loc[out.index, label_col], errors="coerce")
        keep = raw_label.notna()
        out = out.loc[keep].copy()
        raw_label = raw_label.loc[keep].astype(int)

        out["plm4ace_label"] = raw_label.values
        out["cls_label"] = out["plm4ace_label"].map({0: 1, 1: 0}).astype(int)
        out["cls_source"] = f"{label_col} converted from pLM4ACE direction"

        if group_col is not None:
            out["raw_group"] = df.loc[out.index, group_col].astype(str).values

    elif ic50_col is not None:
        ic50 = pd.to_numeric(df.loc[out.index, ic50_col], errors="coerce")
        keep = ic50.notna()
        out = out.loc[keep].copy()
        ic50 = ic50.loc[keep]

        out["label_ic50_uM"] = ic50.astype(float).values
        out["cls_label"] = (out["label_ic50_uM"] <= ic50_threshold_uM).astype(int)
        out["plm4ace_label"] = out["cls_label"].map({1: 0, 0: 1}).astype(int)
        out["cls_source"] = f"{ic50_col} <= {ic50_threshold_uM} uM"

        if group_col is not None:
            out["raw_group"] = df.loc[out.index, group_col].astype(str).values

    elif group_col is not None:
        raw_group = df.loc[out.index, group_col].astype(str).str.strip().str.upper()
        group_map = {
            "A": 1,
            "B": 0,
            "0": 1,
            "1": 0,
            "POSITIVE": 1,
            "NEGATIVE": 0,
        }

        cls = raw_group.map(group_map)
        keep = cls.notna()
        out = out.loc[keep].copy()
        cls = cls.loc[keep].astype(int)

        out["raw_group"] = raw_group.loc[keep].values
        out["cls_label"] = cls.values
        out["plm4ace_label"] = out["cls_label"].map({1: 0, 0: 1}).astype(int)
        out["cls_source"] = f"{group_col} converted from pLM4ACE group"

    else:
        raise ValueError(f"Cannot find label, group, or IC50 column. Columns: {list(df.columns)}")

    out["cls_label"] = out["cls_label"].astype(int)
    out["cls_label_name"] = out["cls_label"].map(
        {
            1: "high_activity",
            0: "low_or_non_activity",
        }
    )

    out["source_dataset"] = "pLM4ACE_cleaned"
    out["sample_weight"] = 1.0

    before = len(out)
    out = out.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    after = len(out)

    print(f"Deduplicated sequences: {before} -> {after}")

    preferred_cols = [
        "sequence",
        "length",
        "label_ic50_uM",
        "cls_label",
        "cls_label_name",
        "plm4ace_label",
        "cls_source",
        "raw_group",
        "source_dataset",
        "sample_weight",
    ]

    cols = [c for c in preferred_cols if c in out.columns]
    others = [c for c in out.columns if c not in cols]
    out = out[cols + others]

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

    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/external/plm4ace/Orignal dataset and Cleaned dataset.xlsx",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional sheet name. Recommended: After clean",
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

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find: {input_path}\n"
            f"Please download pLM4ACE Excel dataset and place it here."
        )

    raw_df = pick_sheet(input_path, args.sheet)
    std_df = standardize_plm4ace(raw_df, ic50_threshold_uM=args.ic50_threshold_uM)

    datasets = {
        "plm4ace_cleaned_2_3": std_df[std_df["length"].isin([2, 3])].copy(),
        "plm4ace_cleaned_2_5": std_df[std_df["length"].between(2, 5)].copy(),
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