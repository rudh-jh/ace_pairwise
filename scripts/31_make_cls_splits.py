from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def split_one_seed(
    df: pd.DataFrame,
    seed: int,
    test_ratio: float,
    valid_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["cls_label"].astype(int)

    train_valid_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    valid_ratio_inside = valid_ratio / (1.0 - test_ratio)

    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_ratio_inside,
        random_state=seed,
        shuffle=True,
        stratify=train_valid_df["cls_label"].astype(int),
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def summarize_split(
    dataset: str,
    seed: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for split_name, sub in [
        ("train", train_df),
        ("valid", valid_df),
        ("test", test_df),
    ]:
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "split": split_name,
                "metric": "n_total",
                "value": len(sub),
            }
        )

        for k, v in sub["cls_label"].value_counts().sort_index().items():
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "split": split_name,
                    "metric": f"cls_label_{k}",
                    "value": int(v),
                }
            )

        for k, v in sub["length"].value_counts().sort_index().items():
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "split": split_name,
                    "metric": f"length_{k}",
                    "value": int(v),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="acepair_cls_2_3",
        # choices=["acepair_cls_2_3", "acepair_cls_2_5"],
    )
    parser.add_argument(
        "--data_dir",
        default="data/final/classification",
    )
    parser.add_argument(
        "--out_root",
        default="data/final/classification/splits",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 52, 62, 72, 82],
    )
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--valid_ratio", type=float, default=0.16)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / args.data_dir / f"{args.dataset}.csv"
    out_root = project_root / args.out_root / args.dataset
    ensure_dir(out_root)

    df = pd.read_csv(input_path)
    df["cls_label"] = df["cls_label"].astype(int)

    label_counts = df["cls_label"].value_counts()
    if label_counts.min() < 3:
        raise ValueError(f"Too few samples in at least one class: {label_counts.to_dict()}")

    all_summary = []

    for seed in args.seeds:
        train_df, valid_df, test_df = split_one_seed(
            df=df,
            seed=seed,
            test_ratio=args.test_ratio,
            valid_ratio=args.valid_ratio,
        )

        seed_dir = out_root / f"seed_{seed}"
        save_csv(train_df, seed_dir / "train.csv")
        save_csv(valid_df, seed_dir / "valid.csv")
        save_csv(test_df, seed_dir / "test.csv")

        summary = summarize_split(args.dataset, seed, train_df, valid_df, test_df)
        save_csv(summary, seed_dir / "split_summary.csv")
        all_summary.append(summary)

        print("=" * 80)
        print(f"dataset={args.dataset} | seed={seed}")
        print("train:", len(train_df), train_df["cls_label"].value_counts().to_dict())
        print("valid:", len(valid_df), valid_df["cls_label"].value_counts().to_dict())
        print("test :", len(test_df), test_df["cls_label"].value_counts().to_dict())

    all_summary_df = pd.concat(all_summary, ignore_index=True)
    save_csv(all_summary_df, out_root / "all_split_summary.csv")

    print("=" * 80)
    print("Done.")
    print("saved:", out_root)


if __name__ == "__main__":
    main()