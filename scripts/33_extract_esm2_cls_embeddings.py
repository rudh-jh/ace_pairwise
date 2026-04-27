from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_sequence(seq: object) -> str:
    return str(seq).strip().upper()


def load_esm2_t6(device: str):
    try:
        import esm
    except Exception as e:
        raise RuntimeError(
            "Cannot import esm. Please install fair-esm first:\n"
            "python -m pip install fair-esm -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from e

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def mean_pool_sequence_embedding(
    representations: torch.Tensor,
    seqs: list[str],
) -> np.ndarray:
    """
    representations: [B, T, D]
    ESM token layout usually:
    token 0 = BOS/CLS
    token 1..L = amino acids
    token L+1 = EOS
    """
    embs = []

    for i, seq in enumerate(seqs):
        L = len(seq)
        seq_repr = representations[i, 1 : L + 1, :]
        pooled = seq_repr.mean(dim=0)
        embs.append(pooled.detach().cpu().numpy())

    return np.vstack(embs)


@torch.no_grad()
def extract_embeddings(
    sequences: list[str],
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    model, alphabet, batch_converter = load_esm2_t6(device)

    all_rows = []

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start : start + batch_size]
        batch_data = [(seq, seq) for seq in batch_seqs]

        labels, strs, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        outputs = model(tokens, repr_layers=[6], return_contacts=False)
        reps = outputs["representations"][6]

        pooled = mean_pool_sequence_embedding(reps, batch_seqs)

        for seq, emb in zip(batch_seqs, pooled):
            row = {
                "sequence": seq,
                "length": len(seq),
            }
            for j, value in enumerate(emb):
                row[f"esm2_{j:03d}"] = float(value)
            all_rows.append(row)

        print(f"Processed {min(start + batch_size, len(sequences))}/{len(sequences)}")

    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="acepair_cls_2_3",
        choices=["acepair_cls_2_3", "acepair_cls_2_5"],
    )
    parser.add_argument(
        "--data_dir",
        default="data/final/classification",
    )
    parser.add_argument(
        "--out_dir",
        default="data/final/classification/features/esm2",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / args.data_dir / f"{args.dataset}.csv"
    out_dir = project_root / args.out_dir
    ensure_dir(out_dir)

    out_path = out_dir / f"{args.dataset}_esm2_t6_320.csv"

    if out_path.exists() and not args.overwrite:
        print(f"Output already exists, skip: {out_path}")
        print("Use --overwrite if you want to regenerate it.")
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 80)
    print("Dataset:", args.dataset)
    print("Input:", input_path)
    print("Output:", out_path)
    print("Device:", device)

    df = pd.read_csv(input_path)
    df["sequence"] = df["sequence"].map(normalize_sequence)

    sequences = sorted(df["sequence"].dropna().unique().tolist())

    print("Unique sequences:", len(sequences))

    emb_df = extract_embeddings(
        sequences=sequences,
        batch_size=args.batch_size,
        device=device,
    )

    emb_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Saved:", out_path)
    print("Shape:", emb_df.shape)
    print("Done.")


if __name__ == "__main__":
    main()