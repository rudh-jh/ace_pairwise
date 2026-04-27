from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parents[1]
src_root = project_root / "src"

for p in [project_root, src_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ace_pre.baselines.tabular_features import build_descriptor_frame
from ace_pre.baselines.tensor_flat_features import build_a1_flat_feature_frame


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}
PAD_ID = 0
UNK_ID = len(AA_ORDER) + 1


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_sequence(seq: object) -> str:
    return str(seq).strip().upper()


def encode_sequences(df: pd.DataFrame, max_len: int = 5) -> np.ndarray:
    arr = np.zeros((len(df), max_len), dtype=np.int64)

    for i, seq in enumerate(df["sequence"].map(normalize_sequence).tolist()):
        seq = seq[:max_len]
        for j, aa in enumerate(seq):
            arr[i, j] = AA_TO_ID.get(aa, UNK_ID)

    return arr


def get_sample_weight(df: pd.DataFrame) -> np.ndarray:
    if "sample_weight" not in df.columns:
        return np.ones(len(df), dtype=np.float32)

    w = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).astype(float).values
    w = np.clip(w, 0.05, 10.0)
    return w.astype(np.float32)


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def align_columns(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_x = valid_x.reindex(columns=train_x.columns, fill_value=0.0)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0.0)
    return train_x, valid_x, test_x


class ExtraFeatureTransformer:
    def __init__(self, use_pca: bool = False, pca_dim: int = 64):
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca = None

    def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
        x_imp = self.imputer.fit_transform(x)
        x_scaled = self.scaler.fit_transform(x_imp)

        if self.use_pca:
            n_components = min(self.pca_dim, x_scaled.shape[0] - 1, x_scaled.shape[1])
            n_components = max(1, n_components)
            self.pca = PCA(n_components=n_components, random_state=42)
            x_scaled = self.pca.fit_transform(x_scaled)

        return x_scaled.astype(np.float32)

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        x_imp = self.imputer.transform(x)
        x_scaled = self.scaler.transform(x_imp)

        if self.use_pca and self.pca is not None:
            x_scaled = self.pca.transform(x_scaled)

        return x_scaled.astype(np.float32)


def build_extra_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str,
    pca_dim: int = 64,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    parts_train = []
    parts_valid = []
    parts_test = []

    if "Descriptor" in variant:
        tr = numeric_frame(build_descriptor_frame(train_df, sequence_col="sequence"))
        va = numeric_frame(build_descriptor_frame(valid_df, sequence_col="sequence"))
        te = numeric_frame(build_descriptor_frame(test_df, sequence_col="sequence"))
        tr, va, te = align_columns(tr, va, te)

        transformer = ExtraFeatureTransformer(use_pca=False)
        parts_train.append(transformer.fit_transform(tr))
        parts_valid.append(transformer.transform(va))
        parts_test.append(transformer.transform(te))

    if "A1PCA" in variant:
        tr = numeric_frame(build_a1_flat_feature_frame(train_df, sequence_col="sequence"))
        va = numeric_frame(build_a1_flat_feature_frame(valid_df, sequence_col="sequence"))
        te = numeric_frame(build_a1_flat_feature_frame(test_df, sequence_col="sequence"))
        tr, va, te = align_columns(tr, va, te)

        transformer = ExtraFeatureTransformer(use_pca=True, pca_dim=pca_dim)
        parts_train.append(transformer.fit_transform(tr))
        parts_valid.append(transformer.transform(va))
        parts_test.append(transformer.transform(te))

    if not parts_train:
        return None, None, None, 0

    x_train = np.concatenate(parts_train, axis=1)
    x_valid = np.concatenate(parts_valid, axis=1)
    x_test = np.concatenate(parts_test, axis=1)

    return x_train, x_valid, x_test, int(x_train.shape[1])


class PeptideClsDataset(Dataset):
    def __init__(
        self,
        tokens: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        extra: np.ndarray | None = None,
    ):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

        if extra is None:
            self.extra = None
        else:
            self.extra = torch.tensor(extra, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {
            "tokens": self.tokens[idx],
            "labels": self.labels[idx],
            "weights": self.weights[idx],
        }

        if self.extra is not None:
            item["extra"] = self.extra[idx]

        return item


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [B, L, D], mask: [B, L]
        score = self.score(h).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        attn = torch.softmax(score, dim=-1)
        pooled = torch.sum(h * attn.unsqueeze(-1), dim=1)
        return pooled


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        extra_dim: int = 0,
        emb_dim: int = 32,
        hidden_dim: int = 32,
        dropout: float = 0.25,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.pool = AttentionPool(gru_out_dim)

        self.extra_dim = extra_dim
        fusion_dim = gru_out_dim + extra_dim

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, tokens: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        mask = tokens.ne(PAD_ID)
        emb = self.embedding(tokens)
        h, _ = self.gru(emb)
        z = self.pool(h, mask)

        if self.extra_dim > 0 and extra is not None:
            z = torch.cat([z, extra], dim=-1)

        logit = self.head(z).squeeze(-1)
        return logit


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": safe_auc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
    }


def best_threshold_by_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t = 0.5
    best_mcc = -999.0

    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = float(t)

    return best_t


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    probs = []
    labels = []

    for batch in loader:
        tokens = batch["tokens"].to(device)
        extra = batch.get("extra")
        if extra is not None:
            extra = extra.to(device)

        logits = model(tokens, extra)
        prob = torch.sigmoid(logits)

        probs.append(prob.detach().cpu().numpy())
        labels.append(batch["labels"].detach().cpu().numpy())

    return np.concatenate(labels), np.concatenate(probs)


def train_one_model(
    train_tokens: np.ndarray,
    valid_tokens: np.ndarray,
    test_tokens: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_valid: np.ndarray,
    w_test: np.ndarray,
    extra_train: np.ndarray | None,
    extra_valid: np.ndarray | None,
    extra_test: np.ndarray | None,
    extra_dim: int,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> tuple[dict, dict, np.ndarray, np.ndarray, float]:
    set_seed(seed)

    train_ds = PeptideClsDataset(train_tokens, y_train, w_train, extra_train)
    valid_ds = PeptideClsDataset(valid_tokens, y_valid, w_valid, extra_valid)
    test_ds = PeptideClsDataset(test_tokens, y_test, w_test, extra_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = GRUClassifier(
        vocab_size=UNK_ID + 1,
        extra_dim=extra_dim,
        emb_dim=32,
        hidden_dim=32,
        dropout=0.25,
        bidirectional=True,
    ).to(device)

    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_auc = -999.0
    best_epoch = -1
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_total = 0

        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)
            extra = batch.get("extra")
            if extra is not None:
                extra = extra.to(device)

            optimizer.zero_grad()
            logits = model(tokens, extra)
            loss_vec = criterion(logits, labels)
            loss = (loss_vec * weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(labels)
            n_total += len(labels)

        valid_y, valid_prob = predict_probs(model, valid_loader, device)
        valid_auc = safe_auc(valid_y, valid_prob)

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"epoch={epoch:03d} | train_loss={total_loss / max(1, n_total):.4f} | "
                f"valid_auc={valid_auc:.4f} | best_auc={best_auc:.4f}"
            )

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_y, valid_prob = predict_probs(model, valid_loader, device)
    test_y, test_prob = predict_probs(model, test_loader, device)

    threshold = best_threshold_by_mcc(valid_y, valid_prob)
    valid_metrics = evaluate_binary(valid_y, valid_prob, threshold)
    test_metrics = evaluate_binary(test_y, test_prob, threshold)

    info = {
        "best_epoch": best_epoch,
        "best_valid_auc": best_auc,
    }

    return valid_metrics, test_metrics, valid_prob, test_prob, threshold, info


def summarize_results(all_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "threshold",
        "acc",
        "bacc",
        "auc",
        "auprc",
        "f1",
        "mcc",
        "precision",
        "sensitivity",
        "specificity",
    ]

    test_df = all_df[all_df["split"] == "test"].copy()
    rows = []

    for model, g in test_df.groupby("model"):
        row = {
            "model": model,
            "n_seeds": g["seed"].nunique(),
            "extra_dim_mean": float(g["extra_dim"].mean()),
        }

        for m in metric_cols:
            row[f"test_{m}_mean"] = float(g[m].mean())
            row[f"test_{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0

        rows.append(row)

    summary = pd.DataFrame(rows)

    if len(summary) > 0:
        summary = summary.sort_values(
            ["test_mcc_mean", "test_auc_mean", "test_bacc_mean"],
            ascending=[False, False, False],
        )

    return summary


def run_one_seed(
    dataset: str,
    seed_dir: Path,
    out_dir: Path,
    variants: list[str],
    args,
) -> pd.DataFrame:
    seed = int(seed_dir.name.replace("seed_", ""))
    set_seed(seed)

    train_df = pd.read_csv(seed_dir / "train.csv")
    valid_df = pd.read_csv(seed_dir / "valid.csv")
    test_df = pd.read_csv(seed_dir / "test.csv")

    y_train = train_df["cls_label"].astype(int).values.astype(np.float32)
    y_valid = valid_df["cls_label"].astype(int).values.astype(np.float32)
    y_test = test_df["cls_label"].astype(int).values.astype(np.float32)

    w_train = get_sample_weight(train_df)
    w_valid = get_sample_weight(valid_df)
    w_test = get_sample_weight(test_df)

    train_tokens = encode_sequences(train_df, max_len=args.max_len)
    valid_tokens = encode_sequences(valid_df, max_len=args.max_len)
    test_tokens = encode_sequences(test_df, max_len=args.max_len)

    print("=" * 80)
    print(f"dataset={dataset} | seed={seed}")
    print(f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    print("train label:", train_df["cls_label"].value_counts().to_dict())

    rows = []

    pred_test = pd.DataFrame(
        {
            "sequence": test_df["sequence"].values,
            "y_true": y_test.astype(int),
        }
    )

    for variant in variants:
        print("-" * 80)
        print(f"Variant: {variant}")

        extra_train, extra_valid, extra_test, extra_dim = build_extra_features(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            variant=variant,
            pca_dim=args.pca_dim,
        )

        print(f"extra_dim={extra_dim}")

        valid_metrics, test_metrics, valid_prob, test_prob, threshold, info = train_one_model(
            train_tokens=train_tokens,
            valid_tokens=valid_tokens,
            test_tokens=test_tokens,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            w_train=w_train,
            w_valid=w_valid,
            w_test=w_test,
            extra_train=extra_train,
            extra_valid=extra_valid,
            extra_test=extra_test,
            extra_dim=extra_dim,
            seed=seed,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )

        model_name = variant

        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model": model_name,
                "split": "valid",
                "extra_dim": extra_dim,
                **valid_metrics,
                "params": json.dumps(info, ensure_ascii=False),
            }
        )

        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model": model_name,
                "split": "test",
                "extra_dim": extra_dim,
                **test_metrics,
                "params": json.dumps(info, ensure_ascii=False),
            }
        )

        pred_test[f"{model_name}_prob"] = test_prob
        pred_test[f"{model_name}_pred"] = (test_prob >= threshold).astype(int)

        print(
            f"{model_name} | threshold={threshold:.2f} | "
            f"valid_mcc={valid_metrics['mcc']:.4f} | "
            f"test_mcc={test_metrics['mcc']:.4f} | "
            f"test_auc={test_metrics['auc']:.4f} | "
            f"test_bacc={test_metrics['bacc']:.4f}"
        )

    seed_out = out_dir / f"seed_{seed}"
    ensure_dir(seed_out)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(seed_out / "gru4ace_like_cls_results.csv", index=False, encoding="utf-8-sig")
    pred_test.to_csv(seed_out / "gru4ace_like_test_predictions.csv", index=False, encoding="utf-8-sig")

    return result_df


def main() -> None:
    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="acepair_cls_2_3",
        choices=["acepair_cls_2_3", "acepair_cls_2_5"],
    )
    parser.add_argument(
        "--split_root",
        default="data/final/classification/splits",
    )
    parser.add_argument(
        "--out_root",
        default="artifacts/classification/gru4ace_like",
    )
    parser.add_argument("--max_len", type=int, default=5)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    split_root = project_root / args.split_root / args.dataset
    out_dir = project_root / args.out_root / args.dataset
    ensure_dir(out_dir)

    seed_dirs = sorted([p for p in split_root.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed dirs found under: {split_root}")

    variants = [
        "GRUOnly",
        "GRU_Descriptor",
        "GRU_A1PCA",
        "GRU_Descriptor_A1PCA",
    ]

    print("=" * 80)
    print("Dataset:", args.dataset)
    print("Device:", args.device)
    print("Variants:", variants)

    all_results = []

    for seed_dir in seed_dirs:
        one = run_one_seed(
            dataset=args.dataset,
            seed_dir=seed_dir,
            out_dir=out_dir,
            variants=variants,
            args=args,
        )
        all_results.append(one)

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(out_dir / "gru4ace_like_cls_all_seed_results.csv", index=False, encoding="utf-8-sig")

    summary = summarize_results(all_df)
    summary.to_csv(out_dir / "gru4ace_like_cls_summary.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Summary:")
    print(summary)
    print("=" * 80)
    print(f"Done in {time.time() - t0:.1f}s")
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()