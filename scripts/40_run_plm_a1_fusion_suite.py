from __future__ import annotations

"""
PLM-informed A1 fusion suite for ace_pairwise.

This script implements three practical fusion routes based on your current repo:

Route 1, classification feature-level fusion:
    ESM2-LR
    A1FlatSelected-LR
    A1FlatSelected + ESM2-PCA -> LR

Route 2, regression feature-level fusion:
    A1FlatSelected-SVR
    ESM2-PCA-SVR
    A1FlatSelected + ESM2-PCA -> SVR

Route 3, lightweight PLM-informed A1 interaction fusion:
    A1FlatSelected + ESM2-PCA + interaction(A1Selected_topK, ESM2-PCA) -> LR/SVR

Note about Route 3:
    Your current repo stores sequence-level ESM2 embeddings as CSV files with esm2_* columns.
    That is enough for a "PLM-informed A1Flat" route via PCA and interaction features.
    True residue-level injection into the [C,5,5] A1 tensor requires saving token-level ESM2
    embeddings first, so this script implements a runnable low-risk version that fits your current files.

Typical commands:
    # Classification, 2-3 aa
    python scripts/40_run_plm_a1_fusion_suite.py --task cls --dataset plm4ace_cleaned_2_3

    # Classification, 2-5 aa
    python scripts/40_run_plm_a1_fusion_suite.py --task cls --dataset plm4ace_cleaned_2_5

    # Regression, requires you to provide a sequence-level ESM2 feature csv for regression sequences
    python scripts/40_run_plm_a1_fusion_suite.py --task reg --reg_esm_path data/final/regression/features/esm2/a1_joint_2_5_esm2_t6_320.csv
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in [PROJECT_ROOT, SRC_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ace_pre.baselines.tensor_flat_features import build_a1_flat_feature_frame


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_sequence(seq: object) -> str:
    return str(seq).strip().upper()


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    vals: list[float] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def get_sample_weight(df: pd.DataFrame) -> np.ndarray | None:
    if "sample_weight" not in df.columns:
        return None
    w = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).astype(float).values
    return np.clip(w, 0.05, 10.0)


def load_esm_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ESM2 feature file not found: {path}")
    esm_df = pd.read_csv(path)
    if "sequence" not in esm_df.columns:
        raise ValueError(f"ESM2 feature file must contain a sequence column: {path}")
    esm_df["sequence"] = esm_df["sequence"].map(normalize_sequence)
    feature_cols = [c for c in esm_df.columns if c.startswith("esm2_")]
    if not feature_cols:
        raise ValueError(f"No esm2_* columns found in: {path}")
    return esm_df[["sequence"] + feature_cols].drop_duplicates("sequence")


def build_esm_frame(df: pd.DataFrame, esm_df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["sequence"]].copy()
    tmp["sequence"] = tmp["sequence"].map(normalize_sequence)
    merged = tmp.merge(esm_df, on="sequence", how="left")
    feature_cols = [c for c in merged.columns if c.startswith("esm2_")]
    missing = merged[feature_cols].isna().any(axis=1).sum()
    if missing > 0:
        examples = merged.loc[merged[feature_cols].isna().any(axis=1), "sequence"].head(10).tolist()
        raise ValueError(
            f"Missing ESM2 embeddings for {missing} rows. Examples: {examples}.\n"
            "For regression, you may need to extract ESM2 embeddings for the regression model-input table first."
        )
    X = merged[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def build_a1full_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = build_a1_flat_feature_frame(df, sequence_col="sequence")
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def align_to_train_columns(
    X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    return X_train, X_valid, X_test


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = pd.Series(y_true).rank(method="average").to_numpy()
    b = pd.Series(y_pred).rank(method="average").to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# -----------------------------------------------------------------------------
# Transformers for A1Selected, ESM2PCA, and interactions
# -----------------------------------------------------------------------------
@dataclass
class A1SelectorBundle:
    imputer: SimpleImputer
    scaler: StandardScaler
    variance: VarianceThreshold
    selector: SelectKBest
    selected_features: list[str]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.imputer.transform(X)
        Z = self.scaler.transform(Z)
        Z = self.variance.transform(Z)
        Z = self.selector.transform(Z)
        return Z


@dataclass
class EsmPcaBundle:
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.imputer.transform(X)
        Z = self.scaler.transform(Z)
        Z = self.pca.transform(Z)
        return Z


def fit_a1_selector(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    task: str,
    k: int,
) -> A1SelectorBundle:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    variance = VarianceThreshold(threshold=1e-10)
    score_func = f_classif if task == "cls" else f_regression

    X0 = imputer.fit_transform(X_train)
    X1 = scaler.fit_transform(X0)
    X2 = variance.fit_transform(X1)

    k_eff = int(min(k, X2.shape[1]))
    selector = SelectKBest(score_func=score_func, k=k_eff)
    selector.fit(X2, y_train)

    cols_after_var = np.array(X_train.columns)[variance.get_support()]
    selected_features = cols_after_var[selector.get_support()].tolist()
    return A1SelectorBundle(imputer, scaler, variance, selector, selected_features)


def fit_esm_pca(X_train: pd.DataFrame, pca_dim: int, seed: int) -> EsmPcaBundle:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X0 = imputer.fit_transform(X_train)
    X1 = scaler.fit_transform(X0)
    n_comp = int(min(pca_dim, X1.shape[0] - 1, X1.shape[1]))
    if n_comp < 1:
        raise ValueError(f"Invalid PCA dim after bounds: {n_comp}")
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(X1)
    return EsmPcaBundle(imputer, scaler, pca)


def build_interaction_features(
    a1_selected: np.ndarray,
    esm_pca: np.ndarray,
    max_a1_for_interaction: int,
) -> np.ndarray:
    """Low-dimensional interaction approximation for PLM-informed A1Flat.

    This creates pairwise products between the first max_a1_for_interaction selected A1
    dimensions and ESM2 PCA dimensions. This is deliberately small to avoid overfitting.
    """
    a1_small = a1_selected[:, : min(max_a1_for_interaction, a1_selected.shape[1])]
    parts = []
    for j in range(esm_pca.shape[1]):
        parts.append(a1_small * esm_pca[:, [j]])
    return np.concatenate(parts, axis=1) if parts else np.empty((a1_selected.shape[0], 0))


def make_feature_matrix(
    variant: str,
    X_a1: pd.DataFrame,
    X_esm: pd.DataFrame,
    a1_bundle: A1SelectorBundle | None = None,
    esm_bundle: EsmPcaBundle | None = None,
    max_a1_for_interaction: int = 50,
) -> np.ndarray:
    if variant == "ESM2":
        return X_esm.values.astype(float)
    if variant == "A1FlatSelected":
        assert a1_bundle is not None
        return a1_bundle.transform(X_a1)
    if variant == "ESM2PCA":
        assert esm_bundle is not None
        return esm_bundle.transform(X_esm)
    if variant == "A1FlatSelected_ESM2PCA":
        assert a1_bundle is not None and esm_bundle is not None
        A = a1_bundle.transform(X_a1)
        E = esm_bundle.transform(X_esm)
        return np.concatenate([A, E], axis=1)
    if variant == "A1FlatSelected_ESM2PCAInteract":
        assert a1_bundle is not None and esm_bundle is not None
        A = a1_bundle.transform(X_a1)
        E = esm_bundle.transform(X_esm)
        I = build_interaction_features(A, E, max_a1_for_interaction=max_a1_for_interaction)
        return np.concatenate([A, E, I], axis=1)
    raise ValueError(f"Unknown feature variant: {variant}")


# -----------------------------------------------------------------------------
# Classification metrics and model
# -----------------------------------------------------------------------------
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


def evaluate_cls(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
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


def fit_lr(X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray | None, seed: int, C: float) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l2",
                    C=C,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    kwargs = {}
    if sample_weight is not None:
        kwargs["lr__sample_weight"] = sample_weight
    model.fit(X_train, y_train, **kwargs)
    return model


def predict_lr_prob(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# -----------------------------------------------------------------------------
# Regression metrics and model
# -----------------------------------------------------------------------------
def evaluate_reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": spearman_corr(y_true, y_pred),
    }


def fit_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    C: float,
    epsilon: float,
    gamma: str | float,
) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)),
        ]
    )
    kwargs = {}
    if sample_weight is not None:
        kwargs["svr__sample_weight"] = sample_weight
    model.fit(X_train, y_train, **kwargs)
    return model


# -----------------------------------------------------------------------------
# Classification runner
# -----------------------------------------------------------------------------
def run_classification(args: argparse.Namespace) -> None:
    t0 = time.time()
    dataset = args.dataset
    split_root = PROJECT_ROOT / args.cls_split_root / dataset
    esm_path = PROJECT_ROOT / args.cls_esm_dir / f"{dataset}_esm2_t6_320.csv"
    out_dir = PROJECT_ROOT / args.out_root / "classification" / dataset
    ensure_dir(out_dir)

    seed_dirs = sorted([p for p in split_root.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed dirs found under: {split_root}")
    esm_df = load_esm_features(esm_path)

    k_list = parse_int_list(args.a1_k_list)
    pca_list = parse_int_list(args.esm_pca_list)
    C_list = parse_float_list(args.lr_C_list)

    variants = [
        "ESM2",
        "A1FlatSelected",
        "A1FlatSelected_ESM2PCA",
        "A1FlatSelected_ESM2PCAInteract",
    ]

    all_rows: list[dict] = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.replace("seed_", ""))
        set_seed(seed)
        train_df = pd.read_csv(seed_dir / "train.csv")
        valid_df = pd.read_csv(seed_dir / "valid.csv")
        test_df = pd.read_csv(seed_dir / "test.csv")

        y_train = train_df["cls_label"].astype(int).values
        y_valid = valid_df["cls_label"].astype(int).values
        y_test = test_df["cls_label"].astype(int).values
        sample_weight = get_sample_weight(train_df)

        print("=" * 90)
        print(f"[CLS] dataset={dataset} seed={seed}")
        print(f"train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")

        Xtr_a1 = build_a1full_frame(train_df)
        Xva_a1 = build_a1full_frame(valid_df)
        Xte_a1 = build_a1full_frame(test_df)
        Xtr_a1, Xva_a1, Xte_a1 = align_to_train_columns(Xtr_a1, Xva_a1, Xte_a1)
        Xtr_esm = build_esm_frame(train_df, esm_df)
        Xva_esm = build_esm_frame(valid_df, esm_df)
        Xte_esm = build_esm_frame(test_df, esm_df)
        Xtr_esm, Xva_esm, Xte_esm = align_to_train_columns(Xtr_esm, Xva_esm, Xte_esm)

        pred_test = pd.DataFrame({"sequence": test_df["sequence"].values, "y_true": y_test})

        seed_out = out_dir / f"seed_{seed}"
        ensure_dir(seed_out)

        for variant in variants:
            print("-" * 90)
            print(f"Variant: {variant}")
            candidates: list[tuple[float, float, float, dict, Pipeline, np.ndarray, np.ndarray, dict, dict]] = []

            if variant == "ESM2":
                grid = [(None, None, C) for C in C_list]
            elif variant == "A1FlatSelected":
                grid = [(k, None, C) for k in k_list for C in C_list]
            else:
                grid = [(k, p, C) for k in k_list for p in pca_list for C in C_list]

            for k, pca_dim, C in grid:
                try:
                    a1_bundle = fit_a1_selector(Xtr_a1, y_train, task="cls", k=int(k)) if k is not None else None
                    esm_bundle = fit_esm_pca(Xtr_esm, pca_dim=int(pca_dim), seed=seed) if pca_dim is not None else None

                    Z_train = make_feature_matrix(
                        variant,
                        Xtr_a1,
                        Xtr_esm,
                        a1_bundle=a1_bundle,
                        esm_bundle=esm_bundle,
                        max_a1_for_interaction=args.max_a1_for_interaction,
                    )
                    Z_valid = make_feature_matrix(
                        variant,
                        Xva_a1,
                        Xva_esm,
                        a1_bundle=a1_bundle,
                        esm_bundle=esm_bundle,
                        max_a1_for_interaction=args.max_a1_for_interaction,
                    )
                    Z_test = make_feature_matrix(
                        variant,
                        Xte_a1,
                        Xte_esm,
                        a1_bundle=a1_bundle,
                        esm_bundle=esm_bundle,
                        max_a1_for_interaction=args.max_a1_for_interaction,
                    )
                    model = fit_lr(Z_train, y_train, sample_weight=sample_weight, seed=seed, C=float(C))
                    valid_prob = predict_lr_prob(model, Z_valid)
                    threshold = best_threshold_by_mcc(y_valid, valid_prob)
                    valid_metrics = evaluate_cls(y_valid, valid_prob, threshold)
                    test_prob = predict_lr_prob(model, Z_test)
                    test_metrics = evaluate_cls(y_test, test_prob, threshold)
                    params = {
                        "k": k,
                        "pca_dim": pca_dim,
                        "C": C,
                        "max_a1_for_interaction": args.max_a1_for_interaction if "Interact" in variant else None,
                    }
                    candidates.append(
                        (
                            valid_metrics["mcc"],
                            valid_metrics["auc"],
                            valid_metrics["bacc"],
                            params,
                            model,
                            test_prob,
                            Z_train,
                            valid_metrics,
                            test_metrics,
                        )
                    )
                except Exception as e:
                    print(f"[SKIP] {variant} k={k} pca={pca_dim} C={C}: {e}")

            if not candidates:
                print(f"[WARN] no candidate succeeded for {variant}")
                continue

            candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            valid_mcc, valid_auc, valid_bacc, params, model, test_prob, Z_train_best, valid_metrics, test_metrics = candidates[0]
            model_name = f"{variant}-LR"
            pred_test[f"{model_name}_prob"] = test_prob
            pred_test[f"{model_name}_pred"] = (test_prob >= valid_metrics["threshold"]).astype(int)

            base = {
                "dataset": dataset,
                "seed": seed,
                "feature_set": variant,
                "model": model_name,
                "base_model": "LR",
                "n_features": int(Z_train_best.shape[1]),
                "params": json.dumps(params, ensure_ascii=False),
            }
            all_rows.append({**base, "split": "valid", **valid_metrics})
            all_rows.append({**base, "split": "test", **test_metrics})

            print(
                f"{model_name} | n_features={Z_train_best.shape[1]} | params={params} | "
                f"valid_mcc={valid_metrics['mcc']:.4f} | test_mcc={test_metrics['mcc']:.4f} | "
                f"test_auc={test_metrics['auc']:.4f} | test_bacc={test_metrics['bacc']:.4f}"
            )

        pred_test.to_csv(seed_out / "plm_a1_fusion_cls_test_predictions.csv", index=False, encoding="utf-8-sig")

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "plm_a1_fusion_cls_all_seed_results.csv", index=False, encoding="utf-8-sig")
    summary = summarize_cls(all_df)
    summary.to_csv(out_dir / "plm_a1_fusion_cls_summary.csv", index=False, encoding="utf-8-sig")
    print("=" * 90)
    print(summary)
    print("Saved:", out_dir)
    print(f"Done in {time.time() - t0:.1f}s")


def summarize_cls(all_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["threshold", "acc", "bacc", "auc", "auprc", "f1", "mcc", "precision", "sensitivity", "specificity"]
    test_df = all_df[all_df["split"] == "test"].copy()
    rows = []
    for (feature_set, model), g in test_df.groupby(["feature_set", "model"]):
        row = {
            "feature_set": feature_set,
            "model": model,
            "n_seeds": g["seed"].nunique(),
            "n_features_mean": float(g["n_features"].mean()),
            "n_features_std": float(g["n_features"].std(ddof=1)) if len(g) > 1 else 0.0,
        }
        for m in metric_cols:
            row[f"test_{m}_mean"] = float(g[m].mean())
            row[f"test_{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["test_mcc_mean", "test_auc_mean", "test_bacc_mean"], ascending=[False, False, False])
    return out


# -----------------------------------------------------------------------------
# Regression runner
# -----------------------------------------------------------------------------
def run_regression(args: argparse.Namespace) -> None:
    t0 = time.time()
    split_root = PROJECT_ROOT / args.reg_split_root
    esm_path = PROJECT_ROOT / args.reg_esm_path
    out_dir = PROJECT_ROOT / args.out_root / "regression" / esm_path.stem
    ensure_dir(out_dir)

    seed_dirs = sorted([p for p in split_root.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed dirs found under: {split_root}")
    esm_df = load_esm_features(esm_path)

    k_list = parse_int_list(args.a1_k_list)
    pca_list = parse_int_list(args.esm_pca_list)
    C_list = parse_float_list(args.svr_C_list)
    eps_list = parse_float_list(args.svr_epsilon_list)
    gamma_list: list[str | float] = []
    for x in args.svr_gamma_list.split(","):
        x = x.strip()
        if not x:
            continue
        gamma_list.append(x if x == "scale" else float(x))

    variants = [
        "A1FlatSelected",
        "ESM2PCA",
        "A1FlatSelected_ESM2PCA",
        "A1FlatSelected_ESM2PCAInteract",
    ]

    all_rows: list[dict] = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.replace("seed_", ""))
        set_seed(seed)
        train_df = pd.read_csv(seed_dir / "train_joint.csv")
        valid_df = pd.read_csv(seed_dir / "val_main.csv")
        test_df = pd.read_csv(seed_dir / "test_main.csv")

        y_train = pd.to_numeric(train_df["label_pIC50"], errors="coerce").values.astype(float)
        y_valid = pd.to_numeric(valid_df["label_pIC50"], errors="coerce").values.astype(float)
        y_test = pd.to_numeric(test_df["label_pIC50"], errors="coerce").values.astype(float)
        sample_weight = get_sample_weight(train_df)

        print("=" * 90)
        print(f"[REG] seed={seed}")
        print(f"train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")

        Xtr_a1 = build_a1full_frame(train_df)
        Xva_a1 = build_a1full_frame(valid_df)
        Xte_a1 = build_a1full_frame(test_df)
        Xtr_a1, Xva_a1, Xte_a1 = align_to_train_columns(Xtr_a1, Xva_a1, Xte_a1)
        Xtr_esm = build_esm_frame(train_df, esm_df)
        Xva_esm = build_esm_frame(valid_df, esm_df)
        Xte_esm = build_esm_frame(test_df, esm_df)
        Xtr_esm, Xva_esm, Xte_esm = align_to_train_columns(Xtr_esm, Xva_esm, Xte_esm)

        pred_test = pd.DataFrame({"sequence": test_df["sequence"].values, "y_true": y_test})
        seed_out = out_dir / f"seed_{seed}"
        ensure_dir(seed_out)

        for variant in variants:
            print("-" * 90)
            print(f"Variant: {variant}")
            candidates: list[tuple[float, float, float, dict, np.ndarray, np.ndarray, dict, dict]] = []

            if variant == "A1FlatSelected":
                grid = [(k, None, C, eps, gamma) for k in k_list for C in C_list for eps in eps_list for gamma in gamma_list]
            elif variant == "ESM2PCA":
                grid = [(None, p, C, eps, gamma) for p in pca_list for C in C_list for eps in eps_list for gamma in gamma_list]
            else:
                grid = [(k, p, C, eps, gamma) for k in k_list for p in pca_list for C in C_list for eps in eps_list for gamma in gamma_list]

            for k, pca_dim, C, eps, gamma in grid:
                try:
                    a1_bundle = fit_a1_selector(Xtr_a1, y_train, task="reg", k=int(k)) if k is not None else None
                    esm_bundle = fit_esm_pca(Xtr_esm, pca_dim=int(pca_dim), seed=seed) if pca_dim is not None else None
                    Z_train = make_feature_matrix(
                        variant,
                        Xtr_a1,
                        Xtr_esm,
                        a1_bundle=a1_bundle,
                        esm_bundle=esm_bundle,
                        max_a1_for_interaction=args.max_a1_for_interaction,
                    )
                    Z_valid = make_feature_matrix(
                        variant,
                        Xva_a1,
                        Xva_esm,
                        a1_bundle=a1_bundle,
                        esm_bundle=esm_bundle,
                        max_a1_for_interaction=args.max_a1_for_interaction,
                    )
                    Z_test = make_feature_matrix(
                        variant,
                        Xte_a1,
                        Xte_esm,
                        a1_bundle=a1_bundle,
                        esm_bundle=esm_bundle,
                        max_a1_for_interaction=args.max_a1_for_interaction,
                    )
                    model = fit_svr(Z_train, y_train, sample_weight=sample_weight, C=float(C), epsilon=float(eps), gamma=gamma)
                    valid_pred = model.predict(Z_valid)
                    test_pred = model.predict(Z_test)
                    valid_metrics = evaluate_reg(y_valid, valid_pred)
                    test_metrics = evaluate_reg(y_test, test_pred)
                    params = {
                        "k": k,
                        "pca_dim": pca_dim,
                        "C": C,
                        "epsilon": eps,
                        "gamma": gamma,
                        "max_a1_for_interaction": args.max_a1_for_interaction if "Interact" in variant else None,
                    }
                    candidates.append(
                        (
                            -valid_metrics["rmse"],
                            -valid_metrics["mae"],
                            valid_metrics["spearman"],
                            params,
                            test_pred,
                            Z_train,
                            valid_metrics,
                            test_metrics,
                        )
                    )
                except Exception as e:
                    print(f"[SKIP] {variant} k={k} pca={pca_dim} C={C} eps={eps} gamma={gamma}: {e}")

            if not candidates:
                print(f"[WARN] no candidate succeeded for {variant}")
                continue
            candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            _, _, _, params, test_pred, Z_train_best, valid_metrics, test_metrics = candidates[0]
            model_name = f"{variant}-SVR"
            pred_test[f"{model_name}_pred"] = test_pred
            base = {
                "seed": seed,
                "feature_set": variant,
                "model": model_name,
                "base_model": "SVR",
                "n_features": int(Z_train_best.shape[1]),
                "params": json.dumps(params, ensure_ascii=False),
            }
            all_rows.append({**base, "split": "valid", **valid_metrics})
            all_rows.append({**base, "split": "test", **test_metrics})
            print(
                f"{model_name} | n_features={Z_train_best.shape[1]} | params={params} | "
                f"valid_rmse={valid_metrics['rmse']:.4f} | test_rmse={test_metrics['rmse']:.4f} | "
                f"test_mae={test_metrics['mae']:.4f} | test_spearman={test_metrics['spearman']:.4f}"
            )

        pred_test.to_csv(seed_out / "plm_a1_fusion_reg_test_predictions.csv", index=False, encoding="utf-8-sig")

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "plm_a1_fusion_reg_all_seed_results.csv", index=False, encoding="utf-8-sig")
    summary = summarize_reg(all_df)
    summary.to_csv(out_dir / "plm_a1_fusion_reg_summary.csv", index=False, encoding="utf-8-sig")
    print("=" * 90)
    print(summary)
    print("Saved:", out_dir)
    print(f"Done in {time.time() - t0:.1f}s")


def summarize_reg(all_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["rmse", "mae", "spearman"]
    test_df = all_df[all_df["split"] == "test"].copy()
    rows = []
    for (feature_set, model), g in test_df.groupby(["feature_set", "model"]):
        row = {
            "feature_set": feature_set,
            "model": model,
            "n_seeds": g["seed"].nunique(),
            "n_features_mean": float(g["n_features"].mean()),
            "n_features_std": float(g["n_features"].std(ddof=1)) if len(g) > 1 else 0.0,
        }
        for m in metric_cols:
            row[f"test_{m}_mean"] = float(g[m].mean())
            row[f"test_{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["test_rmse_mean", "test_mae_mean", "test_spearman_mean"], ascending=[True, True, False])
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cls", "reg"], default="cls")

    # Classification paths
    parser.add_argument("--dataset", default="plm4ace_cleaned_2_3")
    parser.add_argument("--cls_split_root", default="data/final/classification/splits")
    parser.add_argument("--cls_esm_dir", default="data/final/classification/features/esm2")

    # Regression paths
    parser.add_argument("--reg_split_root", default="data/final/splits")
    parser.add_argument(
        "--reg_esm_path",
        default="data/final/regression/features/esm2/a1_joint_2_5_esm2_t6_320.csv",
        help="Sequence-level ESM2 csv for regression sequences. Must contain sequence and esm2_* columns.",
    )

    parser.add_argument("--out_root", default="artifacts/plm_informed_a1_fusion")

    # Shared feature-search settings
    parser.add_argument("--a1_k_list", default="50,100,200,400")
    parser.add_argument("--esm_pca_list", default="16,32,64")
    parser.add_argument("--max_a1_for_interaction", type=int, default=50)

    # Classification LR settings
    parser.add_argument("--lr_C_list", default="0.1,1.0,10.0")

    # Regression SVR settings
    parser.add_argument("--svr_C_list", default="1.0,2.0,5.0")
    parser.add_argument("--svr_epsilon_list", default="0.05,0.10,0.20")
    parser.add_argument("--svr_gamma_list", default="scale,0.03")

    args = parser.parse_args()

    if args.task == "cls":
        run_classification(args)
    else:
        run_regression(args)


if __name__ == "__main__":
    main()
