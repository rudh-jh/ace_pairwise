from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

project_root = Path(__file__).resolve().parents[1]
src_root = project_root / "src"

for p in [project_root, src_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ace_pre.baselines.tabular_features import build_descriptor_frame
from ace_pre.baselines.tensor_flat_features import build_a1_flat_feature_frame

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_sequence(seq: object) -> str:
    return str(seq).strip().upper()


def build_seq_identity_frame(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    max_len: int = 5,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        seq = normalize_sequence(row[sequence_col])
        feats = {}

        feats["length"] = float(len(seq))
        for L in range(2, max_len + 1):
            feats[f"len_is_{L}"] = 1.0 if len(seq) == L else 0.0

        for pos in range(max_len):
            aa_at_pos = seq[pos] if pos < len(seq) else "[PAD]"
            feats[f"pos{pos + 1}_is_PAD"] = 1.0 if aa_at_pos == "[PAD]" else 0.0
            for aa in AA_ORDER:
                feats[f"pos{pos + 1}_aa_{aa}"] = 1.0 if aa_at_pos == aa else 0.0

        seq_len = max(1, len(seq))
        for aa in AA_ORDER:
            feats[f"aac_{aa}"] = seq.count(aa) / seq_len

        rows.append(feats)

    return pd.DataFrame(rows, index=df.index)


def build_physchem_only_frame(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    desc = build_descriptor_frame(df, sequence_col=sequence_col)
    drop_prefixes = ("aac_", "nterm_", "cterm_")
    keep_cols = [c for c in desc.columns if not c.startswith(drop_prefixes)]
    return desc[keep_cols].copy()


def build_descriptor_full_frame(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    return build_descriptor_frame(df, sequence_col=sequence_col)


def build_a1flat_full_frame(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    return build_a1_flat_feature_frame(df, sequence_col=sequence_col)


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"{prefix}__{c}" for c in out.columns]
    return out


def build_descriptor_a1flat_frame(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    desc = prefix_columns(build_descriptor_full_frame(df, sequence_col), "desc")
    a1 = prefix_columns(build_a1flat_full_frame(df, sequence_col), "a1")
    return pd.concat([desc, a1], axis=1)


def keep_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def align_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = keep_numeric_frame(X_train)
    X_valid = keep_numeric_frame(X_valid)
    X_test = keep_numeric_frame(X_test)

    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    return X_train, X_valid, X_test


def get_sample_weight(df: pd.DataFrame) -> np.ndarray | None:
    if "sample_weight" not in df.columns:
        return None

    w = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).astype(float).values
    w = np.clip(w, 0.05, 10.0)
    return w


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


def evaluate_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
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


def predict_positive_prob(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-score))

    raise RuntimeError("Model does not support predict_proba or decision_function.")


def fit_lr(X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray | None, seed: int):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["lr__sample_weight"] = sample_weight

    model.fit(X_train, y_train, **fit_kwargs)
    return model, {"C": 1.0, "class_weight": "balanced"}


def fit_svc(X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray | None, seed: int):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    C=2.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["svc__sample_weight"] = sample_weight

    model.fit(X_train, y_train, **fit_kwargs)
    return model, {"C": 2.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}


def fit_rf(X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray | None, seed: int):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["rf__sample_weight"] = sample_weight

    model.fit(X_train, y_train, **fit_kwargs)
    return model, {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
    }


def fit_mlp(X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray | None, seed: int):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=1000,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )

    # MLPClassifier 在不同 sklearn 版本中 sample_weight 支持不一致，这里先不传，避免版本报错。
    model.fit(X_train, y_train)
    return model, {
        "hidden_layer_sizes": [64, 32],
        "alpha": 1e-3,
        "early_stopping": True,
    }


def fit_catboost(X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray | None, seed: int):
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost is not installed.")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=4,
        learning_rate=0.03,
        iterations=500,
        random_seed=seed,
        verbose=False,
        auto_class_weights="Balanced",
    )

    if sample_weight is not None:
        model.fit(X_imp, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_imp, y_train)

    class CatBoostWrapper:
        def __init__(self, imputer, model):
            self.imputer = imputer
            self.model = model

        def predict_proba(self, X):
            X_imp2 = self.imputer.transform(X)
            return self.model.predict_proba(X_imp2)

    return CatBoostWrapper(imputer, model), {
        "depth": 4,
        "learning_rate": 0.03,
        "iterations": 500,
        "auto_class_weights": "Balanced",
    }


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

    for (feature_set, model), g in test_df.groupby(["feature_set", "model"]):
        row = {
            "feature_set": feature_set,
            "model": model,
            "n_seeds": g["seed"].nunique(),
            "n_features_mean": float(g["n_features"].mean()),
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
    feature_builders: dict,
    model_builders: dict,
) -> pd.DataFrame:
    seed = int(seed_dir.name.replace("seed_", ""))
    set_seed(seed)

    train_df = pd.read_csv(seed_dir / "train.csv")
    valid_df = pd.read_csv(seed_dir / "valid.csv")
    test_df = pd.read_csv(seed_dir / "test.csv")

    y_train = train_df["cls_label"].astype(int).values
    y_valid = valid_df["cls_label"].astype(int).values
    y_test = test_df["cls_label"].astype(int).values

    sample_weight = get_sample_weight(train_df)

    print("=" * 80)
    print(f"dataset={dataset} | seed={seed}")
    print(f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    print("train label:", train_df["cls_label"].value_counts().to_dict())

    all_rows = []

    pred_valid = pd.DataFrame(
        {
            "sequence": valid_df["sequence"].values,
            "y_true": y_valid,
        }
    )
    pred_test = pd.DataFrame(
        {
            "sequence": test_df["sequence"].values,
            "y_true": y_test,
        }
    )

    for feature_name, feature_builder in feature_builders.items():
        print("-" * 80)
        print(f"Building feature set: {feature_name}")

        X_train = feature_builder(train_df, sequence_col="sequence")
        X_valid = feature_builder(valid_df, sequence_col="sequence")
        X_test = feature_builder(test_df, sequence_col="sequence")

        X_train, X_valid, X_test = align_features(X_train, X_valid, X_test)
        n_features = int(X_train.shape[1])

        print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}")

        for model_short_name, model_builder in model_builders.items():
            model_name = f"{feature_name}-{model_short_name}"
            print(f"Training {model_name}...")

            try:
                model, params = model_builder(X_train, y_train, sample_weight, seed)
            except Exception as e:
                print(f"[SKIP] {model_name}: {e}")
                continue

            valid_prob = predict_positive_prob(model, X_valid)
            test_prob = predict_positive_prob(model, X_test)

            threshold = best_threshold_by_mcc(y_valid, valid_prob)

            valid_metrics = evaluate_binary(y_valid, valid_prob, threshold)
            test_metrics = evaluate_binary(y_test, test_prob, threshold)

            all_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "feature_set": feature_name,
                    "model": model_name,
                    "base_model": model_short_name,
                    "split": "valid",
                    "n_features": n_features,
                    **valid_metrics,
                    "params": json.dumps(params, ensure_ascii=False),
                }
            )

            all_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "feature_set": feature_name,
                    "model": model_name,
                    "base_model": model_short_name,
                    "split": "test",
                    "n_features": n_features,
                    **test_metrics,
                    "params": json.dumps(params, ensure_ascii=False),
                }
            )

            pred_valid[f"{model_name}_prob"] = valid_prob
            pred_valid[f"{model_name}_pred"] = (valid_prob >= threshold).astype(int)
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

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(seed_out / "cls_grid_results.csv", index=False, encoding="utf-8-sig")
    pred_valid.to_csv(seed_out / "cls_grid_valid_predictions.csv", index=False, encoding="utf-8-sig")
    pred_test.to_csv(seed_out / "cls_grid_test_predictions.csv", index=False, encoding="utf-8-sig")

    return result_df


def main() -> None:
    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="acepair_cls_2_3",
        # choices=["acepair_cls_2_3", "acepair_cls_2_5"],
    )
    parser.add_argument(
        "--split_root",
        default="data/final/classification/splits",
    )
    parser.add_argument(
        "--out_root",
        default="artifacts/classification/feature_model_grid",
    )
    args = parser.parse_args()

    split_root = project_root / args.split_root / args.dataset
    out_dir = project_root / args.out_root / args.dataset
    ensure_dir(out_dir)

    seed_dirs = sorted([p for p in split_root.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed dirs found under: {split_root}")

    feature_builders = {
        "SeqOnly": build_seq_identity_frame,
        "PhysChemOnly": build_physchem_only_frame,
        "Descriptor": build_descriptor_full_frame,
        "A1FlatFull": build_a1flat_full_frame,
        "Descriptor_A1FlatFull": build_descriptor_a1flat_frame,
    }

    model_builders = {
        "LR": fit_lr,
        "SVC": fit_svc,
        "RF": fit_rf,
        "MLP": fit_mlp,
    }

    if HAS_CATBOOST:
        model_builders["CatBoost"] = fit_catboost
    else:
        print("CatBoost is not installed. CatBoost models will be skipped.")

    all_results = []

    for seed_dir in seed_dirs:
        one = run_one_seed(
            dataset=args.dataset,
            seed_dir=seed_dir,
            out_dir=out_dir,
            feature_builders=feature_builders,
            model_builders=model_builders,
        )
        all_results.append(one)

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(out_dir / "cls_grid_all_seed_results.csv", index=False, encoding="utf-8-sig")

    summary = summarize_results(all_df)
    summary.to_csv(out_dir / "cls_grid_summary.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Summary:")
    print(summary)
    print("=" * 80)
    print(f"Done in {time.time() - t0:.1f}s")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()