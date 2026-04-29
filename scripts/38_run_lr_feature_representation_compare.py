from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in [PROJECT_ROOT, SRC_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ace_pre.baselines.tensor_flat_features import build_a1_flat_feature_frame


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_sequence(seq: object) -> str:
    return str(seq).strip().upper()


def get_sample_weight(df: pd.DataFrame) -> np.ndarray | None:
    if "sample_weight" not in df.columns:
        return None
    w = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).astype(float).values
    return np.clip(w, 0.05, 10.0)


def load_esm_features(path: Path) -> pd.DataFrame:
    esm_df = pd.read_csv(path)
    if "sequence" not in esm_df.columns:
        raise ValueError(f"ESM2 feature file must contain a sequence column: {path}")
    esm_df["sequence"] = esm_df["sequence"].map(normalize_sequence)
    return esm_df


def build_esm2_frame(df: pd.DataFrame, esm_df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["sequence"]].copy()
    tmp["sequence"] = tmp["sequence"].map(normalize_sequence)

    merged = tmp.merge(esm_df, on="sequence", how="left")
    feature_cols = [c for c in merged.columns if c.startswith("esm2_")]
    if not feature_cols:
        raise ValueError("No columns starting with esm2_ found in ESM feature file.")

    missing = merged[feature_cols].isna().any(axis=1).sum()
    if missing > 0:
        examples = merged.loc[merged[feature_cols].isna().any(axis=1), "sequence"].head(10).tolist()
        raise ValueError(f"Missing ESM2 embeddings for {missing} rows. Examples: {examples}")

    X = merged[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def build_a1flat_full_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = build_a1_flat_feature_frame(df, sequence_col="sequence")
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def align_to_train_columns(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    return X_train, X_valid, X_test


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


def make_lr(seed: int, C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )


def fit_lr_full(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    C: float = 1.0,
) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", make_lr(seed=seed, C=C)),
        ]
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["lr__sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def fit_lr_selected(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    k: int,
    C: float = 1.0,
) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("variance", VarianceThreshold(threshold=1e-10)),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("lr", make_lr(seed=seed, C=C)),
        ]
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["lr__sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def predict_positive_prob(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-score))
    raise RuntimeError("Model has neither predict_proba nor decision_function.")


def choose_best_lr_by_valid_mcc(
    feature_name: str,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    select_k_list: list[int],
    C_list: list[float],
) -> tuple[object, dict, np.ndarray, dict]:
    candidates: list[tuple[float, float, float, object, dict, np.ndarray, dict]] = []

    if feature_name == "A1FlatSelected":
        for k0 in select_k_list:
            for C in C_list:
                try:
                    model = fit_lr_selected(
                        X_train=X_train,
                        y_train=y_train,
                        sample_weight=sample_weight,
                        seed=seed,
                        k=k0,
                        C=C,
                    )
                    valid_prob = predict_positive_prob(model, X_valid)
                    threshold = best_threshold_by_mcc(y_valid, valid_prob)
                    valid_metrics = evaluate_binary(y_valid, valid_prob, threshold)
                    params = {
                        "C": C,
                        "class_weight": "balanced",
                        "selector": "VarianceThreshold + SelectKBest(f_classif)",
                        "k": k0,
                    }
                    candidates.append(
                        (
                            valid_metrics["mcc"],
                            valid_metrics["auc"],
                            valid_metrics["bacc"],
                            model,
                            params,
                            valid_prob,
                            valid_metrics,
                        )
                    )
                except Exception as e:
                    print(f"[SKIP] {feature_name} k={k0}, C={C}: {e}")
    else:
        for C in C_list:
            model = fit_lr_full(
                X_train=X_train,
                y_train=y_train,
                sample_weight=sample_weight,
                seed=seed,
                C=C,
            )
            valid_prob = predict_positive_prob(model, X_valid)
            threshold = best_threshold_by_mcc(y_valid, valid_prob)
            valid_metrics = evaluate_binary(y_valid, valid_prob, threshold)
            params = {
                "C": C,
                "class_weight": "balanced",
                "selector": "none",
                "k": None,
            }
            candidates.append(
                (
                    valid_metrics["mcc"],
                    valid_metrics["auc"],
                    valid_metrics["bacc"],
                    model,
                    params,
                    valid_prob,
                    valid_metrics,
                )
            )

    if not candidates:
        raise RuntimeError(f"No valid LR candidate for feature set: {feature_name}")

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best_mcc, best_auc, best_bacc, best_model, best_params, best_valid_prob, best_valid_metrics = candidates[0]
    return best_model, best_params, best_valid_prob, best_valid_metrics


def save_selected_feature_names(model: Pipeline, X_train: pd.DataFrame, out_path: Path) -> None:
    if "variance" not in model.named_steps or "select" not in model.named_steps:
        return

    variance: VarianceThreshold = model.named_steps["variance"]
    selector: SelectKBest = model.named_steps["select"]

    cols_after_var = np.array(X_train.columns)[variance.get_support()]
    selected_cols = cols_after_var[selector.get_support()]

    score_df = pd.DataFrame(
        {
            "feature": cols_after_var,
            "score": selector.scores_,
            "pvalue": getattr(selector, "pvalues_", np.full(len(cols_after_var), np.nan)),
            "selected": selector.get_support(),
        }
    ).sort_values(["selected", "score"], ascending=[False, False])

    selected_df = pd.DataFrame({"selected_feature": selected_cols})
    ensure_dir(out_path.parent)
    score_df.to_csv(out_path.with_name(out_path.stem + "_scores.csv"), index=False, encoding="utf-8-sig")
    selected_df.to_csv(out_path, index=False, encoding="utf-8-sig")


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
            "n_features_std": float(g["n_features"].std(ddof=1)) if len(g) > 1 else 0.0,
        }
        vals = pd.to_numeric(g.get("selected_k", pd.Series(dtype=float)), errors="coerce").dropna()
        row["selected_k_mode"] = int(vals.mode().iloc[0]) if len(vals) else np.nan
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
    esm_df: pd.DataFrame,
    out_dir: Path,
    select_k_list: list[int],
    C_list: list[float],
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

    print("=" * 90)
    print(f"dataset={dataset} | seed={seed}")
    print(f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    print("train label:", train_df["cls_label"].value_counts().to_dict())

    feature_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    print("Building ESM2 features...")
    Xtr_esm = build_esm2_frame(train_df, esm_df)
    Xva_esm = build_esm2_frame(valid_df, esm_df)
    Xte_esm = build_esm2_frame(test_df, esm_df)
    feature_frames["ESM2"] = align_to_train_columns(Xtr_esm, Xva_esm, Xte_esm)

    print("Building A1FlatFull features...")
    Xtr_a1 = build_a1flat_full_frame(train_df)
    Xva_a1 = build_a1flat_full_frame(valid_df)
    Xte_a1 = build_a1flat_full_frame(test_df)
    Xtr_a1, Xva_a1, Xte_a1 = align_to_train_columns(Xtr_a1, Xva_a1, Xte_a1)
    feature_frames["A1FlatFull"] = (Xtr_a1, Xva_a1, Xte_a1)
    feature_frames["A1FlatSelected"] = (Xtr_a1, Xva_a1, Xte_a1)

    rows = []
    pred_valid = pd.DataFrame({"sequence": valid_df["sequence"].values, "y_true": y_valid})
    pred_test = pd.DataFrame({"sequence": test_df["sequence"].values, "y_true": y_test})

    seed_out = out_dir / f"seed_{seed}"
    ensure_dir(seed_out)

    for feature_set in ["ESM2", "A1FlatFull", "A1FlatSelected"]:
        print("-" * 90)
        print(f"Feature representation: {feature_set}")

        X_train, X_valid, X_test = feature_frames[feature_set]
        print("X:", X_train.shape, X_valid.shape, X_test.shape)

        model, params, valid_prob, valid_metrics = choose_best_lr_by_valid_mcc(
            feature_name=feature_set,
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            sample_weight=sample_weight,
            seed=seed,
            select_k_list=select_k_list,
            C_list=C_list,
        )

        threshold = float(valid_metrics["threshold"])
        test_prob = predict_positive_prob(model, X_test)
        test_metrics = evaluate_binary(y_test, test_prob, threshold)

        selected_k = params.get("k", None)
        n_features_used = int(selected_k) if selected_k is not None else int(X_train.shape[1])

        model_name = f"{feature_set}-LR"
        base = {
            "dataset": dataset,
            "seed": seed,
            "feature_set": feature_set,
            "model": model_name,
            "base_model": "LR",
            "n_features": n_features_used,
            "raw_n_features": int(X_train.shape[1]),
            "selected_k": selected_k,
            "params": json.dumps(params, ensure_ascii=False),
        }

        rows.append({**base, "split": "valid", **valid_metrics})
        rows.append({**base, "split": "test", **test_metrics})

        pred_valid[f"{model_name}_prob"] = valid_prob
        pred_valid[f"{model_name}_pred"] = (valid_prob >= threshold).astype(int)
        pred_test[f"{model_name}_prob"] = test_prob
        pred_test[f"{model_name}_pred"] = (test_prob >= threshold).astype(int)

        if feature_set == "A1FlatSelected":
            save_selected_feature_names(
                model,
                X_train,
                seed_out / "A1FlatSelected_LR_selected_features.csv",
            )

        print(
            f"{model_name} | selected_k={selected_k} | threshold={threshold:.2f} | "
            f"valid_mcc={valid_metrics['mcc']:.4f} | test_mcc={test_metrics['mcc']:.4f} | "
            f"test_auc={test_metrics['auc']:.4f} | test_bacc={test_metrics['bacc']:.4f}"
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(seed_out / "lr_feature_representation_results.csv", index=False, encoding="utf-8-sig")
    pred_valid.to_csv(seed_out / "lr_feature_representation_valid_predictions.csv", index=False, encoding="utf-8-sig")
    pred_test.to_csv(seed_out / "lr_feature_representation_test_predictions.csv", index=False, encoding="utf-8-sig")

    return result_df


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    t0 = time.time()

    parser = argparse.ArgumentParser(
        description=(
            "Compare three feature representations with the same LR classifier: "
            "ESM2-LR, A1FlatFull-LR, A1FlatSelected-LR."
        )
    )
    parser.add_argument(
        "--dataset",
        default="plm4ace_cleaned_2_3",
        help=(
            "Classification dataset name under data/final/classification/splits. "
            "Examples: plm4ace_cleaned_2_3, plm4ace_cleaned_2_5, "
            "acepair_cls_2_3, acepair_cls_2_5"
        ),
    )
    parser.add_argument("--split_root", default="data/final/classification/splits")
    parser.add_argument("--esm_dir", default="data/final/classification/features/esm2")
    parser.add_argument("--out_root", default="artifacts/classification/lr_feature_representation_compare")
    parser.add_argument("--select_k_list", default="50,100,200,400")
    parser.add_argument("--C_list", default="0.1,1.0,10.0")
    args = parser.parse_args()

    split_root = PROJECT_ROOT / args.split_root / args.dataset
    esm_path = PROJECT_ROOT / args.esm_dir / f"{args.dataset}_esm2_t6_320.csv"
    out_dir = PROJECT_ROOT / args.out_root / args.dataset
    ensure_dir(out_dir)

    if not split_root.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_root}\n"
            f"Please run scripts/31_make_cls_splits.py for this dataset first."
        )
    if not esm_path.exists():
        raise FileNotFoundError(
            f"ESM2 feature file not found: {esm_path}\n"
            f"Please run scripts/33_extract_esm2_cls_embeddings.py --dataset {args.dataset} first."
        )

    seed_dirs = sorted([p for p in split_root.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed dirs found under: {split_root}")

    select_k_list = parse_int_list(args.select_k_list)
    C_list = parse_float_list(args.C_list)

    print("=" * 90)
    print("LR feature representation comparison")
    print("dataset:", args.dataset)
    print("split_root:", split_root)
    print("esm_path:", esm_path)
    print("out_dir:", out_dir)
    print("select_k_list:", select_k_list)
    print("C_list:", C_list)

    esm_df = load_esm_features(esm_path)

    all_results = []
    for seed_dir in seed_dirs:
        one = run_one_seed(
            dataset=args.dataset,
            seed_dir=seed_dir,
            esm_df=esm_df,
            out_dir=out_dir,
            select_k_list=select_k_list,
            C_list=C_list,
        )
        all_results.append(one)

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(
        out_dir / "lr_feature_representation_all_seed_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_results(all_df)
    summary.to_csv(
        out_dir / "lr_feature_representation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 90)
    print("Summary:")
    print(summary)
    print("=" * 90)
    print(f"Done in {time.time() - t0:.1f}s")
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
