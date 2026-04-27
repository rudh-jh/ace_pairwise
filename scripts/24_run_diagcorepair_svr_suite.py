from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.svm import SVR

# =============================================================================
# Project path
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for p in [PROJECT_ROOT, SRC_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ace_pre.baselines.tensor_flat_features import build_a1_flat_feature_frame

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Basic settings
# =============================================================================

SEEDS = [42, 52, 62]
TOP_KS = [5, 10, 20]

OUT_DIR = PROJECT_ROOT / "artifacts" / "diagcorepair_svr_suite"

SVR_PARAM_GRID = [
    {"C": C, "epsilon": eps, "gamma": gamma}
    for C in [1.0, 2.0, 5.0, 10.0]
    for eps in [0.05, 0.10, 0.20]
    for gamma in ["scale", 0.01, 0.03]
]

SELECT_K_LIST = [50, 100, 200, 400]

STRONG_ACTIVITY_THRESHOLD = 6.0
STRONG_WEIGHT_ALPHA_LIST = [0.5, 1.0, 2.0]


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    val = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(val) if pd.notna(val) else 0.0


def topk_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    if len(y_true) < k:
        return np.nan

    true_top_idx = set(np.argsort(-y_true)[:k].tolist())
    pred_top_idx = set(np.argsort(-y_pred)[:k].tolist())

    return float(len(true_top_idx & pred_top_idx) / k)


def strong_activity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_true >= STRONG_ACTIVITY_THRESHOLD

    if mask.sum() == 0:
        return {
            "strong_n": 0,
            "strong_rmse": np.nan,
            "strong_mae": np.nan,
            "strong_bias": np.nan,
        }

    return {
        "strong_n": int(mask.sum()),
        "strong_rmse": rmse(y_true[mask], y_pred[mask]),
        "strong_mae": mae(y_true[mask], y_pred[mask]),
        "strong_bias": float(np.mean(y_pred[mask] - y_true[mask])),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }

    for k in TOP_KS:
        out[f"top{k}_hit_rate"] = topk_hit_rate(y_true, y_pred, k)

    out.update(strong_activity_metrics(y_true, y_pred))
    return out


def val_score(y_true: np.ndarray, y_pred: np.ndarray, mode: str) -> float:
    """
    score 越小越好。

    rmse:
        只按验证集 RMSE 选参。

    rank_aware:
        同时考虑 RMSE、Spearman 和 Top10。
        不是最终评价指标，只是验证集选参标准。
    """
    val_rmse = rmse(y_true, y_pred)

    if mode == "rmse":
        return val_rmse

    if mode == "rank_aware":
        val_spearman = spearman(y_true, y_pred)
        val_top10 = topk_hit_rate(y_true, y_pred, 10)

        if np.isnan(val_top10):
            val_top10 = 0.0

        return val_rmse - 0.15 * val_spearman - 0.15 * val_top10

    raise ValueError(f"Unknown score mode: {mode}")


def get_label_and_weight(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "label_pIC50" in df.columns:
        y = df["label_pIC50"].astype(float).values
    elif "pic50" in df.columns:
        y = df["pic50"].astype(float).values
    elif "pIC50" in df.columns:
        y = df["pIC50"].astype(float).values
    else:
        raise KeyError("Cannot find label column. Expected label_pIC50 / pic50 / pIC50.")

    if "sample_weight" in df.columns:
        w = df["sample_weight"].astype(float).values
    else:
        w = np.ones_like(y, dtype=float)

    return y, w


def make_strong_tail_weight(
    y_train: np.ndarray,
    base_weight: np.ndarray,
    alpha: float,
    threshold: float = STRONG_ACTIVITY_THRESHOLD,
) -> np.ndarray:
    """
    对强活性样本加权。

    alpha=1.0 表示强活性样本权重变为 base_weight * 2。
    alpha=2.0 表示强活性样本权重变为 base_weight * 3。
    """
    w = base_weight.astype(float).copy()
    strong_mask = y_train >= threshold
    w[strong_mask] = w[strong_mask] * (1.0 + alpha)
    return w


# =============================================================================
# Feature building
# =============================================================================

def align_feature_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = build_a1_flat_feature_frame(train_df, sequence_col="sequence")
    X_val = build_a1_flat_feature_frame(val_df, sequence_col="sequence")
    X_test = build_a1_flat_feature_frame(test_df, sequence_col="sequence")

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    return X_train, X_val, X_test


def infer_feature_groups(feature_names: list[str]) -> dict:
    """
    根据 A1Flat 特征名划分通道组。

    特征名示例：
        diag_hydrophobicity__r0__c0
        pair_hydrophobicity_sum__r0__c1
        heur_c_terminal_favorable_flag__r0__c1
        seq_length
        residue_id_0
    """
    diag_cols = []
    pair_cols = []
    heur_cols = []

    core_pair_cols = []
    terminal_pair_cols = []
    order_pair_cols = []
    other_pair_cols = []

    other_cols = []

    core_pair_markers = [
        "pair_hydrophobicity_sum",
        "pair_hydrophobicity_abs_diff",
        "pair_hydrophobicity_signed_diff",
        "pair_charge_product",
        "pair_charge_signed_diff",
        "pair_polarity_abs_diff",
        "pair_volume_abs_diff",
        "pair_both_hydrophobic_flag",
        "pair_both_bulky_flag",
        "pair_aromatic_pair_flag",
        "pair_donor_acceptor_compatibility",
        "pair_substitution_similarity_proxy",
        "pair_residue_class_compatibility",
    ]

    terminal_pair_markers = [
        "pair_terminal_interaction_flag",
        "pair_end_to_end_flag",
        "pair_end_to_internal_flag",
    ]

    order_pair_markers = [
        "pair_forward_order_flag",
        "pair_reverse_order_flag",
    ]

    for col in feature_names:
        lower = col.lower()

        if lower.startswith("diag_"):
            diag_cols.append(col)
            continue

        if lower.startswith("heur_"):
            heur_cols.append(col)
            continue

        if lower.startswith("pair_"):
            pair_cols.append(col)

            if any(marker in lower for marker in core_pair_markers):
                core_pair_cols.append(col)
            elif any(marker in lower for marker in terminal_pair_markers):
                terminal_pair_cols.append(col)
            elif any(marker in lower for marker in order_pair_markers):
                order_pair_cols.append(col)
            else:
                other_pair_cols.append(col)

            continue

        other_cols.append(col)

    all_cols = list(feature_names)

    diag_core_pair_cols = sorted(
        set(diag_cols + core_pair_cols),
        key=all_cols.index,
    )

    diag_core_terminal_pair_cols = sorted(
        set(diag_cols + core_pair_cols + terminal_pair_cols),
        key=all_cols.index,
    )

    diag_all_pair_cols = sorted(
        set(diag_cols + pair_cols),
        key=all_cols.index,
    )

    no_heur_cols = [c for c in all_cols if c not in set(heur_cols)]
    no_pair_cols = [c for c in all_cols if c not in set(pair_cols)]

    feature_sets = {
        "A1FlatFull": all_cols,
        "A1FlatDiagOnly": diag_cols,
        "A1FlatPairOnly": pair_cols,
        "A1FlatNoHeur": no_heur_cols,
        "A1FlatNoPairwise": no_pair_cols,
        "A1FlatDiagCorePair": diag_core_pair_cols,
        "A1FlatDiagCoreTerminalPair": diag_core_terminal_pair_cols,
        "A1FlatDiagAllPair": diag_all_pair_cols,
    }

    group_parts = {
        "diag": diag_cols,
        "pair": pair_cols,
        "core_pair": core_pair_cols,
        "terminal_pair": terminal_pair_cols,
        "order_pair": order_pair_cols,
        "other_pair": other_pair_cols,
        "heur": heur_cols,
        "other": other_cols,
    }

    return {
        "feature_sets": feature_sets,
        "group_parts": group_parts,
    }


def standardize_from_train(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if len(cols) == 0:
        raise ValueError("Empty feature columns.")

    X_train = X_train_df[cols].copy()
    X_val = X_val_df[cols].copy()
    X_test = X_test_df[cols].copy()

    med = X_train.median(axis=0)

    X_train = X_train.fillna(med)
    X_val = X_val.fillna(med)
    X_test = X_test.fillna(med)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).replace(0, 1.0)

    X_train_np = ((X_train - mean) / std).values.astype(np.float32)
    X_val_np = ((X_val - mean) / std).values.astype(np.float32)
    X_test_np = ((X_test - mean) / std).values.astype(np.float32)

    return X_train_np, X_val_np, X_test_np, cols


def standardize_and_select(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: np.ndarray,
    cols: list[str],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_train, X_val, X_test, used_cols = standardize_from_train(
        X_train_df,
        X_val_df,
        X_test_df,
        cols,
    )

    vt = VarianceThreshold(threshold=1e-10)
    X_train_v = vt.fit_transform(X_train)
    X_val_v = vt.transform(X_val)
    X_test_v = vt.transform(X_test)

    kept_after_vt = [c for c, keep in zip(used_cols, vt.get_support()) if keep]

    if X_train_v.shape[1] == 0:
        raise ValueError("No feature left after VarianceThreshold.")

    real_k = min(k, X_train_v.shape[1])

    if real_k < X_train_v.shape[1]:
        selector = SelectKBest(score_func=f_regression, k=real_k)
        X_train_s = selector.fit_transform(X_train_v, y_train)
        X_val_s = selector.transform(X_val_v)
        X_test_s = selector.transform(X_test_v)

        selected_cols = [
            c for c, keep in zip(kept_after_vt, selector.get_support()) if keep
        ]
    else:
        X_train_s = X_train_v
        X_val_s = X_val_v
        X_test_s = X_test_v
        selected_cols = kept_after_vt

    return X_train_s, X_val_s, X_test_s, selected_cols


# =============================================================================
# SVR fitting
# =============================================================================

def fit_best_svr(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    sample_weight: np.ndarray,
    param_grid: list[dict],
    score_mode: str,
) -> dict:
    best = None

    for params in param_grid:
        model = SVR(
            kernel="rbf",
            C=float(params["C"]),
            epsilon=float(params["epsilon"]),
            gamma=params["gamma"],
        )

        model.fit(X_train, y_train, sample_weight=sample_weight)

        val_pred = model.predict(X_val)
        score = val_score(y_val, val_pred, mode=score_mode)

        if best is None or score < best["score"]:
            test_pred = model.predict(X_test)

            best = {
                "model": model,
                "params": params,
                "score": score,
                "val_pred": val_pred,
                "test_pred": test_pred,
                "val_metrics": evaluate_predictions(y_val, val_pred),
                "test_metrics": evaluate_predictions(y_test, test_pred),
            }

    return best


# =============================================================================
# Saving and summary
# =============================================================================

def save_predictions(
    path: Path,
    seed: int,
    model_name: str,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    out = pd.DataFrame(
        {
            "seed": seed,
            "model": model_name,
            "row_index": np.arange(len(test_df)),
        }
    )

    for col in [
        "sample_id",
        "sequence",
        "length",
        "label_pIC50",
        "pic50",
        "pIC50",
        "sample_weight",
        "source_role",
    ]:
        if col in test_df.columns:
            out[col] = test_df[col].values

    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["error"] = y_pred - y_true
    out["abs_error"] = np.abs(y_pred - y_true)

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def result_row(
    seed: int,
    step: str,
    model_name: str,
    feature_set: str,
    n_features: int,
    best: dict,
    score_mode: str,
    extra: dict | None = None,
) -> dict:
    extra = extra or {}

    row = {
        "seed": seed,
        "step": step,
        "model": model_name,
        "feature_set": feature_set,
        "n_features": n_features,
        "score_mode": score_mode,
        "val_selection_score": best["score"],
        "params": json.dumps(best["params"], ensure_ascii=False),
    }

    for prefix, metrics in [
        ("val", best["val_metrics"]),
        ("test", best["test_metrics"]),
    ]:
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = v

    row.update(extra)
    return row


def summarize_results(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    输出两张 summary：

    1. summary_by_model:
       只按 model 聚合，适合看最终模型整体表现。
       这样 A1FlatSelected-TunedSVR 不会因为不同 seed 选到不同 k 被拆开。

    2. summary_detail:
       按 model + step + feature_set 聚合，适合看具体实验来源。
    """
    agg_dict = dict(
        n_runs=("seed", "count"),
        seeds=("seed", lambda x: ",".join(map(str, sorted(x.unique())))),

        n_features_min=("n_features", "min"),
        n_features_max=("n_features", "max"),

        test_rmse_mean=("test_rmse", "mean"),
        test_rmse_std=("test_rmse", "std"),

        test_mae_mean=("test_mae", "mean"),
        test_mae_std=("test_mae", "std"),

        test_spearman_mean=("test_spearman", "mean"),
        test_spearman_std=("test_spearman", "std"),

        test_top5_hit_rate_mean=("test_top5_hit_rate", "mean"),
        test_top10_hit_rate_mean=("test_top10_hit_rate", "mean"),
        test_top20_hit_rate_mean=("test_top20_hit_rate", "mean"),

        test_strong_n_mean=("test_strong_n", "mean"),
        test_strong_mae_mean=("test_strong_mae", "mean"),
        test_strong_rmse_mean=("test_strong_rmse", "mean"),
        test_strong_bias_mean=("test_strong_bias", "mean"),

        val_rmse_mean=("val_rmse", "mean"),
        val_spearman_mean=("val_spearman", "mean"),
        val_top10_hit_rate_mean=("val_top10_hit_rate", "mean"),
    )

    summary_by_model = (
        raw_df
        .groupby(["model"], as_index=False)
        .agg(**agg_dict)
        .sort_values(["test_rmse_mean", "test_mae_mean"], ascending=[True, True])
    )

    summary_detail = (
        raw_df
        .groupby(["model", "step", "feature_set", "score_mode"], as_index=False)
        .agg(**agg_dict)
        .sort_values(["test_rmse_mean", "test_mae_mean"], ascending=[True, True])
    )

    return summary_by_model, summary_detail


# =============================================================================
# Experiment helpers
# =============================================================================

def run_plain_svr_experiment(
    *,
    rows: list[dict],
    seed: int,
    seed_out_dir: Path,
    test_df: pd.DataFrame,
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    feature_set_name: str,
    cols: list[str],
    model_name: str,
    step: str,
    score_mode: str = "rmse",
    extra: dict | None = None,
) -> None:
    print(f"\n[{step}] {model_name}")

    X_train, X_val, X_test, used_cols = standardize_from_train(
        X_train_df,
        X_val_df,
        X_test_df,
        cols,
    )

    best = fit_best_svr(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        sample_weight=w_train,
        param_grid=SVR_PARAM_GRID,
        score_mode=score_mode,
    )

    save_predictions(
        path=seed_out_dir / "predictions" / f"{model_name}.csv",
        seed=seed,
        model_name=model_name,
        test_df=test_df,
        y_true=y_test,
        y_pred=best["test_pred"],
    )

    rows.append(
        result_row(
            seed=seed,
            step=step,
            model_name=model_name,
            feature_set=feature_set_name,
            n_features=len(used_cols),
            best=best,
            score_mode=score_mode,
            extra=extra,
        )
    )


def run_selected_svr_experiment(
    *,
    rows: list[dict],
    seed: int,
    seed_out_dir: Path,
    test_df: pd.DataFrame,
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    base_feature_set_name: str,
    cols: list[str],
    k: int,
    model_name: str,
    step: str,
    score_mode: str,
    extra: dict | None = None,
) -> dict:
    print(f"\n[{step}] {model_name} | k={k} | score_mode={score_mode}")

    X_train, X_val, X_test, selected_cols = standardize_and_select(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        y_train=y_train,
        cols=cols,
        k=k,
    )

    best = fit_best_svr(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        sample_weight=w_train,
        param_grid=SVR_PARAM_GRID,
        score_mode=score_mode,
    )

    save_predictions(
        path=seed_out_dir / "predictions" / f"{model_name}.csv",
        seed=seed,
        model_name=model_name,
        test_df=test_df,
        y_true=y_test,
        y_pred=best["test_pred"],
    )

    selected_info = {
        "seed": seed,
        "model": model_name,
        "base_feature_set": base_feature_set_name,
        "selected_k": k,
        "selected_n_features": len(selected_cols),
        "selected_cols": selected_cols,
    }

    with open(
        seed_out_dir / "selected_features" / f"{model_name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(selected_info, f, ensure_ascii=False, indent=2)

    row_extra = {
        "selected_k": k,
        "base_feature_set": base_feature_set_name,
    }

    if extra:
        row_extra.update(extra)

    rows.append(
        result_row(
            seed=seed,
            step=step,
            model_name=model_name,
            feature_set=f"{base_feature_set_name}_SelectedK{k}",
            n_features=len(selected_cols),
            best=best,
            score_mode=score_mode,
            extra=row_extra,
        )
    )

    return {
        "best": best,
        "selected_cols": selected_cols,
    }


def run_adaptive_selected_svr_experiment(
    *,
    rows: list[dict],
    seed: int,
    seed_out_dir: Path,
    test_df: pd.DataFrame,
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    base_feature_set_name: str,
    cols: list[str],
    model_name: str,
    step: str,
    score_mode: str,
) -> None:
    """
    在多个 K 里面根据验证集 score 自动选最优 K。
    用于复现你原来的 A1FlatSelected-TunedSVR 思路。
    """
    print(f"\n[{step}] {model_name} | adaptive K | score_mode={score_mode}")

    best_global = None
    best_info = None

    for k in SELECT_K_LIST:
        X_train, X_val, X_test, selected_cols = standardize_and_select(
            X_train_df=X_train_df,
            X_val_df=X_val_df,
            X_test_df=X_test_df,
            y_train=y_train,
            cols=cols,
            k=k,
        )

        best_k = fit_best_svr(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            sample_weight=w_train,
            param_grid=SVR_PARAM_GRID,
            score_mode=score_mode,
        )

        if best_global is None or best_k["score"] < best_global["score"]:
            best_global = best_k
            best_info = {
                "selected_k": k,
                "selected_n_features": len(selected_cols),
                "selected_cols": selected_cols,
            }

    assert best_global is not None
    assert best_info is not None

    save_predictions(
        path=seed_out_dir / "predictions" / f"{model_name}.csv",
        seed=seed,
        model_name=model_name,
        test_df=test_df,
        y_true=y_test,
        y_pred=best_global["test_pred"],
    )

    with open(
        seed_out_dir / "selected_features" / f"{model_name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    rows.append(
        result_row(
            seed=seed,
            step=step,
            model_name=model_name,
            feature_set=f"{base_feature_set_name}_AdaptiveSelected",
            n_features=best_info["selected_n_features"],
            best=best_global,
            score_mode=score_mode,
            extra={
                "selected_k": best_info["selected_k"],
                "base_feature_set": base_feature_set_name,
            },
        )
    )


# =============================================================================
# Main per-seed experiment
# =============================================================================

def run_for_seed(seed: int, out_dir: Path) -> list[dict]:
    print("\n" + "=" * 100)
    print(f"Running DiagCorePair SVR suite | seed={seed}")
    print("=" * 100)

    split_dir = PROJECT_ROOT / "data" / "final" / "splits" / f"seed_{seed}"

    train_path = split_dir / "train_joint.csv"
    val_path = split_dir / "val_main.csv"
    test_path = split_dir / "test_main.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing split file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing split file: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing split file: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    y_train, w_train = get_label_and_weight(train_df)
    y_val, _ = get_label_and_weight(val_df)
    y_test, _ = get_label_and_weight(test_df)

    print(f"Train: {train_df.shape}")
    print(f"Val  : {val_df.shape}")
    print(f"Test : {test_df.shape}")

    print("\nBuilding A1Flat feature frames...")
    X_train_df, X_val_df, X_test_df = align_feature_frames(train_df, val_df, test_df)

    feature_names = list(X_train_df.columns)
    group_info = infer_feature_groups(feature_names)

    feature_sets = group_info["feature_sets"]
    group_parts = group_info["group_parts"]

    seed_out_dir = out_dir / f"seed_{seed}"
    ensure_dir(seed_out_dir)
    ensure_dir(seed_out_dir / "predictions")
    ensure_dir(seed_out_dir / "selected_features")

    print("\nFeature group sizes:")
    for group_name, cols in group_parts.items():
        print(f"  {group_name}: {len(cols)}")

    print("\nFeature set sizes:")
    for set_name, cols in feature_sets.items():
        print(f"  {set_name}: {len(cols)}")

    pd.DataFrame({"feature_name": feature_names}).to_csv(
        seed_out_dir / "a1flat_feature_names.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(seed_out_dir / "feature_groups.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_sets": feature_sets,
                "group_parts": group_parts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    rows: list[dict] = []

    # -------------------------------------------------------------------------
    # Step 1: Anchor baselines
    # -------------------------------------------------------------------------
    anchor_sets = [
        "A1FlatFull",
        "A1FlatDiagOnly",
        "A1FlatPairOnly",
        "A1FlatNoHeur",
        "A1FlatNoPairwise",
        "A1FlatDiagCorePair",
        "A1FlatDiagCoreTerminalPair",
        "A1FlatDiagAllPair",
    ]

    for feature_set_name in anchor_sets:
        cols = feature_sets[feature_set_name]

        if len(cols) == 0:
            print(f"[SKIP] Empty feature set: {feature_set_name}")
            continue

        run_plain_svr_experiment(
            rows=rows,
            seed=seed,
            seed_out_dir=seed_out_dir,
            test_df=test_df,
            X_train_df=X_train_df,
            X_val_df=X_val_df,
            X_test_df=X_test_df,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            w_train=w_train,
            feature_set_name=feature_set_name,
            cols=cols,
            model_name=f"{feature_set_name}-TunedSVR",
            step="01_anchor_baseline",
            score_mode="rmse",
        )

    # -------------------------------------------------------------------------
    # Step 2: Full selected adaptive baseline
    # -------------------------------------------------------------------------
    run_adaptive_selected_svr_experiment(
        rows=rows,
        seed=seed,
        seed_out_dir=seed_out_dir,
        test_df=test_df,
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        w_train=w_train,
        base_feature_set_name="A1FlatFull",
        cols=feature_sets["A1FlatFull"],
        model_name="A1FlatSelected-TunedSVR",
        step="02_full_adaptive_selected_baseline",
        score_mode="rmse",
    )

    # -------------------------------------------------------------------------
    # Step 3: DiagCorePair + fixed SelectK
    # -------------------------------------------------------------------------
    for k in SELECT_K_LIST:
        run_selected_svr_experiment(
            rows=rows,
            seed=seed,
            seed_out_dir=seed_out_dir,
            test_df=test_df,
            X_train_df=X_train_df,
            X_val_df=X_val_df,
            X_test_df=X_test_df,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            w_train=w_train,
            base_feature_set_name="A1FlatDiagCorePair",
            cols=feature_sets["A1FlatDiagCorePair"],
            k=k,
            model_name=f"A1FlatDiagCorePairSelectedK{k}-TunedSVR",
            step="03_diagcorepair_selected_svr",
            score_mode="rmse",
        )

    # -------------------------------------------------------------------------
    # Step 4: DiagCorePair + adaptive SelectK
    # -------------------------------------------------------------------------
    run_adaptive_selected_svr_experiment(
        rows=rows,
        seed=seed,
        seed_out_dir=seed_out_dir,
        test_df=test_df,
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        w_train=w_train,
        base_feature_set_name="A1FlatDiagCorePair",
        cols=feature_sets["A1FlatDiagCorePair"],
        model_name="A1FlatDiagCorePairSelected-TunedSVR",
        step="04_diagcorepair_adaptive_selected_svr",
        score_mode="rmse",
    )

    # -------------------------------------------------------------------------
    # Step 5: Rank-aware versions
    # -------------------------------------------------------------------------
    run_plain_svr_experiment(
        rows=rows,
        seed=seed,
        seed_out_dir=seed_out_dir,
        test_df=test_df,
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        w_train=w_train,
        feature_set_name="A1FlatDiagCorePair",
        cols=feature_sets["A1FlatDiagCorePair"],
        model_name="A1FlatDiagCorePair-RankAwareSVR",
        step="05_rank_aware_svr",
        score_mode="rank_aware",
    )

    run_adaptive_selected_svr_experiment(
        rows=rows,
        seed=seed,
        seed_out_dir=seed_out_dir,
        test_df=test_df,
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        w_train=w_train,
        base_feature_set_name="A1FlatDiagCorePair",
        cols=feature_sets["A1FlatDiagCorePair"],
        model_name="A1FlatDiagCorePairSelected-RankAwareSVR",
        step="05_rank_aware_selected_svr",
        score_mode="rank_aware",
    )

    for k in [100, 200, 400]:
        run_selected_svr_experiment(
            rows=rows,
            seed=seed,
            seed_out_dir=seed_out_dir,
            test_df=test_df,
            X_train_df=X_train_df,
            X_val_df=X_val_df,
            X_test_df=X_test_df,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            w_train=w_train,
            base_feature_set_name="A1FlatDiagCorePair",
            cols=feature_sets["A1FlatDiagCorePair"],
            k=k,
            model_name=f"A1FlatDiagCorePairSelectedK{k}-RankAwareSVR",
            step="05_rank_aware_selected_fixedk_svr",
            score_mode="rank_aware",
        )

    # -------------------------------------------------------------------------
    # Step 6: Strong-tail weighted versions
    # -------------------------------------------------------------------------
    for alpha in STRONG_WEIGHT_ALPHA_LIST:
        w_train_strong = make_strong_tail_weight(
            y_train=y_train,
            base_weight=w_train,
            alpha=alpha,
            threshold=STRONG_ACTIVITY_THRESHOLD,
        )

        run_plain_svr_experiment(
            rows=rows,
            seed=seed,
            seed_out_dir=seed_out_dir,
            test_df=test_df,
            X_train_df=X_train_df,
            X_val_df=X_val_df,
            X_test_df=X_test_df,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            w_train=w_train_strong,
            feature_set_name="A1FlatDiagCorePair",
            cols=feature_sets["A1FlatDiagCorePair"],
            model_name=f"A1FlatDiagCorePair-StrongWeightedSVR-alpha{alpha}",
            step="06_strong_tail_weighted_svr",
            score_mode="rmse",
            extra={"strong_weight_alpha": alpha},
        )

        run_selected_svr_experiment(
            rows=rows,
            seed=seed,
            seed_out_dir=seed_out_dir,
            test_df=test_df,
            X_train_df=X_train_df,
            X_val_df=X_val_df,
            X_test_df=X_test_df,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            w_train=w_train_strong,
            base_feature_set_name="A1FlatDiagCorePair",
            cols=feature_sets["A1FlatDiagCorePair"],
            k=200,
            model_name=f"A1FlatDiagCorePairSelectedK200-StrongWeightedSVR-alpha{alpha}",
            step="06_strong_tail_weighted_selected_svr",
            score_mode="rmse",
            extra={"strong_weight_alpha": alpha},
        )

    seed_raw_df = pd.DataFrame(rows)
    seed_raw_df.to_csv(
        seed_out_dir / "seed_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    show_cols = [
        "model",
        "step",
        "feature_set",
        "n_features",
        "score_mode",
        "test_rmse",
        "test_mae",
        "test_spearman",
        "test_top5_hit_rate",
        "test_top10_hit_rate",
        "test_strong_mae",
        "test_strong_bias",
    ]

    print(f"\nSeed {seed} finished. Top 20 by test RMSE:")
    print(
        seed_raw_df[show_cols]
        .sort_values(["test_rmse", "test_mae"], ascending=[True, True])
        .head(20)
        .to_string(index=False)
    )

    return rows


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dir(OUT_DIR)

    all_rows: list[dict] = []

    for seed in SEEDS:
        seed_rows = run_for_seed(seed=seed, out_dir=OUT_DIR)
        all_rows.extend(seed_rows)

        partial_path = OUT_DIR / "diagcorepair_svr_suite_raw_partial.csv"
        pd.DataFrame(all_rows).to_csv(
            partial_path,
            index=False,
            encoding="utf-8-sig",
        )

    raw_df = pd.DataFrame(all_rows)

    raw_path = OUT_DIR / "diagcorepair_svr_suite_raw.csv"
    summary_by_model_path = OUT_DIR / "diagcorepair_svr_suite_summary_by_model.csv"
    summary_detail_path = OUT_DIR / "diagcorepair_svr_suite_summary_detail.csv"

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    summary_by_model, summary_detail = summarize_results(raw_df)

    summary_by_model.to_csv(
        summary_by_model_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_detail.to_csv(
        summary_detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("DiagCorePair SVR suite finished.")
    print(f"Raw results saved to      : {raw_path}")
    print(f"Summary by model saved to : {summary_by_model_path}")
    print(f"Summary detail saved to   : {summary_detail_path}")
    print("=" * 100)

    show_cols = [
        "model",
        "n_runs",
        "seeds",
        "n_features_min",
        "n_features_max",
        "test_rmse_mean",
        "test_rmse_std",
        "test_mae_mean",
        "test_mae_std",
        "test_spearman_mean",
        "test_spearman_std",
        "test_top5_hit_rate_mean",
        "test_top10_hit_rate_mean",
        "test_top20_hit_rate_mean",
        "test_strong_mae_mean",
        "test_strong_bias_mean",
    ]

    print("\nTop 30 models by test RMSE:")
    print(
        summary_by_model[show_cols]
        .head(30)
        .to_string(index=False)
    )

    print("\nTop 30 models by Spearman:")
    print(
        summary_by_model[show_cols]
        .sort_values(
            ["test_spearman_mean", "test_rmse_mean"],
            ascending=[False, True],
        )
        .head(30)
        .to_string(index=False)
    )

    print("\nTop 30 models by Top10 hit rate:")
    print(
        summary_by_model[show_cols]
        .sort_values(
            ["test_top10_hit_rate_mean", "test_rmse_mean"],
            ascending=[False, True],
        )
        .head(30)
        .to_string(index=False)
    )

    print("\nTop 30 models by strong MAE:")
    print(
        summary_by_model[show_cols]
        .sort_values(
            ["test_strong_mae_mean", "test_rmse_mean"],
            ascending=[True, True],
        )
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()