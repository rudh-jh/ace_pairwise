from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.svm import SVR

project_root = Path(__file__).resolve().parents[1]
src_root = project_root / "src"

for p in [project_root, src_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ace_pre.baselines.tensor_flat_features import build_a1_flat_feature_frame

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Basic settings
# =============================================================================

SEEDS = [42, 52, 62]
TOP_KS = [5, 10, 20]

OUT_DIR = project_root / "artifacts" / "a1flat_group_weighted_svr"

SVR_PARAM_GRID = [
    {"C": C, "epsilon": eps, "gamma": gamma}
    for C in [1.0, 2.0, 5.0, 10.0]
    for eps in [0.05, 0.10, 0.20]
    for gamma in ["scale", 0.01, 0.03]
]

SELECT_K_LIST = [50, 100, 200, 400]


# =============================================================================
# Group-weight settings
# =============================================================================
# 注意：
# 1. 这些权重是在 StandardScaler 之后乘上去的。
# 2. 如果先乘权重再标准化，权重会被抵消。
# 3. diag 是主信号；core_pair 是主要 pairwise 理化交互；terminal_pair/order_pair 是结构辅助；
#    heur 是 ACE 经验启发式，容易有噪声，所以专门设置了降权和置零实验。

GROUP_WEIGHT_GRID = [
    {
        "name": "equal_weight",
        "diag": 1.0,
        "core_pair": 1.0,
        "terminal_pair": 1.0,
        "order_pair": 1.0,
        "other_pair": 1.0,
        "heur": 1.0,
        "other": 1.0,
    },
    {
        "name": "diag1p5_pair_aux",
        "diag": 1.5,
        "core_pair": 0.75,
        "terminal_pair": 0.75,
        "order_pair": 0.50,
        "other_pair": 0.50,
        "heur": 0.50,
        "other": 1.0,
    },
    {
        "name": "diag2_pair_aux",
        "diag": 2.0,
        "core_pair": 0.75,
        "terminal_pair": 0.75,
        "order_pair": 0.50,
        "other_pair": 0.50,
        "heur": 0.50,
        "other": 1.0,
    },
    {
        "name": "diag2_pair_noheur",
        "diag": 2.0,
        "core_pair": 0.75,
        "terminal_pair": 0.75,
        "order_pair": 0.50,
        "other_pair": 0.50,
        "heur": 0.0,
        "other": 1.0,
    },
    {
        "name": "diag2_corepair_only",
        "diag": 2.0,
        "core_pair": 0.75,
        "terminal_pair": 0.0,
        "order_pair": 0.0,
        "other_pair": 0.0,
        "heur": 0.0,
        "other": 1.0,
    },
    {
        "name": "diag3_pair_light",
        "diag": 3.0,
        "core_pair": 0.50,
        "terminal_pair": 0.50,
        "order_pair": 0.25,
        "other_pair": 0.25,
        "heur": 0.25,
        "other": 1.0,
    },
    {
        "name": "diag3_pair_noheur",
        "diag": 3.0,
        "core_pair": 0.50,
        "terminal_pair": 0.50,
        "order_pair": 0.25,
        "other_pair": 0.25,
        "heur": 0.0,
        "other": 1.0,
    },
    {
        "name": "diag4_pair_verylight",
        "diag": 4.0,
        "core_pair": 0.35,
        "terminal_pair": 0.35,
        "order_pair": 0.15,
        "other_pair": 0.15,
        "heur": 0.0,
        "other": 1.0,
    },
]


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

    true_top = set(np.argsort(-y_true)[:k].tolist())
    pred_top = set(np.argsort(-y_pred)[:k].tolist())
    return float(len(true_top & pred_top) / k)


def strong_activity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_true >= 6.0

    if mask.sum() == 0:
        return {
            "strong_n": 0,
            "strong_mae": np.nan,
            "strong_rmse": np.nan,
            "strong_bias": np.nan,
        }

    return {
        "strong_n": int(mask.sum()),
        "strong_mae": float(np.mean(np.abs(y_pred[mask] - y_true[mask]))),
        "strong_rmse": rmse(y_true[mask], y_pred[mask]),
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


def val_score(y_true: np.ndarray, y_pred: np.ndarray, mode: str = "rmse") -> float:
    if mode == "rmse":
        return rmse(y_true, y_pred)

    if mode == "rmse_spearman_top10":
        top10 = topk_hit_rate(y_true, y_pred, 10)
        if np.isnan(top10):
            top10 = 0.0

        return rmse(y_true, y_pred) - 0.15 * spearman(y_true, y_pred) - 0.15 * top10

    raise ValueError(f"Unknown score mode: {mode}")


def get_y_w(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = df["label_pIC50"].astype(float).values

    if "sample_weight" in df.columns:
        w = df["sample_weight"].astype(float).values
    else:
        w = np.ones_like(y, dtype=float)

    return y, w


# =============================================================================
# Feature construction and grouping
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
    diag_cols = []
    pair_cols = []
    heur_cols = []
    core_pair_cols = []
    terminal_pair_cols = []
    order_pair_cols = []
    other_pair_cols = []
    other_cols = []

    core_pair_markers = [
        "pair_hydrophobicity",
        "pair_charge",
        "pair_polarity",
        "pair_volume",
        "pair_both_hydrophobic",
        "pair_both_bulky",
        "pair_aromatic_pair",
        "pair_donor_acceptor",
        "pair_substitution",
        "pair_residue_class",
    ]

    terminal_pair_markers = [
        "pair_terminal",
        "pair_end_to_end",
        "pair_end_to_internal",
    ]

    order_pair_markers = [
        "pair_forward_order",
        "pair_reverse_order",
    ]

    for col in feature_names:
        lower = col.lower()

        is_diag = lower.startswith("diag_")
        is_pair = lower.startswith("pair_")
        is_heur = lower.startswith("heur_")

        if is_diag:
            diag_cols.append(col)
            continue

        if is_heur:
            heur_cols.append(col)
            continue

        if is_pair:
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

    no_heur_cols = [c for c in all_cols if c not in set(heur_cols)]
    no_pairwise_cols = [c for c in all_cols if c not in set(pair_cols)]
    diag_core_pair_cols = sorted(set(diag_cols + core_pair_cols), key=all_cols.index)
    diag_core_terminal_pair_cols = sorted(
        set(diag_cols + core_pair_cols + terminal_pair_cols),
        key=all_cols.index,
    )

    feature_sets = {
        "A1FlatFull": all_cols,
        "A1FlatDiagOnly": diag_cols,
        "A1FlatPairOnly": pair_cols,
        "A1FlatNoHeur": no_heur_cols,
        "A1FlatNoPairwise": no_pairwise_cols,
        "A1FlatDiagCorePair": diag_core_pair_cols,
        "A1FlatDiagCoreTerminalPair": diag_core_terminal_pair_cols,
    }

    group_parts = {
        "diag": diag_cols,
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
    if not cols:
        raise ValueError("No columns selected.")

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
        raise ValueError("No features left after VarianceThreshold.")

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


def build_feature_weight_vector(
    used_cols: list[str],
    group_parts: dict[str, list[str]],
    weight_spec: dict,
) -> np.ndarray:
    col_to_idx = {c: i for i, c in enumerate(used_cols)}
    weights = np.ones(len(used_cols), dtype=np.float32)

    for group_name in [
        "diag",
        "core_pair",
        "terminal_pair",
        "order_pair",
        "other_pair",
        "heur",
        "other",
    ]:
        group_cols = group_parts.get(group_name, [])
        group_weight = float(weight_spec.get(group_name, 1.0))

        for col in group_cols:
            if col in col_to_idx:
                weights[col_to_idx[col]] = group_weight

    return weights


def apply_group_weights(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    used_cols: list[str],
    group_parts: dict[str, list[str]],
    weight_spec: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    必须在 standardize_from_train() 或 standardize_and_select() 之后调用。
    先标准化，再分组加权，权重才真正有效。
    """
    feature_weights = build_feature_weight_vector(
        used_cols=used_cols,
        group_parts=group_parts,
        weight_spec=weight_spec,
    )

    X_train_w = X_train * feature_weights
    X_val_w = X_val * feature_weights
    X_test_w = X_test * feature_weights

    return X_train_w, X_val_w, X_test_w, feature_weights


# =============================================================================
# Model fitting
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
    score_mode: str = "rmse",
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

    for col in ["sample_id", "sequence", "length", "label_pIC50"]:
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
    extra: dict | None = None,
) -> dict:
    extra = extra or {}

    row = {
        "seed": seed,
        "step": step,
        "model": model_name,
        "feature_set": feature_set,
        "n_features": n_features,
        "val_score": best["score"],
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


# =============================================================================
# Summary
# =============================================================================

def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df
        .groupby(["model", "step", "feature_set"], as_index=False)
        .agg(
            n_features=("n_features", "first"),

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
            val_rmse_std=("val_rmse", "std"),
        )
        .sort_values(["test_rmse_mean", "test_mae_mean"], ascending=[True, True])
    )

    return summary


# =============================================================================
# Main experiment for each seed
# =============================================================================

def run_for_seed(seed: int, out_dir: Path) -> list[dict]:
    print("\n" + "=" * 100)
    print(f"Running A1Flat Group-Weighted SVR | seed={seed}")
    print("=" * 100)

    split_dir = project_root / "data" / "final" / "splits" / f"seed_{seed}"

    train_path = split_dir / "train_joint.csv"
    val_path = split_dir / "val_main.csv"
    test_path = split_dir / "test_main.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing split file: {train_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    y_train, w_train = get_y_w(train_df)
    y_val, _ = get_y_w(val_df)
    y_test, _ = get_y_w(test_df)

    print("Building A1Flat feature frames...")
    X_train_df, X_val_df, X_test_df = align_feature_frames(train_df, val_df, test_df)

    feature_names = list(X_train_df.columns)
    group_info = infer_feature_groups(feature_names)
    feature_sets = group_info["feature_sets"]
    group_parts = group_info["group_parts"]

    print("\nFeature group sizes:")
    for name, cols in group_parts.items():
        print(f"  {name}: {len(cols)}")

    print("\nFeature set sizes:")
    for name, cols in feature_sets.items():
        print(f"  {name}: {len(cols)}")

    seed_out_dir = out_dir / f"seed_{seed}"
    ensure_dir(seed_out_dir)
    ensure_dir(seed_out_dir / "predictions")

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

    rows = []

    # -------------------------------------------------------------------------
    # Step 1: Anchor baselines
    # -------------------------------------------------------------------------
    anchor_feature_sets = [
        "A1FlatFull",
        "A1FlatDiagOnly",
        "A1FlatPairOnly",
        "A1FlatNoHeur",
        "A1FlatNoPairwise",
        "A1FlatDiagCorePair",
        "A1FlatDiagCoreTerminalPair",
    ]

    for feature_set_name in anchor_feature_sets:
        cols = feature_sets.get(feature_set_name, [])

        if not cols:
            print(f"[SKIP] Empty feature set: {feature_set_name}")
            continue

        print(f"\n[Step 1] Anchor baseline: {feature_set_name}")

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
            score_mode="rmse",
        )

        model_name = f"{feature_set_name}-TunedSVR"

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
                step="01_anchor_baseline",
                model_name=model_name,
                feature_set=feature_set_name,
                n_features=len(used_cols),
                best=best,
            )
        )

    # -------------------------------------------------------------------------
    # Step 2: A1FlatSelected-TunedSVR baseline
    # -------------------------------------------------------------------------
    print("\n[Step 2] A1FlatSelected-TunedSVR baseline")

    selected_best = None
    selected_best_info = None

    for k in SELECT_K_LIST:
        X_train_s, X_val_s, X_test_s, selected_cols = standardize_and_select(
            X_train_df=X_train_df,
            X_val_df=X_val_df,
            X_test_df=X_test_df,
            y_train=y_train,
            cols=feature_sets["A1FlatFull"],
            k=k,
        )

        best_k = fit_best_svr(
            X_train=X_train_s,
            X_val=X_val_s,
            X_test=X_test_s,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            sample_weight=w_train,
            param_grid=SVR_PARAM_GRID,
            score_mode="rmse",
        )

        if selected_best is None or best_k["score"] < selected_best["score"]:
            selected_best = best_k
            selected_best_info = {
                "selected_k": k,
                "selected_n_features": len(selected_cols),
                "selected_cols": selected_cols,
            }

    assert selected_best is not None
    assert selected_best_info is not None

    model_name = "A1FlatSelected-TunedSVR"

    save_predictions(
        path=seed_out_dir / "predictions" / f"{model_name}.csv",
        seed=seed,
        model_name=model_name,
        test_df=test_df,
        y_true=y_test,
        y_pred=selected_best["test_pred"],
    )

    with open(seed_out_dir / "a1flat_selected_cols.json", "w", encoding="utf-8") as f:
        json.dump(selected_best_info, f, ensure_ascii=False, indent=2)

    rows.append(
        result_row(
            seed=seed,
            step="02_selected_svr_baseline",
            model_name=model_name,
            feature_set=f"A1FlatSelected_k{selected_best_info['selected_k']}",
            n_features=selected_best_info["selected_n_features"],
            best=selected_best,
            extra={"selected_k": selected_best_info["selected_k"]},
        )
    )

    # -------------------------------------------------------------------------
    # Step 3: Full feature group-weighted SVR
    # -------------------------------------------------------------------------
    print("\n[Step 3] A1FlatFull Group-Weighted SVR")

    X_train_full, X_val_full, X_test_full, used_cols_full = standardize_from_train(
        X_train_df,
        X_val_df,
        X_test_df,
        feature_sets["A1FlatFull"],
    )

    for weight_spec in GROUP_WEIGHT_GRID:
        weight_name = weight_spec["name"]
        print(f"  Running full group-weight setting: {weight_name}")

        X_train_w, X_val_w, X_test_w, feature_weights = apply_group_weights(
            X_train=X_train_full,
            X_val=X_val_full,
            X_test=X_test_full,
            used_cols=used_cols_full,
            group_parts=group_parts,
            weight_spec=weight_spec,
        )

        best = fit_best_svr(
            X_train=X_train_w,
            X_val=X_val_w,
            X_test=X_test_w,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            sample_weight=w_train,
            param_grid=SVR_PARAM_GRID,
            score_mode="rmse",
        )

        model_name = f"A1FlatFull-GroupWeightedSVR-{weight_name}"

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
                step="03_full_group_weighted_svr",
                model_name=model_name,
                feature_set="A1FlatFull_GroupWeighted",
                n_features=len(used_cols_full),
                best=best,
                extra={
                    "group_weight_name": weight_name,
                    "diag_weight": weight_spec.get("diag", 1.0),
                    "core_pair_weight": weight_spec.get("core_pair", 1.0),
                    "terminal_pair_weight": weight_spec.get("terminal_pair", 1.0),
                    "order_pair_weight": weight_spec.get("order_pair", 1.0),
                    "other_pair_weight": weight_spec.get("other_pair", 1.0),
                    "heur_weight": weight_spec.get("heur", 1.0),
                    "other_weight": weight_spec.get("other", 1.0),
                },
            )
        )

    # -------------------------------------------------------------------------
    # Step 4: Selected feature group-weighted SVR
    # -------------------------------------------------------------------------
    print("\n[Step 4] A1FlatSelected Group-Weighted SVR")

    for k in SELECT_K_LIST:
        print(f"  Selecting k={k} features...")

        X_train_s, X_val_s, X_test_s, selected_cols = standardize_and_select(
            X_train_df=X_train_df,
            X_val_df=X_val_df,
            X_test_df=X_test_df,
            y_train=y_train,
            cols=feature_sets["A1FlatFull"],
            k=k,
        )

        for weight_spec in GROUP_WEIGHT_GRID:
            weight_name = weight_spec["name"]
            print(f"    Running selected group-weight setting: k={k}, {weight_name}")

            X_train_sw, X_val_sw, X_test_sw, feature_weights = apply_group_weights(
                X_train=X_train_s,
                X_val=X_val_s,
                X_test=X_test_s,
                used_cols=selected_cols,
                group_parts=group_parts,
                weight_spec=weight_spec,
            )

            best = fit_best_svr(
                X_train=X_train_sw,
                X_val=X_val_sw,
                X_test=X_test_sw,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                sample_weight=w_train,
                param_grid=SVR_PARAM_GRID,
                score_mode="rmse",
            )

            model_name = f"A1FlatSelectedK{k}-GroupWeightedSVR-{weight_name}"

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
                    step="04_selected_group_weighted_svr",
                    model_name=model_name,
                    feature_set=f"A1FlatSelected_k{k}_GroupWeighted",
                    n_features=len(selected_cols),
                    best=best,
                    extra={
                        "selected_k": k,
                        "group_weight_name": weight_name,
                        "diag_weight": weight_spec.get("diag", 1.0),
                        "core_pair_weight": weight_spec.get("core_pair", 1.0),
                        "terminal_pair_weight": weight_spec.get("terminal_pair", 1.0),
                        "order_pair_weight": weight_spec.get("order_pair", 1.0),
                        "other_pair_weight": weight_spec.get("other_pair", 1.0),
                        "heur_weight": weight_spec.get("heur", 1.0),
                        "other_weight": weight_spec.get("other", 1.0),
                    },
                )
            )

    seed_raw_df = pd.DataFrame(rows)
    seed_raw_df.to_csv(
        seed_out_dir / "seed_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSeed {seed} finished.")
    print(
        seed_raw_df[
            [
                "model",
                "step",
                "n_features",
                "test_rmse",
                "test_mae",
                "test_spearman",
                "test_top10_hit_rate",
                "test_strong_mae",
                "test_strong_bias",
            ]
        ].sort_values(["test_rmse", "test_mae"]).head(20)
    )

    return rows


def main() -> None:
    ensure_dir(OUT_DIR)

    all_rows = []

    for seed in SEEDS:
        rows = run_for_seed(seed=seed, out_dir=OUT_DIR)
        all_rows.extend(rows)

        pd.DataFrame(all_rows).to_csv(
            OUT_DIR / "a1flat_group_weighted_raw_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

    raw_df = pd.DataFrame(all_rows)

    raw_path = OUT_DIR / "a1flat_group_weighted_raw.csv"
    summary_path = OUT_DIR / "a1flat_group_weighted_summary.csv"

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    summary_df = summarize_results(raw_df)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("All A1Flat Group-Weighted SVR experiments finished.")
    print(f"Raw results saved to: {raw_path}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 100)

    show_cols = [
        "model",
        "step",
        "feature_set",
        "n_features",
        "test_rmse_mean",
        "test_rmse_std",
        "test_mae_mean",
        "test_mae_std",
        "test_spearman_mean",
        "test_spearman_std",
        "test_top5_hit_rate_mean",
        "test_top10_hit_rate_mean",
        "test_top20_hit_rate_mean",
        "test_strong_n_mean",
        "test_strong_mae_mean",
        "test_strong_rmse_mean",
        "test_strong_bias_mean",
    ]

    print("\nTop 30 models by test RMSE:")
    print(summary_df[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()