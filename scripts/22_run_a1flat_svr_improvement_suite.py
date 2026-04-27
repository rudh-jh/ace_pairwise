from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR

project_root = Path(__file__).resolve().parents[1]
src_root = project_root / "src"

for p in [project_root, src_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ace_pre.baselines.tensor_flat_features import build_a1_flat_feature_frame


warnings.filterwarnings("ignore", category=RuntimeWarning)


SEEDS = [42, 52, 62]

TOP_KS = [5, 10, 20]

# 第一版不要太大，先跑稳。
SVR_PARAM_GRID = [
    {"C": C, "epsilon": eps, "gamma": gamma}
    for C in [1.0, 2.0, 5.0, 10.0]
    for eps in [0.05, 0.10, 0.20]
    for gamma in ["scale", 0.01, 0.03]
]

SELECT_K_LIST = [50, 100, 200, 400]

STRONG_WEIGHT_SETTINGS = [
    {"strong_factor": 2.0, "medium_factor": 1.3},
    {"strong_factor": 3.0, "medium_factor": 1.5},
]

CORE_PAIR_KEYWORDS = [
    "pair_hydrophobicity",
    "pair_hydro",
    "pair_charge",
    "pair_polarity",
    "pair_volume",
    "pair_bulk",
    "pair_aromatic_pair",
    "pair_donor_acceptor",
]

PAIR_PHYSCHEM_KEYWORDS = [
    "pair_hydrophobicity",
    "pair_hydro",
    "pair_charge",
    "pair_polarity",
    "pair_volume",
    "pair_bulk",
    "pair_aromatic_pair",
    "pair_donor_acceptor",
    "pair_substitution",
    "pair_residue_class",
]

MULTI_KERNEL_WEIGHT_SPECS = [
    {"diag": 1.0, "core_pair": 0.0, "other_pair": 0.0, "heur": 0.0, "other": 0.0},
    {"diag": 1.0, "core_pair": 0.25, "other_pair": 0.0, "heur": 0.0, "other": 0.0},
    {"diag": 1.0, "core_pair": 0.50, "other_pair": 0.0, "heur": 0.0, "other": 0.0},
    {"diag": 1.0, "core_pair": 1.00, "other_pair": 0.0, "heur": 0.0, "other": 0.0},
    {"diag": 1.0, "core_pair": 1.00, "other_pair": 0.25, "heur": 0.0, "other": 0.0},
    {"diag": 1.0, "core_pair": 1.00, "other_pair": 0.50, "heur": 0.10, "other": 0.10},
]

MULTI_KERNEL_PARAM_GRID = [
    {
        "C": C,
        "epsilon": eps,
        "gamma": gamma,
        "weights": weights,
    }
    for C in [1.0, 2.0, 5.0]
    for eps in [0.05, 0.10]
    for gamma in ["scale", 0.01, 0.03]
    for weights in MULTI_KERNEL_WEIGHT_SPECS
]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    return len(true_top & pred_top) / k


def strong_activity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_true >= 6.0

    if mask.sum() == 0:
        return {
            "strong_n": 0,
            "strong_mae": np.nan,
            "strong_bias": np.nan,
        }

    return {
        "strong_n": int(mask.sum()),
        "strong_mae": float(np.mean(np.abs(y_pred[mask] - y_true[mask]))),
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


def boost_activity_weights(
    y: np.ndarray,
    base_weight: np.ndarray,
    strong_factor: float,
    medium_factor: float,
) -> np.ndarray:
    w = base_weight.astype(float).copy()

    w[(y >= 5.0) & (y < 6.0)] *= medium_factor
    w[y >= 6.0] *= strong_factor

    return w


def infer_feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    diag_cols = []
    pair_cols = []
    heur_cols = []
    core_pair_cols = []
    pair_physchem_cols = []
    other_cols = []

    for col in feature_names:
        lower = col.lower()

        is_diag = "diag_" in lower or lower.startswith("diag")
        is_pair = "pair_" in lower or "pair" in lower
        is_heur = "heur_" in lower or "heur" in lower or "favorable" in lower

        is_core_pair = is_pair and any(k in lower for k in CORE_PAIR_KEYWORDS)
        is_pair_physchem = is_pair and any(k in lower for k in PAIR_PHYSCHEM_KEYWORDS)

        if is_diag:
            diag_cols.append(col)
        if is_pair:
            pair_cols.append(col)
        if is_heur:
            heur_cols.append(col)
        if is_core_pair:
            core_pair_cols.append(col)
        if is_pair_physchem:
            pair_physchem_cols.append(col)

        if not is_diag and not is_pair and not is_heur:
            other_cols.append(col)

    all_cols = list(feature_names)

    no_heur_cols = [c for c in all_cols if c not in set(heur_cols)]
    no_pairwise_cols = [c for c in all_cols if c not in set(pair_cols)]
    no_pair_physchem_cols = [c for c in all_cols if c not in set(pair_physchem_cols)]

    diag_core_pair_cols = sorted(set(diag_cols + core_pair_cols), key=all_cols.index)

    groups = {
        "A1FlatFull": all_cols,
        "A1FlatDiagOnly": diag_cols,
        "A1FlatPairOnly": pair_cols,
        "A1FlatNoHeur": no_heur_cols,
        "A1FlatNoPairwise": no_pairwise_cols,
        "A1FlatNoPairPhyschem": no_pair_physchem_cols,
        "A1FlatDiagCorePair": diag_core_pair_cols,
    }

    group_parts = {
        "diag": diag_cols,
        "core_pair": core_pair_cols,
        "other_pair": [c for c in pair_cols if c not in set(core_pair_cols)],
        "heur": heur_cols,
        "other": other_cols,
    }

    return {
        "feature_sets": groups,
        "kernel_parts": group_parts,
    }


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
    cols = ["sample_id", "sequence", "length", "label_pIC50"]

    out = pd.DataFrame({
        "seed": seed,
        "model": model_name,
        "row_index": np.arange(len(test_df)),
    })

    for col in cols:
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


def build_group_arrays_for_kernel(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    kernel_parts: dict[str, list[str]],
) -> dict[str, dict[str, np.ndarray]]:
    out = {}

    for group_name, cols in kernel_parts.items():
        if not cols:
            continue

        X_train, X_val, X_test, _ = standardize_from_train(
            X_train_df,
            X_val_df,
            X_test_df,
            cols,
        )

        out[group_name] = {
            "train": X_train,
            "val": X_val,
            "test": X_test,
            "n_features": X_train.shape[1],
        }

    return out


def gamma_value(gamma_setting, n_features: int):
    if gamma_setting == "scale":
        return 1.0 / max(1, n_features)
    return float(gamma_setting)


def combined_kernel(
    group_arrays: dict[str, dict[str, np.ndarray]],
    split_a: str,
    split_b: str,
    weights: dict,
    gamma_setting,
) -> np.ndarray:
    K = None
    total_weight = 0.0

    for group_name, arrays in group_arrays.items():
        w = float(weights.get(group_name, 0.0))

        if w <= 0:
            continue

        Xa = arrays[split_a]
        Xb = arrays[split_b]

        gamma = gamma_value(gamma_setting, Xa.shape[1])

        Kg = rbf_kernel(Xa, Xb, gamma=gamma)

        if K is None:
            K = w * Kg
        else:
            K += w * Kg

        total_weight += w

    if K is None:
        raise ValueError("No kernel group has positive weight.")

    return K / max(total_weight, 1e-8)


def fit_best_multikernel_svr(
    group_arrays: dict[str, dict[str, np.ndarray]],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    sample_weight: np.ndarray,
    param_grid: list[dict],
) -> dict:
    best = None

    for params in param_grid:
        weights = params["weights"]
        gamma = params["gamma"]

        try:
            K_train = combined_kernel(
                group_arrays,
                split_a="train",
                split_b="train",
                weights=weights,
                gamma_setting=gamma,
            )
            K_val = combined_kernel(
                group_arrays,
                split_a="val",
                split_b="train",
                weights=weights,
                gamma_setting=gamma,
            )
            K_test = combined_kernel(
                group_arrays,
                split_a="test",
                split_b="train",
                weights=weights,
                gamma_setting=gamma,
            )
        except ValueError:
            continue

        model = SVR(
            kernel="precomputed",
            C=float(params["C"]),
            epsilon=float(params["epsilon"]),
        )

        model.fit(K_train, y_train, sample_weight=sample_weight)

        val_pred = model.predict(K_val)
        score = val_score(y_val, val_pred, mode="rmse")

        if best is None or score < best["score"]:
            test_pred = model.predict(K_test)

            best = {
                "model": model,
                "params": {
                    "C": params["C"],
                    "epsilon": params["epsilon"],
                    "gamma": params["gamma"],
                    "weights": params["weights"],
                },
                "score": score,
                "val_pred": val_pred,
                "test_pred": test_pred,
                "val_metrics": evaluate_predictions(y_val, val_pred),
                "test_metrics": evaluate_predictions(y_test, test_pred),
            }

    if best is None:
        raise RuntimeError("No valid multi-kernel SVR model was fitted.")

    return best


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
            test_strong_mae_mean=("test_strong_mae", "mean"),
            test_strong_bias_mean=("test_strong_bias", "mean"),
            val_rmse_mean=("val_rmse", "mean"),
            val_rmse_std=("val_rmse", "std"),
        )
        .sort_values(["test_rmse_mean", "test_mae_mean"], ascending=[True, True])
    )

    return summary


def run_for_seed(seed: int, base_cfg: dict, out_dir: Path) -> list[dict]:
    print("\n" + "=" * 90)
    print(f"Running A1Flat-SVR improvement suite | seed={seed}")
    print("=" * 90)

    split_dir = project_root / "data" / "final" / "splits" / f"seed_{seed}"

    train_df = pd.read_csv(split_dir / "train_joint.csv")
    val_df = pd.read_csv(split_dir / "val_main.csv")
    test_df = pd.read_csv(split_dir / "test_main.csv")

    y_train, w_train_base = get_y_w(train_df)
    y_val, _ = get_y_w(val_df)
    y_test, _ = get_y_w(test_df)

    print("Building A1Flat feature frames...")
    X_train_df, X_val_df, X_test_df = align_feature_frames(train_df, val_df, test_df)

    feature_names = list(X_train_df.columns)
    groups = infer_feature_groups(feature_names)

    feature_sets = groups["feature_sets"]
    kernel_parts = groups["kernel_parts"]

    print("Feature set sizes:")
    for name, cols in feature_sets.items():
        print(f"  {name}: {len(cols)}")

    seed_out_dir = out_dir / f"seed_{seed}"
    ensure_dir(seed_out_dir)

    pd.DataFrame(
        {
            "feature_name": feature_names,
        }
    ).to_csv(seed_out_dir / "a1flat_feature_names.csv", index=False, encoding="utf-8-sig")

    with open(seed_out_dir / "feature_groups.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_sets": {k: v for k, v in feature_sets.items()},
                "kernel_parts": {k: v for k, v in kernel_parts.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    rows = []

    # ------------------------------------------------------------------
    # Step 1 + 2: Tuned Full SVR + feature group ablation
    # ------------------------------------------------------------------
    for feature_set_name, cols in feature_sets.items():
        if not cols:
            print(f"[SKIP] Empty feature set: {feature_set_name}")
            continue

        print(f"\n[Step 1/2] Tuned SVR on feature set: {feature_set_name}")

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
            sample_weight=w_train_base,
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

        step_name = "01_tuned_full_svr" if feature_set_name == "A1FlatFull" else "02_feature_group_ablation"

        rows.append(
            result_row(
                seed=seed,
                step=step_name,
                model_name=model_name,
                feature_set=feature_set_name,
                n_features=len(used_cols),
                best=best,
            )
        )

    # ------------------------------------------------------------------
    # Step 3: A1FlatSelected-SVR
    # ------------------------------------------------------------------
    print("\n[Step 3] A1FlatSelected-SVR")

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
            sample_weight=w_train_base,
            param_grid=SVR_PARAM_GRID,
            score_mode="rmse",
        )

        if selected_best is None or best_k["score"] < selected_best["score"]:
            selected_best = best_k
            selected_best_info = {
                "k": k,
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
            step="03_selected_svr",
            model_name=model_name,
            feature_set=f"A1FlatSelected_k{selected_best_info['k']}",
            n_features=selected_best_info["selected_n_features"],
            best=selected_best,
            extra={"selected_k": selected_best_info["k"]},
        )
    )

    # ------------------------------------------------------------------
    # Step 4: Multi-kernel A1Flat-SVR
    # ------------------------------------------------------------------
    print("\n[Step 4] Multi-kernel A1Flat-SVR")

    group_arrays = build_group_arrays_for_kernel(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        kernel_parts=kernel_parts,
    )

    mk_best = fit_best_multikernel_svr(
        group_arrays=group_arrays,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        sample_weight=w_train_base,
        param_grid=MULTI_KERNEL_PARAM_GRID,
    )

    model_name = "A1FlatMultiKernel-TunedSVR"

    save_predictions(
        path=seed_out_dir / "predictions" / f"{model_name}.csv",
        seed=seed,
        model_name=model_name,
        test_df=test_df,
        y_true=y_test,
        y_pred=mk_best["test_pred"],
    )

    mk_feature_count = int(
        sum(arrays["n_features"] for arrays in group_arrays.values())
    )

    rows.append(
        result_row(
            seed=seed,
            step="04_multi_kernel_svr",
            model_name=model_name,
            feature_set="A1FlatMultiKernel",
            n_features=mk_feature_count,
            best=mk_best,
        )
    )

    # ------------------------------------------------------------------
    # Step 5: Strong-activity weighted SVR
    # ------------------------------------------------------------------
    print("\n[Step 5] Strong-activity weighted SVR")

    X_train, X_val, X_test, used_cols = standardize_from_train(
        X_train_df,
        X_val_df,
        X_test_df,
        feature_sets["A1FlatFull"],
    )

    weighted_best = None
    weighted_best_setting = None

    for setting in STRONG_WEIGHT_SETTINGS:
        w_train_boost = boost_activity_weights(
            y=y_train,
            base_weight=w_train_base,
            strong_factor=setting["strong_factor"],
            medium_factor=setting["medium_factor"],
        )

        best_setting = fit_best_svr(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            sample_weight=w_train_boost,
            param_grid=SVR_PARAM_GRID,
            score_mode="rmse",
        )

        if weighted_best is None or best_setting["score"] < weighted_best["score"]:
            weighted_best = best_setting
            weighted_best_setting = setting

    assert weighted_best is not None
    assert weighted_best_setting is not None

    model_name = "A1FlatFull-StrongWeightedSVR"

    save_predictions(
        path=seed_out_dir / "predictions" / f"{model_name}.csv",
        seed=seed,
        model_name=model_name,
        test_df=test_df,
        y_true=y_test,
        y_pred=weighted_best["test_pred"],
    )

    rows.append(
        result_row(
            seed=seed,
            step="05_strong_activity_weighted_svr",
            model_name=model_name,
            feature_set="A1FlatFull",
            n_features=len(used_cols),
            best=weighted_best,
            extra=weighted_best_setting,
        )
    )

    seed_raw_df = pd.DataFrame(rows)
    seed_raw_df.to_csv(
        seed_out_dir / "seed_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSeed {seed} finished.")
    print(seed_raw_df[["model", "test_rmse", "test_mae", "test_spearman", "test_top10_hit_rate"]].sort_values("test_rmse"))

    return rows


def main() -> None:
    base_cfg_path = project_root / "configs" / "train_a1_v1.yaml"

    if base_cfg_path.exists():
        base_cfg = load_yaml(base_cfg_path)
    else:
        base_cfg = {}

    out_dir = project_root / "artifacts" / "a1flat_svr_improvement_suite"
    ensure_dir(out_dir)

    all_rows = []

    for seed in SEEDS:
        rows = run_for_seed(seed, base_cfg, out_dir)
        all_rows.extend(rows)

        pd.DataFrame(all_rows).to_csv(
            out_dir / "a1flat_svr_improvement_raw_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

    raw_df = pd.DataFrame(all_rows)

    raw_df.to_csv(
        out_dir / "a1flat_svr_improvement_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = summarize_results(raw_df)

    summary_df.to_csv(
        out_dir / "a1flat_svr_improvement_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 90)
    print("All A1Flat-SVR improvement experiments finished.")
    print(f"Raw results saved to: {out_dir / 'a1flat_svr_improvement_raw.csv'}")
    print(f"Summary saved to: {out_dir / 'a1flat_svr_improvement_summary.csv'}")
    print("=" * 90)

    print("\nSummary:")
    print(
        summary_df[
            [
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
                "test_strong_mae_mean",
                "test_strong_bias_mean",
            ]
        ]
    )


if __name__ == "__main__":
    main()