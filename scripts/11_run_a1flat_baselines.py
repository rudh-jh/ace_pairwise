from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

project_root = Path(__file__).resolve().parents[1]
src_root = project_root / "src"
for p in [project_root, src_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ace_pre.baselines.tensor_flat_features import build_a1_flat_feature_frame

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_path(project_root: Path) -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return project_root / "configs" / "train_a1_v1.yaml"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s_true = pd.Series(y_true)
    s_pred = pd.Series(y_pred)
    val = s_true.corr(s_pred, method="spearman")
    return float(val) if pd.notna(val) else 0.0


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
    }


def fit_ridge(X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    model.fit(X_train, y_train, ridge__sample_weight=sample_weight)
    return model, {"alpha": 1.0}


def fit_svr(X_train: pd.DataFrame, y_train: np.ndarray):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr", SVR(C=2.0, epsilon=0.1, kernel="rbf")),
    ])
    model.fit(X_train, y_train)
    return model, {"C": 2.0, "epsilon": 0.1, "kernel": "rbf"}


def fit_catboost(X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray):
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost is not installed.")

    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=6,
        learning_rate=0.05,
        iterations=300,
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model, {
        "depth": 6,
        "learning_rate": 0.05,
        "iterations": 300,
    }


def main() -> None:
    t0 = time.time()

    config_path = get_config_path(project_root)
    cfg = load_yaml(config_path)

    split_dir = project_root / cfg["paths"]["split_dir"]
    artifacts_dir = project_root / cfg["paths"]["a1flat_artifacts_dir"]
    ensure_dir(artifacts_dir)

    seed = int(cfg["split"]["seed"])
    set_seed(seed)

    print(f"[1/6] Loading split files from: {split_dir}")
    train_df = pd.read_csv(split_dir / "train_joint.csv")
    val_df = pd.read_csv(split_dir / "val_main.csv")
    test_df = pd.read_csv(split_dir / "test_main.csv")
    print(f"      train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    print("[2/6] Building A1-flat features for train...")
    X_train = build_a1_flat_feature_frame(train_df, sequence_col="sequence")
    print(f"      X_train shape = {X_train.shape}")

    print("[3/6] Building A1-flat features for val...")
    X_val = build_a1_flat_feature_frame(val_df, sequence_col="sequence")
    print(f"      X_val shape = {X_val.shape}")

    print("[4/6] Building A1-flat features for test...")
    X_test = build_a1_flat_feature_frame(test_df, sequence_col="sequence")
    print(f"      X_test shape = {X_test.shape}")

    y_train = train_df["label_pIC50"].astype(float).values
    y_val = val_df["label_pIC50"].astype(float).values
    y_test = test_df["label_pIC50"].astype(float).values
    sample_weight = train_df["sample_weight"].astype(float).values

    results = []
    val_pred_df = pd.DataFrame({"y_true": y_val})
    test_pred_df = pd.DataFrame({"y_true": y_test})

    print("[5/6] Training A1Flat-Ridge...")
    ridge_model, ridge_params = fit_ridge(X_train, y_train, sample_weight)
    ridge_val_pred = ridge_model.predict(X_val)
    ridge_test_pred = ridge_model.predict(X_test)
    results.append({"model": "A1Flat-Ridge", "split": "val", **evaluate_regression(y_val, ridge_val_pred), "params": json.dumps(ridge_params, ensure_ascii=False)})
    results.append({"model": "A1Flat-Ridge", "split": "test", **evaluate_regression(y_test, ridge_test_pred), "params": json.dumps(ridge_params, ensure_ascii=False)})
    val_pred_df["A1Flat-Ridge"] = ridge_val_pred
    test_pred_df["A1Flat-Ridge"] = ridge_test_pred
    print("      done.")

    print("[5/6] Training A1Flat-SVR...")
    svr_model, svr_params = fit_svr(X_train, y_train)
    svr_val_pred = svr_model.predict(X_val)
    svr_test_pred = svr_model.predict(X_test)
    results.append({"model": "A1Flat-SVR", "split": "val", **evaluate_regression(y_val, svr_val_pred), "params": json.dumps(svr_params, ensure_ascii=False)})
    results.append({"model": "A1Flat-SVR", "split": "test", **evaluate_regression(y_test, svr_test_pred), "params": json.dumps(svr_params, ensure_ascii=False)})
    val_pred_df["A1Flat-SVR"] = svr_val_pred
    test_pred_df["A1Flat-SVR"] = svr_test_pred
    print("      done.")

    if HAS_CATBOOST:
        print("[5/6] Training A1Flat-CatBoost...")
        cb_model, cb_params = fit_catboost(X_train, y_train, sample_weight)
        cb_val_pred = cb_model.predict(X_val)
        cb_test_pred = cb_model.predict(X_test)
        results.append({"model": "A1Flat-CatBoost", "split": "val", **evaluate_regression(y_val, cb_val_pred), "params": json.dumps(cb_params, ensure_ascii=False)})
        results.append({"model": "A1Flat-CatBoost", "split": "test", **evaluate_regression(y_test, cb_test_pred), "params": json.dumps(cb_params, ensure_ascii=False)})
        val_pred_df["A1Flat-CatBoost"] = cb_val_pred
        test_pred_df["A1Flat-CatBoost"] = cb_test_pred
        print("      done.")
    else:
        print("[5/6] CatBoost not installed, skipped.")

    print("[6/6] Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(artifacts_dir / "a1flat_baseline_results.csv", index=False, encoding="utf-8-sig")
    val_pred_df.to_csv(artifacts_dir / "a1flat_val_predictions.csv", index=False, encoding="utf-8-sig")
    test_pred_df.to_csv(artifacts_dir / "a1flat_test_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"Done in {time.time() - t0:.1f}s")
    print(f"Saved results to: {artifacts_dir / 'a1flat_baseline_results.csv'}")


if __name__ == "__main__":
    main()