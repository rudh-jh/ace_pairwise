from __future__ import annotations

import json
import random
import sys
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

from ace_pre.baselines.tabular_features import build_descriptor_frame

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

def get_config_path(project_root: Path) -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return project_root / "configs" / "train_a1_v1.yaml"

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_descriptor_frame(df, sequence_col="sequence")


def fit_mean_predictor(y_train: np.ndarray) -> dict:
    return {"mean_value": float(np.mean(y_train))}


def predict_mean_predictor(model: dict, n: int) -> np.ndarray:
    return np.full(shape=(n,), fill_value=model["mean_value"], dtype=float)


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
    # cfg = load_yaml(project_root / "configs" / "train_a1_v1.yaml")
    config_path = get_config_path(project_root)
    cfg = load_yaml(config_path)
    split_dir = project_root / cfg["paths"]["split_dir"]

    # artifacts_dir = project_root / "artifacts" / "baselines_v1"
    artifacts_dir = project_root / cfg["paths"]["baseline_artifacts_dir"]
    ensure_dir(artifacts_dir)

    seed = int(cfg["split"]["seed"])
    set_seed(seed)

    train_df = pd.read_csv(split_dir / "train_joint.csv")
    val_df = pd.read_csv(split_dir / "val_main.csv")
    test_df = pd.read_csv(split_dir / "test_main.csv")

    X_train = build_features(train_df)
    X_val = build_features(val_df)
    X_test = build_features(test_df)

    y_train = train_df["label_pIC50"].astype(float).values
    y_val = val_df["label_pIC50"].astype(float).values
    y_test = test_df["label_pIC50"].astype(float).values
    sample_weight = train_df["sample_weight"].astype(float).values

    results = []
    val_pred_df = pd.DataFrame({"y_true": y_val})
    test_pred_df = pd.DataFrame({"y_true": y_test})

    # Mean
    mean_model = fit_mean_predictor(y_train)
    mean_val_pred = predict_mean_predictor(mean_model, len(y_val))
    mean_test_pred = predict_mean_predictor(mean_model, len(y_test))
    results.append({"model": "MeanPredictor", "split": "val", **evaluate_regression(y_val, mean_val_pred), "params": json.dumps(mean_model, ensure_ascii=False)})
    results.append({"model": "MeanPredictor", "split": "test", **evaluate_regression(y_test, mean_test_pred), "params": json.dumps(mean_model, ensure_ascii=False)})
    val_pred_df["MeanPredictor"] = mean_val_pred
    test_pred_df["MeanPredictor"] = mean_test_pred

    # Ridge
    ridge_model, ridge_params = fit_ridge(X_train, y_train, sample_weight)
    ridge_val_pred = ridge_model.predict(X_val)
    ridge_test_pred = ridge_model.predict(X_test)
    results.append({"model": "Ridge", "split": "val", **evaluate_regression(y_val, ridge_val_pred), "params": json.dumps(ridge_params, ensure_ascii=False)})
    results.append({"model": "Ridge", "split": "test", **evaluate_regression(y_test, ridge_test_pred), "params": json.dumps(ridge_params, ensure_ascii=False)})
    val_pred_df["Ridge"] = ridge_val_pred
    test_pred_df["Ridge"] = ridge_test_pred

    # SVR
    svr_model, svr_params = fit_svr(X_train, y_train)
    svr_val_pred = svr_model.predict(X_val)
    svr_test_pred = svr_model.predict(X_test)
    results.append({"model": "SVR", "split": "val", **evaluate_regression(y_val, svr_val_pred), "params": json.dumps(svr_params, ensure_ascii=False)})
    results.append({"model": "SVR", "split": "test", **evaluate_regression(y_test, svr_test_pred), "params": json.dumps(svr_params, ensure_ascii=False)})
    val_pred_df["SVR"] = svr_val_pred
    test_pred_df["SVR"] = svr_test_pred

    # CatBoost
    if HAS_CATBOOST:
        cb_model, cb_params = fit_catboost(X_train, y_train, sample_weight)
        cb_val_pred = cb_model.predict(X_val)
        cb_test_pred = cb_model.predict(X_test)
        results.append({"model": "CatBoost", "split": "val", **evaluate_regression(y_val, cb_val_pred), "params": json.dumps(cb_params, ensure_ascii=False)})
        results.append({"model": "CatBoost", "split": "test", **evaluate_regression(y_test, cb_test_pred), "params": json.dumps(cb_params, ensure_ascii=False)})
        val_pred_df["CatBoost"] = cb_val_pred
        test_pred_df["CatBoost"] = cb_test_pred
    else:
        print("CatBoost not installed, skipped.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(artifacts_dir / "baseline_results.csv", index=False, encoding="utf-8-sig")
    val_pred_df.to_csv(artifacts_dir / "baseline_val_predictions.csv", index=False, encoding="utf-8-sig")
    test_pred_df.to_csv(artifacts_dir / "baseline_test_predictions.csv", index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved results to: {artifacts_dir / 'baseline_results.csv'}")


if __name__ == "__main__":
    main()