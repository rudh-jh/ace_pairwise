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
from ace_pre.baselines.tensor_flat_features import build_a1_flat_feature_frame

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


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


def build_seq_identity_frame(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    max_len: int = 5,
) -> pd.DataFrame:
    """
    纯序列信息特征：
    只使用氨基酸身份、位置、长度、AAC。
    不使用任何理化性质。
    """
    rows = []

    for _, row in df.iterrows():
        seq = str(row[sequence_col]).strip().upper()
        feats = {}

        # 长度信息
        feats["length"] = float(len(seq))
        for L in range(2, max_len + 1):
            feats[f"len_is_{L}"] = 1.0 if len(seq) == L else 0.0

        # 固定位置 one-hot
        for pos in range(max_len):
            aa_at_pos = seq[pos] if pos < len(seq) else "[PAD]"
            feats[f"pos{pos + 1}_is_PAD"] = 1.0 if aa_at_pos == "[PAD]" else 0.0

            for aa in AA_ORDER:
                feats[f"pos{pos + 1}_aa_{aa}"] = 1.0 if aa_at_pos == aa else 0.0

        # AAC 组成，不涉及理化性质
        seq_len = max(1, len(seq))
        for aa in AA_ORDER:
            feats[f"aac_{aa}"] = seq.count(aa) / seq_len

        rows.append(feats)

    return pd.DataFrame(rows, index=df.index)


def build_physchem_only_frame(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    """
    只保留理化性质特征：
    去掉 AAC、N 端 one-hot、C 端 one-hot。
    """
    desc = build_descriptor_frame(df, sequence_col=sequence_col)

    drop_prefixes = ("aac_", "nterm_", "cterm_")
    keep_cols = [
        c for c in desc.columns
        if not c.startswith(drop_prefixes)
    ]

    return desc[keep_cols].copy()


def build_descriptor_full_frame(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    """
    当前普通 tabular baseline 的完整描述符：
    AAC + N/C 端 one-hot + 全局理化性质 + 简单 pairwise 聚合性质。
    """
    return build_descriptor_frame(df, sequence_col=sequence_col)


def build_a1flat_full_frame(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    """
    A1 完整关系张量展平特征：
    与 A1 主模型使用同一套 handcrafted relation tensor。
    """
    return build_a1_flat_feature_frame(df, sequence_col=sequence_col)


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    model.fit(X_train, y_train, ridge__sample_weight=sample_weight)
    return model, {"alpha": 1.0}


def fit_svr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr", SVR(C=2.0, epsilon=0.1, kernel="rbf")),
    ])

    # sklearn 的 SVR 支持 sample_weight。
    model.fit(X_train, y_train, svr__sample_weight=sample_weight)

    return model, {
        "C": 2.0,
        "epsilon": 0.1,
        "kernel": "rbf",
    }


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
):
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

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
    )

    return model, {
        "depth": 6,
        "learning_rate": 0.05,
        "iterations": 300,
    }


def fit_mean_predictor(y_train: np.ndarray) -> dict:
    return {"mean_value": float(np.mean(y_train))}


def predict_mean_predictor(model: dict, n: int) -> np.ndarray:
    return np.full(
        shape=(n,),
        fill_value=model["mean_value"],
        dtype=float,
    )


def run_one_feature_set(
    feature_set_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    results = []
    val_preds = {}
    test_preds = {}

    n_features = int(X_train.shape[1])

    # Ridge
    ridge_model, ridge_params = fit_ridge(X_train, y_train, sample_weight)
    ridge_val_pred = ridge_model.predict(X_val)
    ridge_test_pred = ridge_model.predict(X_test)

    model_name = f"{feature_set_name}-Ridge"
    results.append({
        "feature_set": feature_set_name,
        "model": model_name,
        "split": "val",
        "n_features": n_features,
        **evaluate_regression(y_val, ridge_val_pred),
        "params": json.dumps(ridge_params, ensure_ascii=False),
    })
    results.append({
        "feature_set": feature_set_name,
        "model": model_name,
        "split": "test",
        "n_features": n_features,
        **evaluate_regression(y_test, ridge_test_pred),
        "params": json.dumps(ridge_params, ensure_ascii=False),
    })
    val_preds[model_name] = ridge_val_pred
    test_preds[model_name] = ridge_test_pred

    # SVR
    svr_model, svr_params = fit_svr(X_train, y_train, sample_weight)
    svr_val_pred = svr_model.predict(X_val)
    svr_test_pred = svr_model.predict(X_test)

    model_name = f"{feature_set_name}-SVR"
    results.append({
        "feature_set": feature_set_name,
        "model": model_name,
        "split": "val",
        "n_features": n_features,
        **evaluate_regression(y_val, svr_val_pred),
        "params": json.dumps(svr_params, ensure_ascii=False),
    })
    results.append({
        "feature_set": feature_set_name,
        "model": model_name,
        "split": "test",
        "n_features": n_features,
        **evaluate_regression(y_test, svr_test_pred),
        "params": json.dumps(svr_params, ensure_ascii=False),
    })
    val_preds[model_name] = svr_val_pred
    test_preds[model_name] = svr_test_pred

    # CatBoost
    if HAS_CATBOOST:
        cb_model, cb_params = fit_catboost(X_train, y_train, sample_weight)
        cb_val_pred = cb_model.predict(X_val)
        cb_test_pred = cb_model.predict(X_test)

        model_name = f"{feature_set_name}-CatBoost"
        results.append({
            "feature_set": feature_set_name,
            "model": model_name,
            "split": "val",
            "n_features": n_features,
            **evaluate_regression(y_val, cb_val_pred),
            "params": json.dumps(cb_params, ensure_ascii=False),
        })
        results.append({
            "feature_set": feature_set_name,
            "model": model_name,
            "split": "test",
            "n_features": n_features,
            **evaluate_regression(y_test, cb_test_pred),
            "params": json.dumps(cb_params, ensure_ascii=False),
        })
        val_preds[model_name] = cb_val_pred
        test_preds[model_name] = cb_test_pred

    return results, val_preds, test_preds


def main() -> None:
    config_path = get_config_path(project_root)
    cfg = load_yaml(config_path)

    seed = int(cfg["split"]["seed"])
    set_seed(seed)

    split_dir = project_root / cfg["paths"]["split_dir"]

    artifacts_dir = project_root / cfg["paths"].get(
        "fair_baseline_artifacts_dir",
        "artifacts/fair_feature_baselines_v1",
    )
    ensure_dir(artifacts_dir)

    train_df = pd.read_csv(split_dir / "train_joint.csv")
    val_df = pd.read_csv(split_dir / "val_main.csv")
    test_df = pd.read_csv(split_dir / "test_main.csv")

    y_train = train_df["label_pIC50"].astype(float).values
    y_val = val_df["label_pIC50"].astype(float).values
    y_test = test_df["label_pIC50"].astype(float).values

    sample_weight = train_df["sample_weight"].astype(float).values

    all_results = []

    val_pred_df = pd.DataFrame({
        "sequence": val_df["sequence"].values,
        "y_true": y_val,
    })
    test_pred_df = pd.DataFrame({
        "sequence": test_df["sequence"].values,
        "y_true": y_test,
    })

    # MeanPredictor
    mean_model = fit_mean_predictor(y_train)
    mean_val_pred = predict_mean_predictor(mean_model, len(y_val))
    mean_test_pred = predict_mean_predictor(mean_model, len(y_test))

    all_results.append({
        "feature_set": "None",
        "model": "MeanPredictor",
        "split": "val",
        "n_features": 0,
        **evaluate_regression(y_val, mean_val_pred),
        "params": json.dumps(mean_model, ensure_ascii=False),
    })
    all_results.append({
        "feature_set": "None",
        "model": "MeanPredictor",
        "split": "test",
        "n_features": 0,
        **evaluate_regression(y_test, mean_test_pred),
        "params": json.dumps(mean_model, ensure_ascii=False),
    })

    val_pred_df["MeanPredictor"] = mean_val_pred
    test_pred_df["MeanPredictor"] = mean_test_pred

    feature_builders = {
        "SeqOnly": build_seq_identity_frame,
        "PhysChemOnly": build_physchem_only_frame,
        "Descriptor": build_descriptor_full_frame,
        "A1FlatFull": build_a1flat_full_frame,
    }

    for feature_set_name, builder in feature_builders.items():
        print(f"\nBuilding features: {feature_set_name}")

        X_train = builder(train_df, sequence_col="sequence")
        X_val = builder(val_df, sequence_col="sequence")
        X_test = builder(test_df, sequence_col="sequence")

        # 保证列顺序一致
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_val shape:   {X_val.shape}")
        print(f"  X_test shape:  {X_test.shape}")

        results, val_preds, test_preds = run_one_feature_set(
            feature_set_name=feature_set_name,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            sample_weight=sample_weight,
        )

        all_results.extend(results)

        for name, pred in val_preds.items():
            val_pred_df[name] = pred
        for name, pred in test_preds.items():
            test_pred_df[name] = pred

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        artifacts_dir / "fair_feature_baseline_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    val_pred_df.to_csv(
        artifacts_dir / "fair_feature_val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_pred_df.to_csv(
        artifacts_dir / "fair_feature_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_summary = (
        results_df[results_df["split"] == "test"]
        .sort_values(["rmse", "mae"], ascending=[True, True])
    )

    test_summary.to_csv(
        artifacts_dir / "fair_feature_test_sorted.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDone.")
    print(f"Saved to: {artifacts_dir}")
    print("\nTest ranking:")
    print(test_summary[[
        "model",
        "feature_set",
        "n_features",
        "rmse",
        "mae",
        "spearman",
    ]])


if __name__ == "__main__":
    main()