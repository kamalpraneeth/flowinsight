"""Runtime and memory regression modeling with temporal splitting and prediction intervals.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from flowinsight.features import build_feature_pipeline
from flowinsight.parsers import parse_log_file


class DummyMedianRegressor(BaseEstimator, RegressorMixin):
    """Simple baseline regressor predicting the median training value."""

    def fit(self, X: Any, y: np.ndarray) -> "DummyMedianRegressor":
        self.median_ = float(np.median(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = len(X) if hasattr(X, "__len__") else 1
        return np.full(n_samples, self.median_)


class RuntimePredictionBundle:
    """Encapsulates preprocessor, main regressor, and quantile bounds models."""

    def __init__(
        self,
        feature_pipeline: Any,
        regressor: Any,
        lower_regressor: Optional[Any] = None,
        upper_regressor: Optional[Any] = None,
        target_name: str = "runtime_s",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.feature_pipeline = feature_pipeline
        self.regressor = regressor
        self.lower_regressor = lower_regressor
        self.upper_regressor = upper_regressor
        self.target_name = target_name
        self.metadata = metadata or {}

    def predict(self, data: Union[str, Dict[str, Any], pd.DataFrame, List[Any]]) -> Dict[str, Any]:
        """Predict target value and prediction interval. Accepts raw log path, dict, or df."""
        if isinstance(data, str) and os.path.exists(data):
            parsed = parse_log_file(data)
            df = [parsed]
        elif isinstance(data, dict):
            df = [data]
        else:
            df = data

        X_trans = self.feature_pipeline.transform(df)
        preds = self.regressor.predict(X_trans)
        point_pred = float(preds[0]) if len(preds) == 1 else [float(p) for p in preds]

        result: Dict[str, Any] = {
            "prediction": round(point_pred if isinstance(point_pred, float) else point_pred[0], 2),
            "target": self.target_name,
            "data_source": "synthetic",
        }

        if self.lower_regressor and self.upper_regressor:
            p_low = float(self.lower_regressor.predict(X_trans)[0])
            p_high = float(self.upper_regressor.predict(X_trans)[0])
            result["prediction_interval_80"] = {
                "lower": round(max(0.0, p_low), 2),
                "upper": round(p_high, 2),
                "nominal_coverage": 0.80,
                "empirical_coverage": self.metadata.get("metrics", {}).get(
                    "empirical_coverage", 0.0
                ),
            }

        return result


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        commit = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return commit
    except Exception:
        return "unknown-commit"


def train_runtime_regressor(
    df: pd.DataFrame,
    target_col: str = "runtime_s",
    model_type: str = "gradient_boosting",
    test_ratio: float = 0.20,
    seed: int = 42,
    output_dir: str = "models_saved",
) -> Tuple[RuntimePredictionBundle, Dict[str, Any]]:
    """Train regression model with strictly temporal split, compute dynamic metrics, and persist."""
    # Ensure temporal ordering to prevent data leakage
    df_sorted = df.sort_values("start_time").reset_index(drop=True)

    split_idx = int(len(df_sorted) * (1.0 - test_ratio))
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    # Fit feature pipeline strictly on train data
    feature_pipeline = build_feature_pipeline(include_text=False, include_msg_ids=False)
    X_train = feature_pipeline.fit_transform(train_df)
    y_train = train_df[target_col].to_numpy()

    X_test = feature_pipeline.transform(test_df)
    y_test = test_df[target_col].to_numpy()

    # Baseline: Median Regressor and Linear Regression
    baseline_median = DummyMedianRegressor().fit(X_train, y_train)
    y_pred_base = baseline_median.predict(X_test)
    base_mae = float(mean_absolute_error(y_test, y_pred_base))
    base_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_base)))
    base_r2 = float(r2_score(y_test, y_pred_base))

    lin_reg = LinearRegression().fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    lin_mae = float(mean_absolute_error(y_test, y_pred_lin))
    lin_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_lin)))
    lin_r2 = float(r2_score(y_test, y_pred_lin))

    # Main regressor
    if model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=seed, learning_rate=0.08
        )
        # Quantile regressors for 80% interval (alpha 0.10 and 0.90)
        q_low = GradientBoostingRegressor(
            loss="quantile", alpha=0.10, n_estimators=60, max_depth=3, random_state=seed
        )
        q_high = GradientBoostingRegressor(
            loss="quantile", alpha=0.90, n_estimators=60, max_depth=3, random_state=seed
        )
    else:
        model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=seed)
        q_low = None
        q_high = None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    # Interval coverage calculation
    empirical_coverage = 0.0
    if q_low is not None and q_high is not None:
        q_low.fit(X_train, y_train)
        q_high.fit(X_train, y_train)
        y_low = q_low.predict(X_test)
        y_high = q_high.predict(X_test)
        covered = (y_test >= y_low) & (y_test <= y_high)
        empirical_coverage = float(np.mean(covered))

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "empirical_coverage": round(empirical_coverage, 4),
        "nominal_coverage": 0.80,
        "baseline_median": {
            "mae": round(base_mae, 3),
            "rmse": round(base_rmse, 3),
            "r2": round(base_r2, 4),
        },
        "baseline_linear": {
            "mae": round(lin_mae, 3),
            "rmse": round(lin_rmse, 3),
            "r2": round(lin_r2, 4),
        },
    }

    metadata = {
        "model_name": f"{target_col}_{model_type}",
        "target": target_col,
        "model_type": model_type,
        "date_trained": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit_hash(),
        "random_seed": seed,
        "dataset_size": len(df),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "split_method": "temporal_start_time_split",
        "metrics": metrics,
        "limitations": [
            "Trained exclusively on synthetic EDA scaling laws.",
            "Prediction intervals assume stationary execution environments.",
            "Extreme abnormal runs may violate coverage bounds.",
        ],
        "data_source": "synthetic",
    }

    bundle = RuntimePredictionBundle(
        feature_pipeline=feature_pipeline,
        regressor=model,
        lower_regressor=q_low,
        upper_regressor=q_high,
        target_name=target_col,
        metadata=metadata,
    )

    # Persist bundle and metadata
    os.makedirs(output_dir, exist_ok=True)
    bundle_path = os.path.join(output_dir, f"{target_col}_model.joblib")
    meta_path = os.path.join(output_dir, f"{target_col}_metadata.json")

    joblib.dump(bundle, bundle_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return bundle, metadata
