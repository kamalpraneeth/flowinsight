"""Anomaly detection comparing Isolation Forest against statistical Robust MAD per design-size bucket.

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
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from flowinsight.features import build_feature_pipeline
from flowinsight.parsers import parse_log_file


class RobustMADDetector:
    """Statistical baseline anomaly detector using Median Absolute Deviation (MAD) partitioned by design-size buckets."""

    def __init__(self, threshold_z: float = 3.0) -> None:
        self.threshold_z = threshold_z
        self.bucket_stats_: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> "RobustMADDetector":
        # Partition into 4 gate count buckets
        df_work = df.copy()
        df_work["size_bucket"] = pd.qcut(
            df_work["design_size"], q=4, labels=["tier_1", "tier_2", "tier_3", "tier_4"]
        )

        for bucket in ["tier_1", "tier_2", "tier_3", "tier_4"]:
            subset = df_work[df_work["size_bucket"] == bucket]
            if len(subset) == 0:
                continue

            med_rt = float(np.median(subset["runtime_s"]))
            mad_rt = float(stats.median_abs_deviation(subset["runtime_s"], scale="normal")) or 1.0

            med_mem = float(np.median(subset["peak_memory_mb"]))
            mad_mem = (
                float(stats.median_abs_deviation(subset["peak_memory_mb"], scale="normal")) or 1.0
            )

            med_warn = float(np.median(subset["warnings_count"]))
            mad_warn = (
                float(stats.median_abs_deviation(subset["warnings_count"], scale="normal")) or 1.0
            )

            self.bucket_stats_[str(bucket)] = {
                "med_rt": med_rt,
                "mad_rt": mad_rt,
                "med_mem": med_mem,
                "mad_mem": mad_mem,
                "med_warn": med_warn,
                "mad_warn": mad_warn,
            }

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns 1 for anomaly, 0 for normal."""
        df_work = df.copy()
        try:
            df_work["size_bucket"] = pd.qcut(
                df_work["design_size"], q=4, labels=["tier_1", "tier_2", "tier_3", "tier_4"]
            )
        except Exception:
            df_work["size_bucket"] = "tier_2"

        predictions = []
        for _, row in df_work.iterrows():
            b = str(row.get("size_bucket", "tier_2"))
            if b not in self.bucket_stats_:
                b = (
                    "tier_2"
                    if "tier_2" in self.bucket_stats_
                    else list(self.bucket_stats_.keys())[0]
                )

            s = self.bucket_stats_[b]
            z_rt = abs(row["runtime_s"] - s["med_rt"]) / s["mad_rt"]
            z_mem = abs(row["peak_memory_mb"] - s["med_mem"]) / s["mad_mem"]
            z_warn = abs(row["warnings_count"] - s["med_warn"]) / s["mad_warn"]

            is_anom = (
                1
                if (
                    z_rt > self.threshold_z or z_mem > self.threshold_z or z_warn > self.threshold_z
                )
                else 0
            )
            predictions.append(is_anom)

        return np.array(predictions)


class AnomalyDetectionBundle:
    """Encapsulates Isolation Forest, Robust MAD statistical detector, and evaluation records."""

    def __init__(
        self,
        feature_pipeline: Any,
        isolation_forest: Any,
        robust_mad: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.feature_pipeline = feature_pipeline
        self.isolation_forest = isolation_forest
        self.robust_mad = robust_mad
        self.metadata = metadata or {}

    def detect(self, data: Union[str, Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """Detect anomalies via Isolation Forest and Robust MAD. Accepts log file or record."""
        if isinstance(data, str) and os.path.exists(data):
            rec = parse_log_file(data)
            df = [rec]
            # Construct tabular representation
            tab_df = pd.DataFrame(
                [
                    {
                        "design_size": rec.design_size or 100_000,
                        "runtime_s": rec.runtime_s,
                        "peak_memory_mb": rec.peak_memory_mb,
                        "warnings_count": rec.warnings_count,
                    }
                ]
            )
        elif isinstance(data, dict):
            df = [data]
            tab_df = pd.DataFrame([data])
        else:
            df = data
            tab_df = data

        X_trans = self.feature_pipeline.transform(df)
        if_score = float(-self.isolation_forest.score_samples(X_trans)[0])
        # In scikit-learn, -1 is anomaly, 1 is inlier
        if_pred = 1 if self.isolation_forest.predict(X_trans)[0] == -1 else 0

        mad_pred = int(self.robust_mad.predict(tab_df)[0])

        evidence: List[str] = []
        if if_pred == 1 or mad_pred == 1:
            row = tab_df.iloc[0]
            if row.get("runtime_s", 0) > 600:
                evidence.append(
                    f"Runtime ({row['runtime_s']}s) substantially exceeds normal scaling expectation."
                )
            if row.get("peak_memory_mb", 0) > 3500:
                evidence.append(
                    f"Peak memory ({row['peak_memory_mb']}MB) displays abnormal heap growth."
                )
            if row.get("warnings_count", 0) > 100:
                evidence.append(
                    f"Warning count ({row['warnings_count']}) indicates a message cascade anomaly."
                )
            if not evidence:
                evidence.append(
                    "Multi-dimensional feature vector lies outside high-density kernel manifold."
                )

        return {
            "is_anomaly": 1 if (if_pred == 1 or mad_pred == 1) else 0,
            "isolation_forest_flag": if_pred,
            "isolation_forest_anomaly_score": round(if_score, 4),
            "robust_mad_flag": mad_pred,
            "evidence": evidence,
            "data_source": "synthetic",
        }


def get_git_commit_hash() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown-commit"


def train_anomaly_detectors(
    df: pd.DataFrame,
    contamination: float = 0.08,
    seed: int = 42,
    output_dir: str = "models_saved",
) -> Tuple[AnomalyDetectionBundle, Dict[str, Any]]:
    """Train Isolation Forest and Robust MAD, evaluate against ground-truth anomalies, and report honest metrics."""
    feature_pipeline = build_feature_pipeline(include_text=False, include_msg_ids=True)
    X = feature_pipeline.fit_transform(df)
    y_ground_truth = df["is_anomaly"].to_numpy()

    # 1. Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=100,
    )
    iso_forest.fit(X)
    # Convert (-1: anomaly, 1: normal) -> (1: anomaly, 0: normal)
    iso_pred_raw = iso_forest.predict(X)
    iso_pred = (iso_pred_raw == -1).astype(int)

    iso_precision = float(precision_score(y_ground_truth, iso_pred, zero_division=0))
    iso_recall = float(recall_score(y_ground_truth, iso_pred, zero_division=0))
    iso_f1 = float(f1_score(y_ground_truth, iso_pred, zero_division=0))

    # 2. Robust MAD statistical detector
    mad_detector = RobustMADDetector(threshold_z=2.8).fit(df)
    mad_pred = mad_detector.predict(df)

    mad_precision = float(precision_score(y_ground_truth, mad_pred, zero_division=0))
    mad_recall = float(recall_score(y_ground_truth, mad_pred, zero_division=0))
    mad_f1 = float(f1_score(y_ground_truth, mad_pred, zero_division=0))

    # Identify where they miss (False Negatives) and false positives
    iso_false_neg = (
        df[(y_ground_truth == 1) & (iso_pred == 0)][
            ["job_id", "anomaly_type", "runtime_s", "peak_memory_mb"]
        ]
        .head(5)
        .to_dict(orient="records")
    )
    iso_false_pos = (
        df[(y_ground_truth == 0) & (iso_pred == 1)][
            ["job_id", "design_size", "runtime_s", "peak_memory_mb"]
        ]
        .head(5)
        .to_dict(orient="records")
    )

    mad_false_neg = (
        df[(y_ground_truth == 1) & (mad_pred == 0)][
            ["job_id", "anomaly_type", "runtime_s", "peak_memory_mb"]
        ]
        .head(5)
        .to_dict(orient="records")
    )
    mad_false_pos = (
        df[(y_ground_truth == 0) & (mad_pred == 1)][
            ["job_id", "design_size", "runtime_s", "peak_memory_mb"]
        ]
        .head(5)
        .to_dict(orient="records")
    )

    metrics = {
        "isolation_forest": {
            "precision": round(iso_precision, 4),
            "recall": round(iso_recall, 4),
            "f1_score": round(iso_f1, 4),
            "missed_cases_fn_sample": iso_false_neg,
            "false_positives_fp_sample": iso_false_pos,
        },
        "robust_mad_statistical": {
            "precision": round(mad_precision, 4),
            "recall": round(mad_recall, 4),
            "f1_score": round(mad_f1, 4),
            "missed_cases_fn_sample": mad_false_neg,
            "false_positives_fp_sample": mad_false_pos,
        },
    }

    metadata = {
        "model_name": "anomaly_detection_suite",
        "task": "anomaly_detection",
        "methods_evaluated": ["isolation_forest", "robust_mad_statistical"],
        "date_trained": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit_hash(),
        "random_seed": seed,
        "dataset_size": len(df),
        "ground_truth_anomaly_rate": round(float(np.mean(y_ground_truth)), 4),
        "metrics": metrics,
        "limitations": [
            "Evaluated against synthetically injected runtime spikes, memory growth, and warning storms.",
            "Subtle compounding micro-anomalies may not trip either threshold.",
            "Contamination parameter requires tuning per cluster deployment.",
        ],
        "data_source": "synthetic",
    }

    bundle = AnomalyDetectionBundle(
        feature_pipeline=feature_pipeline,
        isolation_forest=iso_forest,
        robust_mad=mad_detector,
        metadata=metadata,
    )

    os.makedirs(output_dir, exist_ok=True)
    bundle_path = os.path.join(output_dir, "anomaly_model.joblib")
    meta_path = os.path.join(output_dir, "anomaly_metadata.json")

    joblib.dump(bundle, bundle_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return bundle, metadata
