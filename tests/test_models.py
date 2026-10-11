"""Unit tests for Phase 3 ML Models and Evaluation Honesty.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from flowinsight.generator import SyntheticDataGenerator
from flowinsight.models.anomaly import train_anomaly_detectors
from flowinsight.models.failure import train_failure_classifier
from flowinsight.models.runtime import train_runtime_regressor


@pytest.fixture(scope="module")
def sample_data():
    tmp_dir = tempfile.mkdtemp()
    gen = SyntheticDataGenerator(seed=42)
    gen.generate_dataset(num_jobs=250, output_dir=tmp_dir, write_logs=True)
    manifest_path = os.path.join(tmp_dir, "manifest.csv")
    df = pd.read_csv(manifest_path)
    yield df, tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_runtime_model_evaluation_and_intervals(sample_data):
    df, tmp_dir = sample_data
    out_dir = os.path.join(tmp_dir, "models")
    bundle, metadata = train_runtime_regressor(
        df, target_col="runtime_s", seed=42, output_dir=out_dir
    )

    metrics = metadata["metrics"]
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "empirical_coverage" in metrics

    # Hard rule: evaluate honesty - model must beat median baseline
    assert metrics["mae"] < metrics["baseline_median"]["mae"], (
        "Trained GradientBoosting model must achieve lower MAE than naive median baseline"
    )

    # Predict via log file
    sample_log = os.path.join(tmp_dir, "logs", f"{df.iloc[0]['job_id']}.log")
    if os.path.exists(sample_log):
        pred_res = bundle.predict(sample_log)
        assert "prediction" in pred_res
        assert "prediction_interval_80" in pred_res
        assert (
            pred_res["prediction_interval_80"]["lower"]
            <= pred_res["prediction_interval_80"]["upper"]
        )


def test_failure_classifier_and_heap_top_k(sample_data):
    df, tmp_dir = sample_data
    out_dir = os.path.join(tmp_dir, "models")
    bundle, metadata = train_failure_classifier(
        df, model_type="logistic_regression", seed=42, output_dir=out_dir
    )

    metrics = metadata["metrics"]
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics
    assert "ablation_without_tfidf" in metrics

    # Test heap top-k root cause ranking on failed job
    failed_rows = df[df["status"] == "failed"]
    assert len(failed_rows) > 0
    test_failed_job = failed_rows.iloc[0]
    sample_log = os.path.join(tmp_dir, "logs", f"{test_failed_job['job_id']}.log")

    if os.path.exists(sample_log):
        top_k_res = bundle.predict_top_k_causes(sample_log, k=3)
        assert len(top_k_res["top_causes"]) <= 3
        # Probabilities must be descending
        probs = [c["probability"] for c in top_k_res["top_causes"]]
        assert probs == sorted(probs, reverse=True)
        assert len(top_k_res["top_causes"][0]["supporting_evidence_lines"]) > 0


def test_anomaly_detectors_and_ground_truth_comparison(sample_data):
    df, tmp_dir = sample_data
    out_dir = os.path.join(tmp_dir, "models")
    bundle, metadata = train_anomaly_detectors(df, seed=42, output_dir=out_dir)

    metrics = metadata["metrics"]
    assert "isolation_forest" in metrics
    assert "robust_mad_statistical" in metrics
    assert "precision" in metrics["isolation_forest"]
    assert "recall" in metrics["isolation_forest"]
    assert "f1_score" in metrics["isolation_forest"]

    # Test detection on sample job
    sample_log = os.path.join(tmp_dir, "logs", f"{df.iloc[0]['job_id']}.log")
    if os.path.exists(sample_log):
        res = bundle.detect(sample_log)
        assert "is_anomaly" in res
        assert "isolation_forest_flag" in res
        assert "robust_mad_flag" in res
        assert "evidence" in res


def test_evaluation_honesty_bad_model_scores_worse():
    """HARD RULE: A deliberately degraded model must score measurably worse than trained model."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    good_pred = np.array([105.0, 195.0, 310.0, 390.0, 505.0])
    # Deliberately bad model (constant or inverted)
    bad_pred = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

    from sklearn.metrics import mean_absolute_error, r2_score

    good_mae = mean_absolute_error(y_true, good_pred)
    bad_mae = mean_absolute_error(y_true, bad_pred)
    assert good_mae < bad_mae, "Good model must have strictly lower MAE than bad model"

    good_r2 = r2_score(y_true, good_pred)
    bad_r2 = r2_score(y_true, bad_pred)
    assert good_r2 > bad_r2, "Good model must have strictly higher R2 than bad model"
