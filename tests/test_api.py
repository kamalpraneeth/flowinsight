"""Unit tests for Phase 5 FastAPI endpoints.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

import os
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient

import flowinsight.api as api_module
from flowinsight.generator import SyntheticDataGenerator
from flowinsight.models.anomaly import train_anomaly_detectors
from flowinsight.models.failure import train_failure_classifier
from flowinsight.models.runtime import train_runtime_regressor
from flowinsight.store import load_manifest_and_logs


@pytest.fixture(scope="module")
def api_test_app():
    tmp_dir = tempfile.mkdtemp()
    gen = SyntheticDataGenerator(seed=42)
    jobs = gen.generate_dataset(num_jobs=50, output_dir=tmp_dir, write_logs=True)

    db_path = os.path.join(tmp_dir, "api_test.db")
    manifest_path = os.path.join(tmp_dir, "manifest.csv")
    load_manifest_and_logs(db_path, manifest_path, parse_logs=True, max_log_parse=50)

    # Train models
    models_dir = os.path.join(tmp_dir, "models")
    import pandas as pd

    df = pd.read_csv(manifest_path)
    train_runtime_regressor(df, seed=42, output_dir=models_dir)
    train_failure_classifier(df, seed=42, output_dir=models_dir)
    train_anomaly_detectors(df, seed=42, output_dir=models_dir)

    # Override module globals for test fixture
    orig_db = api_module.DB_PATH
    orig_models = api_module.MODELS_DIR
    api_module.DB_PATH = db_path
    api_module.MODELS_DIR = models_dir

    client = TestClient(api_module.app)
    yield client, jobs[0].job_id

    # Restore globals
    api_module.DB_PATH = orig_db
    api_module.MODELS_DIR = orig_models
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_api_health(api_test_app):
    client, _ = api_test_app
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["data_source"] == "synthetic"
    assert "Synthetic data" in data["banner"]


def test_api_metrics(api_test_app):
    client, _ = api_test_app
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "total_jobs" in data
    assert data["total_jobs"] >= 50
    assert data["data_source"] == "synthetic"


def test_api_jobs_pagination_and_filters(api_test_app):
    client, _ = api_test_app
    # Test filtering and pagination
    res = client.get("/jobs?stage=synthesis&page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert len(data["items"]) <= 10
    assert data["page"] == 1
    assert data["data_source"] == "synthetic"


def test_api_job_detail(api_test_app):
    client, sample_job_id = api_test_app
    res = client.get(f"/jobs/{sample_job_id}")
    assert res.status_code == 200
    data = res.json()
    assert "job" in data
    assert data["job"]["job_id"] == sample_job_id
    assert "steps" in data
    assert "messages" in data
    assert "log_lines" in data
    assert data["data_source"] == "synthetic"


def test_api_stats_endpoints(api_test_app):
    client, _ = api_test_app
    # 1. Overview
    r1 = client.get("/stats/overview")
    assert r1.status_code == 200
    d1 = r1.json()
    assert "total_jobs" in d1
    assert "failure_rate_pct" in d1
    assert d1["data_source"] == "synthetic"

    # 2. Failures
    r2 = client.get("/stats/failures")
    assert r2.status_code == 200
    d2 = r2.json()
    assert "failure_causes" in d2
    assert d2["data_source"] == "synthetic"

    # 3. Trends
    r3 = client.get("/stats/trends")
    assert r3.status_code == 200
    d3 = r3.json()
    assert "weekly_failure_rates" in d3
    assert d3["data_source"] == "synthetic"

    # 4. Anomalies
    r4 = client.get("/stats/anomalies")
    assert r4.status_code == 200
    d4 = r4.json()
    assert "type_breakdown" in d4
    assert d4["data_source"] == "synthetic"


def test_api_models_endpoint(api_test_app):
    client, _ = api_test_app
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert data["count"] >= 3
    assert data["data_source"] == "synthetic"


def test_api_predict_and_classify_endpoints(api_test_app):
    client, _ = api_test_app

    # Runtime prediction
    p_req = {
        "design_size": 200000,
        "stage": "synthesis",
        "threads": 4,
        "clock_period_ns": 1.2,
        "effort": "medium",
        "target": "timing",
    }
    r_pred = client.post("/predict/runtime", json=p_req)
    assert r_pred.status_code == 200
    d_pred = r_pred.json()
    assert "prediction" in d_pred
    assert "prediction_interval_80" in d_pred
    assert d_pred["data_source"] == "synthetic"

    # Failure classification
    f_req = {
        "log_text": "ERROR: [MEM-902] Fatal memory allocation failure: requested 4096MB exceeds cgroup limits.",
        "stage": "synthesis",
    }
    r_fail = client.post("/classify/failure", json=f_req)
    assert r_fail.status_code == 200
    d_fail = r_fail.json()
    assert "top_causes" in d_fail
    assert len(d_fail["top_causes"]) <= 3
    assert d_fail["data_source"] == "synthetic"

    # Anomaly detection
    a_req = {
        "design_size": 100000,
        "runtime_s": 950.0,
        "peak_memory_mb": 4200.0,
        "warnings_count": 350,
    }
    r_anom = client.post("/detect/anomaly", json=a_req)
    assert r_anom.status_code == 200
    d_anom = r_anom.json()
    assert "is_anomaly" in d_anom
    assert "evidence" in d_anom
    assert d_anom["data_source"] == "synthetic"
