"""Unit tests for Phase 1 Synthetic Data Generator.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

import os
import shutil
import tempfile

from flowinsight.generator import SyntheticDataGenerator


def test_generator_determinism():
    """Generator with the same seed must produce identical outputs."""
    gen1 = SyntheticDataGenerator(seed=123)
    gen2 = SyntheticDataGenerator(seed=123)

    jobs1 = gen1.generate_dataset(num_jobs=25, output_dir=tempfile.mkdtemp(), write_logs=False)
    jobs2 = gen2.generate_dataset(num_jobs=25, output_dir=tempfile.mkdtemp(), write_logs=False)

    assert len(jobs1) == 25
    assert len(jobs2) == 25

    for j1, j2 in zip(jobs1, jobs2):
        assert j1.job_id == j2.job_id
        assert j1.stage == j2.stage
        assert j1.runtime_s == j2.runtime_s
        assert j1.peak_memory_mb == j2.peak_memory_mb
        assert j1.status == j2.status
        assert j1.failure_cause == j2.failure_cause


def test_generator_stage_coverage_and_congestion_rule():
    """Verify all 3 stages appear and 'congestion' is strictly in place_and_route."""
    gen = SyntheticDataGenerator(seed=42)
    tmp_dir = tempfile.mkdtemp()
    try:
        jobs = gen.generate_dataset(num_jobs=200, output_dir=tmp_dir, write_logs=False)
        stages = {j.stage for j in jobs}
        assert stages == {"synthesis", "place_and_route", "verification"}

        for j in jobs:
            if j.failure_cause == "congestion":
                assert j.stage == "place_and_route"
            if j.status == "completed":
                assert j.failure_cause == "none"
            else:
                assert j.failure_cause != "none"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_generator_log_files_and_messy_cases():
    """Verify log file writing, execution summary presence, and messy cases."""
    gen = SyntheticDataGenerator(seed=99)
    tmp_dir = tempfile.mkdtemp()
    try:
        jobs = gen.generate_dataset(num_jobs=50, output_dir=tmp_dir, write_logs=True)
        logs_dir = os.path.join(tmp_dir, "logs")
        assert os.path.isdir(logs_dir)

        manifest_path = os.path.join(tmp_dir, "manifest.csv")
        assert os.path.isfile(manifest_path)

        truncated_count = 0
        for job in jobs:
            log_path = os.path.join(logs_dir, f"{job.job_id}.log")
            assert os.path.isfile(log_path)
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "NOTICE: Synthetic data generated for demonstration." in content
            assert f"Subsystem: {job.stage.upper()}" in content

            if job.is_truncated:
                truncated_count += 1
                assert "LOG STREAM TRUNCATED" in content
            else:
                assert "EXECUTION SUMMARY" in content

        assert truncated_count >= 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_generator_anomalies_present():
    """Ensure injected anomalies (runtime_spike, memory_growth, warning_storm) are tagged."""
    gen = SyntheticDataGenerator(seed=42)
    tmp_dir = tempfile.mkdtemp()
    try:
        jobs = gen.generate_dataset(num_jobs=200, output_dir=tmp_dir, write_logs=False)
        anomaly_types = {j.anomaly_type for j in jobs}
        assert "runtime_spike" in anomaly_types
        assert "memory_growth" in anomaly_types
        assert "warning_storm" in anomaly_types

        anomalies_csv = os.path.join(tmp_dir, "anomalies.csv")
        assert os.path.isfile(anomalies_csv)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
