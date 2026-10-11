"""Unit tests for Phase 2 Storage and SQL Analytics.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

import os
import shutil
import tempfile

import pytest

from flowinsight.generator import SyntheticDataGenerator
from flowinsight.store import (
    init_db,
    load_manifest_and_logs,
    run_failure_rate_by_stage_and_week,
    run_most_frequent_errors,
    run_runtime_percentiles_by_stage,
    run_slowest_designs,
    run_top_warning_ids_before_failures,
)


@pytest.fixture(scope="module")
def populated_db():
    tmp_dir = tempfile.mkdtemp()
    gen = SyntheticDataGenerator(seed=42)
    gen.generate_dataset(num_jobs=60, output_dir=tmp_dir, write_logs=True)

    db_path = os.path.join(tmp_dir, "test_flowinsight.db")
    manifest_path = os.path.join(tmp_dir, "manifest.csv")
    load_manifest_and_logs(db_path, manifest_path, parse_logs=True, max_log_parse=60)

    yield db_path
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_database_init_and_counts(populated_db):
    conn = init_db(populated_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM jobs;")
    jobs_count = cur.fetchone()[0]
    assert jobs_count == 60

    cur.execute("SELECT COUNT(*) FROM messages;")
    msgs_count = cur.fetchone()[0]
    assert msgs_count > 0

    cur.execute("SELECT COUNT(*) FROM step_timings;")
    steps_count = cur.fetchone()[0]
    assert steps_count > 0
    conn.close()


def test_query_failure_rate_by_stage_and_week(populated_db):
    rows = run_failure_rate_by_stage_and_week(populated_db)
    assert len(rows) > 0
    first = rows[0]
    assert "stage" in first
    assert "week_bucket" in first
    assert "total_jobs" in first
    assert "failure_rate_pct" in first
    assert isinstance(first["failure_rate_pct"], (float, int))


def test_query_slowest_designs(populated_db):
    rows = run_slowest_designs(populated_db)
    assert len(rows) > 0
    first = rows[0]
    assert "design_name" in first
    assert "avg_runtime_s" in first
    assert first["avg_runtime_s"] > 0


def test_query_most_frequent_errors(populated_db):
    rows = run_most_frequent_errors(populated_db)
    # May have zero or more depending on failure occurrences in the 60 sample
    assert isinstance(rows, list)
    if rows:
        assert "message_id" in rows[0]
        assert "occurrence_count" in rows[0]


def test_query_runtime_percentiles_by_stage(populated_db):
    rows = run_runtime_percentiles_by_stage(populated_db)
    assert len(rows) > 0
    first = rows[0]
    assert "stage" in first
    assert "p50_runtime_s" in first


def test_query_top_warning_ids_before_failures(populated_db):
    rows = run_top_warning_ids_before_failures(populated_db)
    assert isinstance(rows, list)
    if rows:
        assert "message_id" in rows[0]
        assert "warning_occurrences" in rows[0]
