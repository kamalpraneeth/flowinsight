"""Unit tests for Phase 4 Trends and Reporting.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

import os
import shutil
import tempfile

import pandas as pd
import pytest

from flowinsight.generator import SyntheticDataGenerator
from flowinsight.reports import (
    generate_markdown_report,
    plot_anomaly_timeline,
    plot_failure_cause_distribution,
    plot_runtime_vs_design_size,
    plot_trend_lines,
)
from flowinsight.trends import (
    calculate_weekly_failure_rate,
    detect_change_point_alerts,
    detect_runtime_drift,
)


@pytest.fixture(scope="module")
def trend_dataset():
    tmp_dir = tempfile.mkdtemp()
    gen = SyntheticDataGenerator(seed=42)
    gen.generate_dataset(num_jobs=150, output_dir=tmp_dir, write_logs=False)
    df = pd.read_csv(os.path.join(tmp_dir, "manifest.csv"))
    yield df, tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_calculate_weekly_failure_rate(trend_dataset):
    df, _ = trend_dataset
    weekly = calculate_weekly_failure_rate(df)
    assert len(weekly) > 0
    first = weekly[0]
    assert "week" in first
    assert "total_jobs" in first
    assert "failed_jobs" in first
    assert "failure_rate_pct" in first
    assert first["data_source"] == "synthetic"


def test_detect_runtime_drift(trend_dataset):
    df, _ = trend_dataset
    versions = df["tool_version"].unique()
    assert len(versions) >= 2
    res = detect_runtime_drift(df, versions[0], versions[1])
    assert "baseline_version" in res
    assert "candidate_version" in res
    assert "mean_delta_s" in res
    assert "confidence_interval_95" in res
    assert "p_value" in res
    assert "statistically_significant" in res
    assert res["data_source"] == "synthetic"


def test_detect_change_point_alerts(trend_dataset):
    df, _ = trend_dataset
    alerts = detect_change_point_alerts(df, window=10, threshold_sigmas=2.0)
    assert isinstance(alerts, list)
    if alerts:
        first = alerts[0]
        assert "job_id" in first
        assert "z_score" in first
        assert "statistical_method" in first


def test_report_and_chart_generation(trend_dataset):
    df, tmp_dir = trend_dataset
    rep_dir = os.path.join(tmp_dir, "rep")
    chart_dir = os.path.join(rep_dir, "charts")

    p1 = plot_runtime_vs_design_size(df, os.path.join(chart_dir, "rt.png"))
    assert os.path.isfile(p1) and os.path.getsize(p1) > 1000

    p2 = plot_failure_cause_distribution(df, os.path.join(chart_dir, "fc.png"))
    assert os.path.isfile(p2) and os.path.getsize(p2) > 1000

    p3 = plot_anomaly_timeline(df, os.path.join(chart_dir, "an.png"))
    assert os.path.isfile(p3) and os.path.getsize(p3) > 1000

    p4 = plot_trend_lines(df, os.path.join(chart_dir, "tr.png"))
    assert os.path.isfile(p4) and os.path.getsize(p4) > 1000

    rep_file = os.path.join(rep_dir, "report.md")
    out = generate_markdown_report(df, output_file=rep_file, chart_dir=chart_dir)
    assert os.path.isfile(out)

    with open(out, "r", encoding="utf-8") as f:
        content = f.read()

    assert "NOTICE: SYNTHETIC DATA GENERATION" in content
    assert "Executive Summary" in content
    assert "Breakdown by Design Stage" in content
