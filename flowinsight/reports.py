"""Report generation and Matplotlib visual analytics.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flowinsight.trends import calculate_weekly_failure_rate, detect_change_point_alerts


def plot_runtime_vs_design_size(df: pd.DataFrame, output_path: str) -> str:
    """Generate scatter plot of runtime vs gate count partitioned by stage."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    stages = df["stage"].unique()
    colors = {"synthesis": "#2563eb", "place_and_route": "#d97706", "verification": "#059669"}

    for stg in stages:
        subset = df[(df["stage"] == stg) & (df["status"] == "completed")]
        ax.scatter(
            subset["design_size"] / 1000.0,
            subset["runtime_s"],
            alpha=0.6,
            label=stg,
            color=colors.get(stg, "#64748b"),
            edgecolors="none",
            s=35,
        )

    ax.set_title("Runtime vs. Design Gate Count by Stage", fontsize=12, fontweight="bold")
    ax.set_xlabel("Design Scale (kGates)", fontsize=10)
    ax.set_ylabel("Wallclock Runtime (seconds)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_failure_cause_distribution(df: pd.DataFrame, output_path: str) -> str:
    """Generate bar chart displaying distribution of root-cause failure categories."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    failed = df[df["status"] == "failed"]
    causes = failed["failure_cause"].value_counts()

    y_pos = np.arange(len(causes))
    ax.barh(y_pos, causes.values, color="#e11d48", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(causes.index)
    ax.invert_yaxis()

    ax.set_title("Distribution of Failure Root Causes", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency of Failure Incidents", fontsize=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    for i, v in enumerate(causes.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_anomaly_timeline(df: pd.DataFrame, output_path: str) -> str:
    """Plot runtime timeline highlighting flagged anomalous jobs."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["start_time"]):
        df_work["start_time"] = pd.to_datetime(df_work["start_time"])
    df_sorted = df_work.sort_values("start_time")

    normal = df_sorted[df_sorted["is_anomaly"] == 0]
    anom = df_sorted[df_sorted["is_anomaly"] == 1]

    ax.scatter(
        normal["start_time"],
        normal["runtime_s"],
        alpha=0.4,
        color="#0284c7",
        s=20,
        label="Nominal Runs",
    )
    ax.scatter(
        anom["start_time"],
        anom["runtime_s"],
        color="#dc2626",
        s=50,
        edgecolors="black",
        label="Flagged Anomalies",
        zorder=5,
    )

    ax.set_title("Execution Timeline & Injected Anomalies", fontsize=12, fontweight="bold")
    ax.set_xlabel("Execution Date", fontsize=10)
    ax.set_ylabel("Runtime (s)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_trend_lines(df: pd.DataFrame, output_path: str) -> str:
    """Plot weekly failure rate trend line."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    weekly = calculate_weekly_failure_rate(df)
    fig, ax = plt.subplots(figsize=(8, 4))

    weeks = [w["week"] for w in weekly]
    rates = [w["failure_rate_pct"] for w in weekly]

    ax.plot(weeks, rates, marker="o", color="#7c3aed", linewidth=2)
    ax.set_title("Weekly Failure Rate Trend (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calendar Week", fontsize=10)
    ax.set_ylabel("Failure Rate (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_markdown_report(
    df: pd.DataFrame,
    output_file: str = "reports/nightly_report.md",
    chart_dir: str = "reports/charts",
) -> str:
    """Generate end-to-end analytical summary Markdown report with charts and real metrics."""
    os.makedirs(chart_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Generate charts
    p_rt = plot_runtime_vs_design_size(df, os.path.join(chart_dir, "runtime_vs_size.png"))
    p_fc = plot_failure_cause_distribution(df, os.path.join(chart_dir, "failure_causes.png"))
    p_an = plot_anomaly_timeline(df, os.path.join(chart_dir, "anomaly_timeline.png"))
    p_tr = plot_trend_lines(df, os.path.join(chart_dir, "weekly_trends.png"))

    # Compute genuine summary statistics
    total_jobs = len(df)
    failed_jobs = int((df["status"] == "failed").sum())
    overall_failure_rate = round((failed_jobs / total_jobs) * 100.0, 2) if total_jobs > 0 else 0.0

    completed_runs = df[df["status"] == "completed"]
    median_runtime = (
        round(float(completed_runs["runtime_s"].median()), 2) if not completed_runs.empty else 0.0
    )
    p90_runtime = (
        round(float(completed_runs["runtime_s"].quantile(0.90)), 2)
        if not completed_runs.empty
        else 0.0
    )
    median_mem = (
        round(float(completed_runs["peak_memory_mb"].median()), 1)
        if not completed_runs.empty
        else 0.0
    )

    total_anomalies = int(df["is_anomaly"].sum())

    # Stage breakdowns
    stage_breakdown = (
        df.groupby("stage")
        .agg(
            total_runs=("job_id", "count"),
            failed_runs=("status", lambda s: (s == "failed").sum()),
            avg_runtime=("runtime_s", "mean"),
        )
        .reset_index()
    )

    # Change-point alerts
    alerts = detect_change_point_alerts(df)

    lines: List[str] = [
        "# FlowInsight Automated EDA Execution Analytics Report",
        "",
        "> **NOTICE: SYNTHETIC DATA GENERATION**",
        "> Synthetic data generated for demonstration. Not real EDA tool logs.",
        "",
        f"**Generated at:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Dataset Size:** {total_jobs} runs analyzed",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total Tool Executions Analyzed:** {total_jobs}",
        f"- **Overall Job Failure Rate:** {overall_failure_rate}% ({failed_jobs} failed jobs)",
        f"- **Median Completed Wallclock Runtime:** {median_runtime}s (90th Percentile: {p90_runtime}s)",
        f"- **Median Resident Memory Usage:** {median_mem} MB",
        f"- **Flagged Execution Anomalies:** {total_anomalies} runs",
        "",
        "## 2. Breakdown by Design Stage",
        "",
        "| Stage | Total Runs | Failed Runs | Failure Rate (%) | Mean Runtime (s) |",
        "|---|---|---|---|---|",
    ]

    for _, row in stage_breakdown.iterrows():
        stg_rate = round((row["failed_runs"] / row["total_runs"]) * 100.0, 2)
        lines.append(
            f"| `{row['stage']}` | {row['total_runs']} | {row['failed_runs']} | {stg_rate}% | {row['avg_runtime']:.1f}s |"
        )

    lines.extend(
        [
            "",
            "## 3. Visual Analytics Charts",
            "",
            f"- **Physical Scaling Law:** ![Runtime vs Size]({os.path.relpath(p_rt, os.path.dirname(output_file))})",
            f"- **Root Cause Incidents:** ![Failure Causes]({os.path.relpath(p_fc, os.path.dirname(output_file))})",
            f"- **Anomaly Timeline:** ![Anomalies]({os.path.relpath(p_an, os.path.dirname(output_file))})",
            f"- **Weekly Failure Trajectory:** ![Weekly Trends]({os.path.relpath(p_tr, os.path.dirname(output_file))})",
            "",
            "## 4. Statistical Control Alerts",
            "",
        ]
    )

    if alerts:
        lines.append(
            f"Detected **{len(alerts)}** change-point surge events exceeding 2.5 sigma control thresholds:"
        )
        lines.append("")
        lines.append("| Job ID | Timestamp | Alert Type | Rolling Failure Rate | Z-Score |")
        lines.append("|---|---|---|---|---|")
        for alert in alerts[:10]:
            lines.append(
                f"| `{alert['job_id']}` | {alert['timestamp']} | {alert['alert_type']} | {alert['rolling_failure_rate']}% | {alert['z_score']} |"
            )
    else:
        lines.append("No statistical control alerts triggered in the current analysis window.")

    lines.append("")
    report_content = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    return output_file
