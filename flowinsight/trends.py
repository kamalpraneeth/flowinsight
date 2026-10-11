"""Statistical trend analysis, runtime drift hypothesis testing, and change-point alerts.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def calculate_weekly_failure_rate(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute weekly failure rate partitioned chronologically."""
    df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["start_time"]):
        df_work["start_time"] = pd.to_datetime(df_work["start_time"])

    df_work["week"] = df_work["start_time"].dt.strftime("%Y-W%U")
    grouped = df_work.groupby("week")

    results: List[Dict[str, Any]] = []
    for week_label, group in grouped:
        total = len(group)
        failed = int((group["status"] == "failed").sum())
        rate = round((failed / total) * 100.0, 2) if total > 0 else 0.0

        results.append(
            {
                "week": week_label,
                "total_jobs": total,
                "failed_jobs": failed,
                "failure_rate_pct": rate,
                "data_source": "synthetic",
            }
        )

    return results


def detect_runtime_drift(
    df: pd.DataFrame,
    baseline_version: str,
    candidate_version: str,
    stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate runtime drift between EDA tool releases using Welch's two-sample t-test and confidence intervals."""
    df_work = df.copy()
    if stage:
        df_work = df_work[df_work["stage"] == stage]

    # Completed runs only
    df_comp = df_work[df_work["status"] == "completed"]

    base_runs = (
        df_comp[df_comp["tool_version"] == baseline_version]["runtime_s"].dropna().to_numpy()
    )
    cand_runs = (
        df_comp[df_comp["tool_version"] == candidate_version]["runtime_s"].dropna().to_numpy()
    )

    if len(base_runs) < 2 or len(cand_runs) < 2:
        return {
            "drift_detected": False,
            "baseline_version": baseline_version,
            "candidate_version": candidate_version,
            "message": "Insufficient sample count for statistical two-sample comparison.",
            "data_source": "synthetic",
        }

    n1, n2 = len(base_runs), len(cand_runs)
    m1, m2 = float(np.mean(base_runs)), float(np.mean(cand_runs))
    s1, s2 = float(np.var(base_runs, ddof=1)), float(np.var(cand_runs, ddof=1))

    diff = m2 - m1
    pct_change = round((diff / m1) * 100.0, 2) if m1 != 0 else 0.0

    # Welch's t-test (unequal variances)
    t_stat, p_val = stats.ttest_ind(cand_runs, base_runs, equal_var=False)

    # Welch-Satterthwaite degrees of freedom
    se_diff = np.sqrt(s1 / n1 + s2 / n2)
    df_welch = (s1 / n1 + s2 / n2) ** 2 / (
        ((s1 / n1) ** 2) / (n1 - 1) + ((s2 / n2) ** 2) / (n2 - 1)
    )
    t_crit = stats.t.ppf(0.975, df=df_welch) if df_welch > 1 else 1.96

    ci_lower = round(float(diff - t_crit * se_diff), 2)
    ci_upper = round(float(diff + t_crit * se_diff), 2)

    is_significant = bool(p_val < 0.05)

    return {
        "baseline_version": baseline_version,
        "candidate_version": candidate_version,
        "sample_size_baseline": n1,
        "sample_size_candidate": n2,
        "mean_baseline_s": round(m1, 2),
        "mean_candidate_s": round(m2, 2),
        "mean_delta_s": round(diff, 2),
        "percent_change": pct_change,
        "confidence_interval_95": {
            "lower": ci_lower,
            "upper": ci_upper,
        },
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(p_val), 5),
        "statistically_significant": is_significant,
        "method": "Welch's two-sample unequal-variance t-test",
        "data_source": "synthetic",
    }


def detect_change_point_alerts(
    df: pd.DataFrame,
    window: int = 15,
    threshold_sigmas: float = 2.5,
) -> List[Dict[str, Any]]:
    """Detect abrupt regression spikes or failure surges using rolling z-score control charting."""
    df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["start_time"]):
        df_work["start_time"] = pd.to_datetime(df_work["start_time"])

    df_sorted = df_work.sort_values("start_time").reset_index(drop=True)
    df_sorted["is_failed_num"] = (df_sorted["status"] == "failed").astype(float)

    # Rolling failure rate
    rolling_fail = df_sorted["is_failed_num"].rolling(window=window, min_periods=window).mean()
    roll_mean = rolling_fail.rolling(window=window, min_periods=window).mean()
    roll_std = rolling_fail.rolling(window=window, min_periods=window).std().replace(0, 0.01)

    z_scores = (rolling_fail - roll_mean) / roll_std

    alerts: List[Dict[str, Any]] = []
    for idx in range(len(df_sorted)):
        z = z_scores.iloc[idx]
        if pd.notna(z) and z > threshold_sigmas:
            row = df_sorted.iloc[idx]
            alerts.append(
                {
                    "job_id": row["job_id"],
                    "timestamp": row["start_time"].isoformat(),
                    "alert_type": "failure_rate_surge",
                    "rolling_failure_rate": round(float(rolling_fail.iloc[idx]) * 100.0, 2),
                    "z_score": round(float(z), 2),
                    "threshold_sigmas": threshold_sigmas,
                    "statistical_method": "Rolling Gaussian Z-Score Control Chart",
                    "data_source": "synthetic",
                }
            )

    return alerts
