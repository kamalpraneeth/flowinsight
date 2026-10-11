"""Tests verifying Evaluation Honesty (Hard Rule 1).

Rules verified:
1. A deliberately bad model must score worse than a good one.
2. A test must fail if any reported evaluation metric is a constant literal.
3. A test must fail if metric values appear as hardcoded literals in frontend source.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

import os
import re

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, r2_score


def test_bad_model_scores_strictly_worse():
    """A deliberately degraded model must score measurably worse than trained model."""
    y_true = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0])
    # Competent model
    y_competent = np.array([52.0, 98.0, 153.0, 196.0, 254.0, 297.0])
    # Deliberately bad model
    y_bad = np.array([999.0, 999.0, 999.0, 999.0, 999.0, 999.0])

    mae_good = mean_absolute_error(y_true, y_competent)
    mae_bad = mean_absolute_error(y_true, y_bad)
    assert mae_good < mae_bad, (
        f"Competent model MAE ({mae_good}) must be strictly less than bad model MAE ({mae_bad})"
    )

    r2_good = r2_score(y_true, y_competent)
    r2_bad = r2_score(y_true, y_bad)
    assert r2_good > r2_bad, (
        f"Competent model R2 ({r2_good}) must be strictly higher than bad model R2 ({r2_bad})"
    )

    # Failure classification comparison
    y_class_true = ["oom", "timeout", "syntax", "oom", "timeout", "syntax"]
    y_class_good = ["oom", "timeout", "syntax", "oom", "timeout", "syntax"]
    y_class_bad = ["license", "license", "license", "license", "license", "license"]

    f1_good = f1_score(y_class_true, y_class_good, average="macro")
    f1_bad = f1_score(y_class_true, y_class_bad, average="macro")
    assert f1_good > f1_bad, (
        "Good classifier macro F1 must be strictly higher than constant bad classifier"
    )


def test_reported_metrics_are_not_constant_literals():
    """Metrics dynamically calculated by train_runtime_regressor must vary with dataset slices."""
    import tempfile

    import pandas as pd

    from flowinsight.generator import SyntheticDataGenerator
    from flowinsight.models.runtime import train_runtime_regressor

    tmp_dir = tempfile.mkdtemp()
    gen = SyntheticDataGenerator(seed=42)
    gen.generate_dataset(num_jobs=80, output_dir=tmp_dir, write_logs=False)
    df = pd.read_csv(os.path.join(tmp_dir, "manifest.csv"))

    # Train on first half
    _, meta1 = train_runtime_regressor(df.iloc[:40], output_dir=os.path.join(tmp_dir, "m1"))
    mae1 = meta1["metrics"]["mae"]

    # Train on full dataset
    _, meta2 = train_runtime_regressor(df, output_dir=os.path.join(tmp_dir, "m2"))
    mae2 = meta2["metrics"]["mae"]

    # If MAE were a constant literal in code (e.g. return 12.5), mae1 would equal mae2 identically
    assert isinstance(mae1, float)
    assert isinstance(mae2, float)
    assert mae1 != mae2, (
        f"Evaluation metric must be computed dynamically from data, not a constant literal: mae1={mae1}, mae2={mae2}"
    )


def test_no_hardcoded_metric_literals_in_frontend_source():
    """Ensure frontend source files do not embed hardcoded evaluation metric constants."""
    frontend_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "src"
    )
    if not os.path.exists(frontend_dir):
        # Frontend not built yet (Stage B) - passes for Stage A
        return

    # Scan .tsx and .ts files for suspicious hardcoded metric patterns like 'f1: 0.94' or 'accuracy: 98%'
    suspicious_patterns = [
        re.compile(r"f1\s*:\s*0\.\d{2,4}", re.IGNORECASE),
        re.compile(r"accuracy\s*:\s*0\.\d{2,4}", re.IGNORECASE),
        re.compile(r"mae\s*:\s*\d+\.\d{2,4}", re.IGNORECASE),
        re.compile(r"rmse\s*:\s*\d+\.\d{2,4}", re.IGNORECASE),
    ]

    violations = []
    for root, _, files in os.walk(frontend_dir):
        for f in files:
            if (
                f.endswith((".ts", ".tsx"))
                and not f.endswith(".test.tsx")
                and not f.endswith(".test.ts")
            ):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    for line_no, line in enumerate(fh, 1):
                        for pat in suspicious_patterns:
                            if pat.search(line):
                                violations.append(f"{path}:{line_no} -> {line.strip()}")

    assert not violations, "Found hardcoded metric literals in frontend source code:\n" + "\n".join(
        violations
    )
