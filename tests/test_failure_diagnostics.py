"""Tests for failure classifier diagnostic validation.

Requirements:
1. Labels aligned between manifest and parsed data.
2. Easy-mode accuracy above stated threshold (> 0.85).
3. The classifier beats the dummy baseline (most_frequent & stratified) clearly.
4. Ablation runs and demonstrates that text features provide meaningful lift.
"""

import os
import pandas as pd
import pytest

from flowinsight.generator import SyntheticDataGenerator
from flowinsight.models.failure import train_failure_classifier
from flowinsight.parsers import parse_log_file


def test_labels_aligned_manifest_and_parsed_data(tmp_path):
    """Verify 100% label alignment between manifest and parsed log records on non-truncated logs."""
    gen = SyntheticDataGenerator(seed=123, enable_truncation=False)
    output_dir = str(tmp_path / "data")
    jobs = gen.generate_dataset(num_jobs=120, output_dir=output_dir, write_logs=True)

    manifest_df = pd.read_csv(os.path.join(output_dir, "manifest.csv"))
    failed_jobs = manifest_df[manifest_df["status"] == "failed"]
    assert len(failed_jobs) > 0

    mismatches = 0
    for _, row in failed_jobs.iterrows():
        job_id = row["job_id"]
        true_cause = row["failure_cause"]
        log_path = os.path.join(output_dir, "logs", f"{job_id}.log")
        assert os.path.exists(log_path)
        parsed = parse_log_file(log_path)
        assert parsed.status == "failed"
        if parsed.failure_cause != true_cause:
            mismatches += 1

    assert mismatches == 0, f"Found {mismatches} label mismatches between manifest and parsed logs!"


def test_easy_mode_accuracy_above_stated_threshold(tmp_path):
    """In easy-mode (noise=OFF, truncation=OFF, ambiguity=OFF), classifier accuracy must exceed stated threshold (0.85)."""
    import numpy as np

    gen = SyntheticDataGenerator(
        seed=42,
        enable_noise=False,
        enable_truncation=False,
        enable_ambiguity=False,
    )
    output_dir = str(tmp_path / "easy_data")
    gen.generate_dataset(num_jobs=450, output_dir=output_dir, write_logs=True)

    df = pd.read_csv(os.path.join(output_dir, "manifest.csv"))
    models_dir = str(tmp_path / "easy_models")
    bundle, meta = train_failure_classifier(df, seed=42, output_dir=models_dir)

    cm = np.array(meta["metrics"]["confusion_matrix"]["matrix"])
    accuracy = float(np.trace(cm) / np.sum(cm))
    macro_f1 = meta["metrics"]["macro_f1"]
    assert accuracy >= 0.85, f"Easy-mode Accuracy {accuracy:.4f} fell below threshold 0.85"
    assert macro_f1 >= 0.80, f"Easy-mode Macro F1 {macro_f1:.4f} fell below threshold 0.80"


def test_classifier_beats_dummy_baseline(tmp_path):
    """Classifier macro F1 must clearly exceed DummyClassifier baselines (stratified and most_frequent)."""
    gen = SyntheticDataGenerator(seed=42, enable_truncation=True, enable_ambiguity=True)
    output_dir = str(tmp_path / "data")
    gen.generate_dataset(num_jobs=350, output_dir=output_dir, write_logs=True)

    df = pd.read_csv(os.path.join(output_dir, "manifest.csv"))
    models_dir = str(tmp_path / "models")
    bundle, meta = train_failure_classifier(df, seed=42, output_dir=models_dir)

    metrics = meta["metrics"]
    macro_f1 = metrics["macro_f1"]
    dummy_mf = metrics["dummy_baselines"]["most_frequent_f1"]
    dummy_strat = metrics["dummy_baselines"]["stratified_f1"]

    # Must beat both dummy baselines by at least +0.40
    assert macro_f1 > dummy_mf + 0.40, f"Model F1 ({macro_f1}) failed to beat most_frequent ({dummy_mf}) by 0.40"
    assert macro_f1 > dummy_strat + 0.40, f"Model F1 ({macro_f1}) failed to beat stratified ({dummy_strat}) by 0.40"


def test_failure_ablation_runs_and_shows_signal(tmp_path):
    """Ablation without text features must show a meaningful drop (delta F1 >= 0.30)."""
    gen = SyntheticDataGenerator(seed=42, enable_truncation=True, enable_ambiguity=True)
    output_dir = str(tmp_path / "data")
    gen.generate_dataset(num_jobs=350, output_dir=output_dir, write_logs=True)

    df = pd.read_csv(os.path.join(output_dir, "manifest.csv"))
    models_dir = str(tmp_path / "models")
    bundle, meta = train_failure_classifier(df, seed=42, output_dir=models_dir)

    ablation = meta["metrics"]["ablation_without_tfidf"]
    delta_f1 = ablation["delta_f1"]
    assert delta_f1 >= 0.30, f"Ablation delta F1 {delta_f1} is too small (< 0.30), text features must carry signal!"
