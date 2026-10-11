"""Unit tests for Phase 2 Feature Pipeline.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

import pandas as pd

from flowinsight.features import build_feature_pipeline
from flowinsight.parsers import ParsedLogRecord


def test_feature_pipeline_fit_and_transform_consistency():
    """Verify train and predict time feature transformations are identical."""
    pipeline = build_feature_pipeline()

    train_data = pd.DataFrame(
        [
            {
                "design_size": 100_000,
                "threads": 4,
                "clock_period_ns": 1.2,
                "max_routing_layers": 0,
                "warnings_count": 5,
                "errors_count": 0,
                "num_steps": 5,
                "total_step_wall_s": 150.0,
                "max_step_mem_mb": 1200.0,
                "stage": "synthesis",
                "effort": "medium",
                "target": "timing",
                "tool_version": "v2023.03-SP1",
                "message_ids_str": "TIM-104 DRC-201",
                "error_text": "Non-critical timing slack alert",
            },
            {
                "design_size": 500_000,
                "threads": 8,
                "clock_period_ns": 2.0,
                "max_routing_layers": 8,
                "warnings_count": 25,
                "errors_count": 2,
                "num_steps": 6,
                "total_step_wall_s": 800.0,
                "max_step_mem_mb": 3400.0,
                "stage": "place_and_route",
                "effort": "high",
                "target": "area",
                "tool_version": "v23.10-s095_1",
                "message_ids_str": "MEM-902 SYS-004",
                "error_text": "Fatal memory allocation failure",
            },
        ]
    )

    # Fit on train data
    X_train = pipeline.fit_transform(train_data)
    assert X_train.shape[0] == 2
    n_features = X_train.shape[1]
    assert n_features > 0

    # Predict time with a single record (object or dict)
    record = ParsedLogRecord(
        job_id="JOB-SYN-TEST",
        stage="synthesis",
        design_name="test_mod",
        design_size=120_000,
        tool_version="v2023.03-SP1",
        threads=4,
        raw_error_text="Fatal memory allocation alert",
        raw_message_ids=["MEM-902"],
    )

    X_pred = pipeline.transform([record])
    assert X_pred.shape[0] == 1
    assert X_pred.shape[1] == n_features, (
        "Transformed feature count must match between train and predict"
    )


def test_feature_pipeline_no_data_leakage():
    """Verify pipeline does not store or leak row target labels."""
    pipeline = build_feature_pipeline()
    data = pd.DataFrame(
        [
            {"design_size": 200_000, "stage": "verification", "tool_version": "v2024.03-patch2"},
            {"design_size": 800_000, "stage": "synthesis", "tool_version": "v2024.06"},
        ]
    )
    X = pipeline.fit_transform(data)
    assert not hasattr(pipeline, "target_labels_")
    assert X.shape[0] == 2
