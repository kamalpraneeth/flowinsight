"""Failure classification, class imbalance handling, and root-cause ranking.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

from __future__ import annotations

import heapq
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from flowinsight.features import build_feature_pipeline
from flowinsight.parsers import ParsedLogRecord, parse_log_file


class FailureClassificationBundle:
    """Encapsulates preprocessor, failure classifier, classes, and root-cause ranker."""

    def __init__(
        self,
        feature_pipeline: Any,
        classifier: Any,
        classes: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.feature_pipeline = feature_pipeline
        self.classifier = classifier
        self.classes = list(classes)
        self.metadata = metadata or {}

    def predict_top_k_causes(
        self,
        data: Union[str, Dict[str, Any], ParsedLogRecord],
        k: int = 3,
    ) -> Dict[str, Any]:
        """Rank top-k failure causes using min-heap priority queue.

        Complexity Note:
        ----------------
        For C candidate classes and k requested top causes, maintaining a min-heap
        of size k across C classes has time complexity O(C * log(k)), where C is the number
        of target failure classes (here C=7, k=3). This ensures optimal top-k extraction
        compared to full array sorting O(C * log(C)).
        """
        raw_lines: List[str] = []
        if isinstance(data, str) and os.path.exists(data):
            record = parse_log_file(data)
            df = [record]
            with open(data, "r", encoding="utf-8", errors="replace") as f:
                raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        elif isinstance(data, ParsedLogRecord):
            df = [data]
            raw_lines = [f"{m.severity} [{m.message_id}] {m.text}" for m in data.messages]
        elif isinstance(data, dict):
            df = [data]
            raw_lines = data.get("error_text", "").split("\n")
        else:
            raise ValueError(f"Unsupported data type for failure classification: {type(data)}")

        X_trans = self.feature_pipeline.transform(df)
        probas = self.classifier.predict_proba(X_trans)[0]

        # Use min-heap to extract top-k: O(C * log(k))
        # Heap elements are tuples: (probability, class_name)
        heap: List[Tuple[float, str]] = []
        for class_name, prob in zip(self.classes, probas):
            if len(heap) < k:
                heapq.heappush(heap, (prob, class_name))
            else:
                if prob > heap[0][0]:
                    heapq.heapreplace(heap, (prob, class_name))

        # Sort the k elements descending: O(k * log(k))
        ranked_causes = sorted(heap, key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for prob, cause in ranked_causes:
            supporting_lines = self._extract_supporting_lines(cause, raw_lines)
            results.append(
                {
                    "cause": cause,
                    "probability": round(float(prob), 4),
                    "supporting_evidence_lines": supporting_lines[:4],
                }
            )

        return {
            "top_causes": results,
            "predicted_cause": results[0]["cause"] if results else "none",
            "confidence": results[0]["probability"] if results else 0.0,
            "data_source": "synthetic",
        }

    def _extract_supporting_lines(self, cause: str, lines: List[str]) -> List[str]:
        """Extract lines in raw log that provide supporting evidence for candidate cause."""
        keywords = {
            "out_of_memory": ["mem", "oom", "allocation", "cgroup", "resident", "4194304", "mem-101", "mem-902"],
            "timeout": ["timeout", "execution limit", "watchdog", "wallclock", "time-501", "time-102"],
            "missing_library": ["lib-204", "lib-901", "library", "lef", "cell", "unresolved", "lib-102"],
            "constraint_violation": [
                "slack",
                "tim-302",
                "drc-401",
                "violation",
                "transition",
                "sta-500",
                "tim-104",
            ],
            "license_error": ["flexlm", "lic-101", "lic-108", "license", "checkout", "lic-105"],
            "syntax_error": ["syntax", "hdl-105", "parse-010", "unexpected token", "endmodule", "hdl-102"],
            "congestion": ["congestion", "rte-702", "cong-801", "gcell", "overflow", "pnr-880", "cong-102"],
        }
        kw_list = keywords.get(cause.lower(), [cause.lower()])
        matches: List[str] = []
        for line in lines:
            line_lower = line.lower()
            if any(k in line_lower for k in kw_list):
                matches.append(line)
        return matches or ["No direct matching diagnostic line found in log stream."]


def get_git_commit_hash() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown-commit"


def enrich_failed_df_with_logs(failed_df: pd.DataFrame, log_dir: Optional[str] = None) -> pd.DataFrame:
    """Ensure failed_df contains parsed error_text and message_ids_str from log files."""
    df_out = failed_df.copy()
    has_text = "error_text" in df_out.columns and not (
        df_out["error_text"].fillna("__none__").eq("__none__").all()
    )
    if has_text:
        return df_out

    candidate_dirs = [
        log_dir,
        "data/logs",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/logs"),
    ]
    valid_dir = None
    for d in candidate_dirs:
        if d and os.path.isdir(d):
            valid_dir = d
            break

    if not valid_dir:
        return df_out

    error_texts: List[str] = []
    msg_id_strs: List[str] = []
    for _, row in df_out.iterrows():
        job_id = row.get("job_id", "")
        log_path = os.path.join(valid_dir, f"{job_id}.log")
        if os.path.exists(log_path):
            parsed = parse_log_file(log_path)
            error_texts.append(parsed.raw_error_text or "__none__")
            msg_id_strs.append(" ".join(parsed.raw_message_ids) or "__none__")
        else:
            error_texts.append("__none__")
            msg_id_strs.append("__none__")

    df_out["error_text"] = error_texts
    df_out["message_ids_str"] = msg_id_strs
    return df_out


def train_failure_classifier(
    df: pd.DataFrame,
    model_type: str = "logistic_regression",
    seed: int = 42,
    output_dir: str = "models_saved",
    log_dir: Optional[str] = None,
) -> Tuple[FailureClassificationBundle, Dict[str, Any]]:
    """Train failure classifier with stratified split, class-imbalance weights, cross-validation, and ablation."""
    # Filter to failed jobs with labeled failure_cause
    failed_df = df[df["status"] == "failed"].copy()
    if failed_df.empty:
        raise ValueError("No failed jobs available in dataset for training failure classifier.")

    # Enrich from logs if error_text is missing or empty
    failed_df = enrich_failed_df_with_logs(failed_df, log_dir=log_dir)

    classes = sorted(failed_df["failure_cause"].unique())
    y = failed_df["failure_cause"].to_numpy()

    # Stratified train-test split (with safe check for class counts)
    _, class_counts = np.unique(y, return_counts=True)
    can_stratify = int(np.min(class_counts)) >= 2

    train_df, test_df, y_train, y_test = train_test_split(
        failed_df,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y if can_stratify else None,
    )

    # 1. Main Pipeline with TF-IDF and Message IDs
    feature_pipeline = build_feature_pipeline(include_text=True, include_msg_ids=True)
    X_train = feature_pipeline.fit_transform(train_df)
    X_test = feature_pipeline.transform(test_df)

    # Baselines: DummyClassifiers
    dummy_mf = DummyClassifier(strategy="most_frequent")
    dummy_mf.fit(X_train, y_train)
    pred_mf = dummy_mf.predict(X_test)
    f1_mf = float(f1_score(y_test, pred_mf, average="macro", zero_division=0))

    dummy_strat = DummyClassifier(strategy="stratified", random_state=seed)
    dummy_strat.fit(X_train, y_train)
    pred_strat = dummy_strat.predict(X_test)
    f1_strat = float(f1_score(y_test, pred_strat, average="macro", zero_division=0))

    # Cross-validation on train set
    _, train_class_counts = np.unique(y_train, return_counts=True)
    if int(np.min(train_class_counts)) >= 3:
        cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    else:
        cv_splitter = KFold(n_splits=3, shuffle=True, random_state=seed)

    cv_f1_scores: List[float] = []

    for fold_train_idx, fold_val_idx in cv_splitter.split(train_df, y_train):
        X_fold_tr = X_train[fold_train_idx]
        y_fold_tr = y_train[fold_train_idx]
        X_fold_val = X_train[fold_val_idx]
        y_fold_val = y_train[fold_val_idx]

        fold_clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed)
        fold_clf.fit(X_fold_tr, y_fold_tr)
        y_fold_pred = fold_clf.predict(X_fold_val)
        cv_f1_scores.append(float(f1_score(y_fold_val, y_fold_pred, average="macro")))

    # Train Main Model
    if model_type == "logistic_regression":
        classifier = LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed)
    else:
        classifier = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", max_depth=10, random_state=seed
        )

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Compute genuine per-class metrics
    per_class_p = precision_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    per_class_r = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for c_idx, c_name in enumerate(classes):
        per_class_metrics[c_name] = {
            "precision": round(float(per_class_p[c_idx]), 4),
            "recall": round(float(per_class_r[c_idx]), 4),
            "f1_score": round(float(per_class_f1[c_idx]), 4),
        }

    # Top-3 Ranking Recall
    probas = classifier.predict_proba(X_test)
    top3_hits = 0
    per_class_top3_hits = {c: 0 for c in classes}
    per_class_test_counts = {c: 0 for c in classes}
    for i in range(len(y_test)):
        true_c = y_test[i]
        per_class_test_counts[true_c] += 1
        top3_idx = np.argsort(probas[i])[::-1][:3]
        top3_c = [classes[idx] for idx in top3_idx]
        if true_c in top3_c:
            top3_hits += 1
            per_class_top3_hits[true_c] += 1

    top3_recall_per_class = {
        c: round(per_class_top3_hits[c] / max(1, per_class_test_counts[c]), 4)
        for c in classes
    }
    top3_accuracy_overall = round(top3_hits / max(1, len(y_test)), 4)

    # Comparison baseline: RandomForest vs Logistic Regression
    rf_comparison = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", max_depth=10, random_state=seed
    )
    rf_comparison.fit(X_train, y_train)
    rf_pred = rf_comparison.predict(X_test)
    rf_macro_f1 = float(f1_score(y_test, rf_pred, average="macro", zero_division=0))
    rf_p = precision_score(y_test, rf_pred, labels=classes, average=None, zero_division=0)
    rf_r = recall_score(y_test, rf_pred, labels=classes, average=None, zero_division=0)

    # 2. Ablation: Model trained WITHOUT TF-IDF and message ID features
    ablation_pipeline = build_feature_pipeline(include_text=False, include_msg_ids=False)
    X_train_abl = ablation_pipeline.fit_transform(train_df)
    X_test_abl = ablation_pipeline.transform(test_df)

    ablation_clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed)
    ablation_clf.fit(X_train_abl, y_train)
    y_pred_abl = ablation_clf.predict(X_test_abl)
    ablation_macro_f1 = float(f1_score(y_test, y_pred_abl, average="macro", zero_division=0))

    metrics = {
        "macro_f1": round(macro_f1, 4),
        "cv_f1_mean": round(float(np.mean(cv_f1_scores)), 4),
        "cv_f1_std": round(float(np.std(cv_f1_scores)), 4),
        "per_class": per_class_metrics,
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm,
        },
        "dummy_baselines": {
            "most_frequent_f1": round(f1_mf, 4),
            "stratified_f1": round(f1_strat, 4),
        },
        "baseline_random_forest": {
            "macro_f1": round(rf_macro_f1, 4),
            "per_class_precision": {c: round(float(rf_p[idx]), 4) for idx, c in enumerate(classes)},
            "per_class_recall": {c: round(float(rf_r[idx]), 4) for idx, c in enumerate(classes)},
        },
        "top3_ranking": {
            "overall_top3_accuracy": top3_accuracy_overall,
            "per_class_top3_recall": top3_recall_per_class,
        },
        "ablation_without_tfidf": {
            "macro_f1": round(ablation_macro_f1, 4),
            "delta_f1": round(macro_f1 - ablation_macro_f1, 4),
            "description": "Excludes TF-IDF and message ID features; relies solely on numeric tabular attributes.",
        },
    }

    metadata = {
        "model_name": f"failure_classifier_{model_type}",
        "task": "root_cause_failure_classification",
        "model_type": model_type,
        "date_trained": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit_hash(),
        "random_seed": seed,
        "dataset_size": len(failed_df),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "split_method": "stratified_by_failure_cause",
        "classes": classes,
        "metrics": metrics,
        "limitations": [
            "Evaluated on synthetic EDA log errors.",
            "Heuristic evidence line extractor relies on regex keyword alignment.",
            "Ambiguous failure signatures with cross-stage noise may have lower recall.",
        ],
        "data_source": "synthetic",
    }

    bundle = FailureClassificationBundle(
        feature_pipeline=feature_pipeline,
        classifier=classifier,
        classes=classes,
        metadata=metadata,
    )

    os.makedirs(output_dir, exist_ok=True)
    bundle_path = os.path.join(output_dir, "failure_model.joblib")
    meta_path = os.path.join(output_dir, "failure_metadata.json")

    joblib.dump(bundle, bundle_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return bundle, metadata
