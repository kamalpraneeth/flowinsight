"""Feature extraction and scikit-learn preprocessing pipelines for FlowInsight.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.

Feature Dictionary:
==================
1. Numeric Features:
   - 'design_size': Gate count representing design physical scale.
   - 'threads': Worker concurrency allocated to execution.
   - 'clock_period_ns': Clock target period in nanoseconds.
   - 'max_routing_layers': Routing layer allocation limit (0 for non-P&R).
   - 'warnings_count': Total warnings emitted during execution.
   - 'errors_count': Total error/fatal lines identified.
   - 'num_steps': Number of execution steps recorded.
   - 'total_step_wall_s': Sum of wallclock durations across recorded steps.
   - 'max_step_mem_mb': Maximum memory recorded across step checkpoints.

2. Categorical Features (One-Hot Encoded):
   - 'stage': Synthesis, place_and_route, or verification.
   - 'effort': Optimization effort (low, medium, high, extreme).
   - 'target': Target priority (area, timing, power).
   - 'tool_version': EDA tool software release tag.

3. Message ID Features:
   - Bag-of-words message ID count frequencies (top 50 EDA message codes e.g. MEM-902, TIM-302).

4. Text TF-IDF Features:
   - Term frequency-inverse document frequency of error and diagnostic lines (top 100 n-grams).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flowinsight.parsers import ParsedLogRecord


class LogRecordTabularizer(BaseEstimator, TransformerMixin):
    """Converts a list of ParsedLogRecord objects or dicts into a structured pandas DataFrame."""

    def fit(self, X: Any, y: Any = None) -> "LogRecordTabularizer":
        return self

    def transform(
        self, X: Union[List[ParsedLogRecord], List[Dict[str, Any]], pd.DataFrame]
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return self._normalize_dataframe(X)

        rows: List[Dict[str, Any]] = []
        for item in X:
            if isinstance(item, ParsedLogRecord):
                config = {}
                num_steps = len(item.steps)
                total_step_wall = sum(s.wall_s for s in item.steps) if item.steps else 0.0
                max_step_mem = max((s.mem_mb for s in item.steps), default=0.0)

                rows.append(
                    {
                        "design_size": item.design_size or 100_000,
                        "threads": item.threads or 4,
                        "clock_period_ns": 1.5,
                        "max_routing_layers": 6 if item.stage == "place_and_route" else 0,
                        "warnings_count": item.warnings_count,
                        "errors_count": item.errors_count,
                        "num_steps": num_steps,
                        "total_step_wall_s": total_step_wall,
                        "max_step_mem_mb": max_step_mem,
                        "stage": item.stage,
                        "effort": "medium",
                        "target": "timing",
                        "tool_version": item.tool_version or "v2023.03-SP1",
                        "message_ids_str": " ".join(item.raw_message_ids) or "__none__",
                        "error_text": item.raw_error_text or "__none__",
                    }
                )
            elif isinstance(item, dict):
                # May be a row from manifest or DB
                config = {}
                if "config_json" in item and item["config_json"]:
                    try:
                        config = json.loads(item["config_json"])
                    except Exception:
                        pass

                rows.append(
                    {
                        "design_size": item.get("design_size", 100_000),
                        "threads": config.get("threads", item.get("threads", 4)),
                        "clock_period_ns": config.get("clock_period_ns", 1.5),
                        "max_routing_layers": config.get("max_routing_layers", 0),
                        "warnings_count": item.get("warnings_count", 0),
                        "errors_count": item.get("errors_count", 0),
                        "num_steps": item.get("num_steps", 5),
                        "total_step_wall_s": item.get(
                            "total_step_wall_s", item.get("runtime_s", 0.0)
                        ),
                        "max_step_mem_mb": item.get(
                            "max_step_mem_mb", item.get("peak_memory_mb", 0.0)
                        ),
                        "stage": item.get("stage", "synthesis"),
                        "effort": config.get("effort", "medium"),
                        "target": config.get("target", "timing"),
                        "tool_version": item.get("tool_version", "v2023.03-SP1"),
                        "message_ids_str": item.get("message_ids_str") or "__none__",
                        "error_text": item.get("error_text") or "__none__",
                    }
                )
        return pd.DataFrame(rows)

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist in input DataFrame."""
        out = df.copy()
        if "config_json" in out.columns:
            # Extract config attributes if not already top-level
            if "threads" not in out.columns:

                def extract_cfg(raw: Any) -> Dict[str, Any]:
                    if isinstance(raw, str):
                        try:
                            return json.loads(raw)
                        except Exception:
                            return {}
                    return raw if isinstance(raw, dict) else {}

                configs = out["config_json"].apply(extract_cfg)
                out["threads"] = configs.apply(lambda c: c.get("threads", 4))
                out["clock_period_ns"] = configs.apply(lambda c: c.get("clock_period_ns", 1.5))
                out["max_routing_layers"] = configs.apply(lambda c: c.get("max_routing_layers", 0))
                out["effort"] = configs.apply(lambda c: c.get("effort", "medium"))
                out["target"] = configs.apply(lambda c: c.get("target", "timing"))

        required_defaults = {
            "design_size": 100_000,
            "threads": 4,
            "clock_period_ns": 1.5,
            "max_routing_layers": 0,
            "warnings_count": 0,
            "errors_count": 0,
            "num_steps": 5,
            "total_step_wall_s": 100.0,
            "max_step_mem_mb": 800.0,
            "stage": "synthesis",
            "effort": "medium",
            "target": "timing",
            "tool_version": "v2023.03-SP1",
            "message_ids_str": "__none__",
            "error_text": "__none__",
        }
        for col, default_val in required_defaults.items():
            if col not in out.columns:
                out[col] = default_val

        out["message_ids_str"] = (
            out["message_ids_str"]
            .fillna("__none__")
            .apply(lambda s: s.strip() if isinstance(s, str) and s.strip() else "__none__")
        )
        out["error_text"] = (
            out["error_text"]
            .fillna("__none__")
            .apply(lambda s: s.strip() if isinstance(s, str) and s.strip() else "__none__")
        )

        return out


def build_feature_pipeline(
    include_text: bool = True,
    include_msg_ids: bool = True,
) -> Pipeline:
    """Build a ColumnTransformer scikit-learn Pipeline with strict consistency between train and predict."""
    numeric_features = [
        "design_size",
        "threads",
        "clock_period_ns",
        "max_routing_layers",
        "warnings_count",
        "errors_count",
        "num_steps",
        "total_step_wall_s",
        "max_step_mem_mb",
    ]

    categorical_features = ["stage", "effort", "target", "tool_version"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers: List[Tuple[str, Any, Any]] = [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]

    if include_msg_ids:
        msg_vec = CountVectorizer(token_pattern=r"[A-Za-z0-9_\-]+", max_features=50)
        transformers.append(("msg_ids", msg_vec, "message_ids_str"))

    if include_text:
        text_vec = TfidfVectorizer(
            max_features=80,
            stop_words="english",
            ngram_range=(1, 2),
        )
        transformers.append(("text_tfidf", text_vec, "error_text"))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,  # Always output dense ndarray
    )

    full_pipeline = Pipeline(
        steps=[
            ("tabularize", LogRecordTabularizer()),
            ("preprocessor", preprocessor),
        ]
    )
    return full_pipeline
