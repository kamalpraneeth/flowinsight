"""Command-line interface for FlowInsight EDA tool analytics and prediction toolkit.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import pandas as pd

from flowinsight.generator import generate_synthetic_data
from flowinsight.models.anomaly import train_anomaly_detectors
from flowinsight.models.failure import train_failure_classifier
from flowinsight.models.runtime import train_runtime_regressor
from flowinsight.parsers import parse_log_file
from flowinsight.reports import generate_markdown_report
from flowinsight.store import load_manifest_and_logs


def handle_generate(args: argparse.Namespace) -> int:
    """Generate synthetic EDA execution logs and manifest."""
    print(f"Generating {args.jobs} synthetic EDA tool runs (seed={args.seed})...")
    generate_synthetic_data(num_jobs=args.jobs, seed=args.seed, output_dir=args.output_dir)
    print(f"Synthetic dataset created in '{args.output_dir}'.")
    return 0


def handle_parse(args: argparse.Namespace) -> int:
    """Parse log files or manifest and ingest into SQLite database."""
    if args.log:
        record = parse_log_file(args.log)
        print(
            json.dumps(
                {
                    "job_id": record.job_id,
                    "stage": record.stage,
                    "status": record.status,
                    "runtime_s": record.runtime_s,
                    "peak_memory_mb": record.peak_memory_mb,
                    "errors_count": record.errors_count,
                    "warnings_count": record.warnings_count,
                    "parse_warnings": record.parse_warnings,
                    "data_source": "synthetic",
                },
                indent=2,
            )
        )
        return 0

    manifest_path = args.manifest or os.path.join(args.input_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at '{manifest_path}'.", file=sys.stderr)
        return 1

    print(f"Ingesting manifest from '{manifest_path}' into '{args.db}'...")
    loaded = load_manifest_and_logs(
        db_path=args.db,
        manifest_path=manifest_path,
        parse_logs=not args.skip_logs,
        max_log_parse=args.max_logs,
    )
    print(f"Successfully loaded {loaded} job records into database.")
    return 0


def handle_train(args: argparse.Namespace) -> int:
    """Train ML models and save serializations and metadata cards."""
    manifest_path = args.manifest
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at '{manifest_path}'.", file=sys.stderr)
        return 1

    print(f"Loading training dataset from '{manifest_path}'...")
    df = pd.read_csv(manifest_path)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Training Runtime Regressor...")
    train_runtime_regressor(df, target_col="runtime_s", seed=args.seed, output_dir=args.output_dir)

    print("Training Failure Classifier & Root Cause Ranker...")
    train_failure_classifier(df, seed=args.seed, output_dir=args.output_dir)

    print("Training Anomaly Detectors (Isolation Forest vs Robust MAD)...")
    train_anomaly_detectors(df, seed=args.seed, output_dir=args.output_dir)

    print(f"All models trained and metadata cards written to '{args.output_dir}'.")
    return 0


def handle_predict_runtime(args: argparse.Namespace) -> int:
    """Predict execution runtime and 80% interval."""
    import joblib

    model_path = os.path.join(args.model_dir, "runtime_s_model.joblib")
    if not os.path.exists(model_path):
        print(
            f"Error: Model file '{model_path}' not found. Run 'flowinsight train' first.",
            file=sys.stderr,
        )
        return 1

    bundle = joblib.load(model_path)
    if args.log:
        res = bundle.predict(args.log)
    else:
        sample_dict = {
            "design_size": args.size,
            "stage": args.stage,
            "threads": args.threads,
            "clock_period_ns": args.clock_period,
            "effort": args.effort,
            "target": args.target,
        }
        res = bundle.predict(sample_dict)

    print(json.dumps(res, indent=2))
    return 0


def handle_classify_failure(args: argparse.Namespace) -> int:
    """Classify failure root causes and rank top 3 causes with evidence."""
    import joblib

    model_path = os.path.join(args.model_dir, "failure_model.joblib")
    if not os.path.exists(model_path):
        print(
            f"Error: Model file '{model_path}' not found. Run 'flowinsight train' first.",
            file=sys.stderr,
        )
        return 1

    bundle = joblib.load(model_path)
    res = bundle.predict_top_k_causes(args.log, k=args.top_k)
    print(json.dumps(res, indent=2))
    return 0


def handle_detect_anomalies(args: argparse.Namespace) -> int:
    """Detect execution anomalies using Isolation Forest and Robust MAD."""
    import joblib

    model_path = os.path.join(args.model_dir, "anomaly_model.joblib")
    if not os.path.exists(model_path):
        print(
            f"Error: Model file '{model_path}' not found. Run 'flowinsight train' first.",
            file=sys.stderr,
        )
        return 1

    bundle = joblib.load(model_path)
    res = bundle.detect(args.log)
    print(json.dumps(res, indent=2))
    return 0


def handle_report(args: argparse.Namespace) -> int:
    """Generate analytical Markdown executive report and Matplotlib charts."""
    manifest_path = args.manifest
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file '{manifest_path}' not found.", file=sys.stderr)
        return 1

    df = pd.read_csv(manifest_path)
    out_file = generate_markdown_report(
        df=df,
        output_file=args.output,
        chart_dir=args.chart_dir,
    )
    print(f"Report generated successfully: {out_file}")
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    """Run FastAPI server via Uvicorn."""
    import uvicorn

    print(f"Starting FlowInsight API server on {args.host}:{args.port}...")
    uvicorn.run("flowinsight.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowinsight",
        description="FlowInsight: EDA Tool Log Analytics and Prediction Toolkit\n"
        "NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. generate
    p_gen = subparsers.add_parser("generate", help="Generate synthetic EDA tool logs and manifest")
    p_gen.add_argument(
        "--jobs", type=int, default=5000, help="Number of jobs to generate (default: 5000)"
    )
    p_gen.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    p_gen.add_argument("--output-dir", default="data", help="Output directory (default: data)")

    # 2. parse
    p_prs = subparsers.add_parser("parse", help="Parse log files and store in SQLite database")
    p_prs.add_argument("--log", help="Parse a single log file and print structured JSON")
    p_prs.add_argument("--manifest", help="Path to manifest CSV")
    p_prs.add_argument("--input-dir", default="data", help="Directory containing manifest and logs")
    p_prs.add_argument("--db", default="data/flowinsight.db", help="Target SQLite database path")
    p_prs.add_argument(
        "--skip-logs", action="store_true", help="Load manifest only without parsing logs"
    )
    p_prs.add_argument("--max-logs", type=int, default=None, help="Limit number of logs to parse")

    # 3. train
    p_trn = subparsers.add_parser("train", help="Train runtime, failure, and anomaly ML models")
    p_trn.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    p_trn.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_trn.add_argument("--output-dir", default="models_saved", help="Directory to save models")

    # 4. predict-runtime
    p_prt = subparsers.add_parser(
        "predict-runtime", help="Predict job runtime and prediction intervals"
    )
    p_prt.add_argument("--log", help="Path to log file to predict from")
    p_prt.add_argument("--size", type=int, default=250000, help="Gate count")
    p_prt.add_argument(
        "--stage", default="synthesis", choices=["synthesis", "place_and_route", "verification"]
    )
    p_prt.add_argument("--threads", type=int, default=8, help="Worker threads")
    p_prt.add_argument("--clock-period", type=float, default=1.5, help="Clock period (ns)")
    p_prt.add_argument("--effort", default="medium", choices=["low", "medium", "high", "extreme"])
    p_prt.add_argument("--target", default="timing", choices=["timing", "area", "power"])
    p_prt.add_argument("--model-dir", default="models_saved", help="Directory with saved models")

    # 5. classify-failure
    p_cls = subparsers.add_parser(
        "classify-failure", help="Classify root cause and extract evidence lines"
    )
    p_cls.add_argument("--log", required=True, help="Path to failed log file")
    p_cls.add_argument("--top-k", type=int, default=3, help="Number of top causes to rank")
    p_cls.add_argument("--model-dir", default="models_saved", help="Directory with saved models")

    # 6. detect-anomalies
    p_ano = subparsers.add_parser("detect-anomalies", help="Detect anomalies in a log run")
    p_ano.add_argument("--log", required=True, help="Path to log file")
    p_ano.add_argument("--model-dir", default="models_saved", help="Directory with saved models")

    # 7. report
    p_rep = subparsers.add_parser(
        "report", help="Generate executive markdown analytics report and charts"
    )
    p_rep.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    p_rep.add_argument("--output", default="reports/nightly_report.md", help="Markdown output path")
    p_rep.add_argument("--chart-dir", default="reports/charts", help="Chart image output directory")

    # 8. serve
    p_srv = subparsers.add_parser("serve", help="Launch FastAPI server")
    p_srv.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    p_srv.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    p_srv.add_argument("--reload", action="store_true", help="Auto-reload on code changes")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    handlers = {
        "generate": handle_generate,
        "parse": handle_parse,
        "train": handle_train,
        "predict-runtime": handle_predict_runtime,
        "classify-failure": handle_classify_failure,
        "detect-anomalies": handle_detect_anomalies,
        "report": handle_report,
        "serve": handle_serve,
    }

    handler = handlers.get(parsed_args.command)
    if not handler:
        parser.print_help()
        return 1

    return handler(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
