"""SQLite storage layer, indexing, and SQL analytical query execution.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

from __future__ import annotations

import csv
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

from flowinsight.parsers import parse_log_file

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    design_name TEXT NOT NULL,
    design_size INTEGER NOT NULL,
    tool_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    start_time TEXT NOT NULL,
    runtime_s REAL NOT NULL,
    peak_memory_mb REAL NOT NULL,
    warnings_count INTEGER NOT NULL,
    errors_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    failure_cause TEXT NOT NULL,
    is_anomaly INTEGER NOT NULL DEFAULT 0,
    anomaly_type TEXT NOT NULL DEFAULT 'none',
    is_truncated INTEGER NOT NULL DEFAULT 0,
    log_path TEXT,
    data_source TEXT NOT NULL DEFAULT 'synthetic'
);

CREATE INDEX IF NOT EXISTS idx_jobs_stage ON jobs(stage);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_start_time ON jobs(start_time);
CREATE INDEX IF NOT EXISTS idx_jobs_design ON jobs(design_name);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    severity TEXT NOT NULL,
    message_id TEXT NOT NULL,
    message_text TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_job_id ON messages(job_id);
CREATE INDEX IF NOT EXISTS idx_messages_severity ON messages(severity);
CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id);

CREATE TABLE IF NOT EXISTS step_timings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    cpu_s REAL NOT NULL,
    wall_s REAL NOT NULL,
    mem_mb REAL NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_step_timings_job_id ON step_timings(job_id);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    method TEXT NOT NULL,
    is_anomaly INTEGER NOT NULL,
    score REAL NOT NULL,
    details_json TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_anomalies_job_id ON anomalies(job_id);
"""


def init_db(db_path: str = "data/flowinsight.db") -> sqlite3.Connection:
    """Initialize SQLite database schema and indexes."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(SCHEMA_DDL)
    return conn


def load_manifest_and_logs(
    db_path: str,
    manifest_path: str,
    logs_dir: Optional[str] = None,
    parse_logs: bool = True,
    max_log_parse: Optional[int] = None,
) -> int:
    """Load jobs manifest and parse corresponding log files into SQLite database."""
    conn = init_db(db_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if logs_dir is None:
        logs_dir = os.path.join(os.path.dirname(manifest_path), "logs")

    jobs_loaded = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        with conn:
            for row in reader:
                job_id = row["job_id"]
                log_file = os.path.join(logs_dir, f"{job_id}.log")
                log_path_val = log_file if os.path.exists(log_file) else None

                conn.execute(
                    """
                    INSERT OR REPLACE INTO jobs (
                        job_id, stage, design_name, design_size, tool_version,
                        config_json, start_time, runtime_s, peak_memory_mb,
                        warnings_count, errors_count, status, failure_cause,
                        is_anomaly, anomaly_type, is_truncated, log_path, data_source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        row["stage"],
                        row["design_name"],
                        int(row["design_size"]),
                        row["tool_version"],
                        row["config_json"],
                        row["start_time"],
                        float(row["runtime_s"]),
                        float(row["peak_memory_mb"]),
                        int(row["warnings_count"]),
                        int(row["errors_count"]),
                        row["status"],
                        row["failure_cause"],
                        int(row.get("is_anomaly", 0)),
                        row.get("anomaly_type", "none"),
                        int(row.get("is_truncated", 0)),
                        log_path_val,
                        row.get("data_source", "synthetic"),
                    ),
                )
                jobs_loaded += 1

                # Parse detailed log messages and steps if logs available
                if (
                    parse_logs
                    and log_path_val
                    and (max_log_parse is None or jobs_loaded <= max_log_parse)
                ):
                    try:
                        record = parse_log_file(log_path_val, stage_hint=row["stage"])
                        # Clear any existing messages or timings for idempotency
                        conn.execute("DELETE FROM messages WHERE job_id = ?", (job_id,))
                        conn.execute("DELETE FROM step_timings WHERE job_id = ?", (job_id,))

                        msg_tuples = [
                            (
                                job_id,
                                msg.line_number,
                                msg.severity,
                                msg.message_id,
                                msg.text,
                            )
                            for msg in record.messages
                        ]
                        if msg_tuples:
                            conn.executemany(
                                "INSERT INTO messages (job_id, line_number, severity, message_id, message_text) VALUES (?, ?, ?, ?, ?)",
                                msg_tuples,
                            )

                        step_tuples = [
                            (
                                job_id,
                                step.step_name,
                                step.step_index,
                                step.cpu_s,
                                step.wall_s,
                                step.mem_mb,
                            )
                            for step in record.steps
                        ]
                        if step_tuples:
                            conn.executemany(
                                "INSERT INTO step_timings (job_id, step_name, step_index, cpu_s, wall_s, mem_mb) VALUES (?, ?, ?, ?, ?, ?)",
                                step_tuples,
                            )
                    except Exception:
                        pass  # Fail soft on individual log parse to preserve manifest

    conn.close()
    return jobs_loaded


def load_sql_queries(sql_file_path: Optional[str] = None) -> Dict[str, str]:
    """Parse sql/queries.sql into a dictionary of named queries."""
    if sql_file_path is None:
        sql_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sql",
            "queries.sql",
        )

    queries: Dict[str, str] = {}
    if not os.path.exists(sql_file_path):
        return queries

    with open(sql_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("-- name:")
    for block in blocks[1:]:
        lines = block.strip().split("\n")
        query_name = lines[0].strip()
        sql_text = "\n".join(lines[1:]).strip()
        # Remove trailing semicolons or commentary
        sql_text = re.sub(r";\s*$", "", sql_text)
        queries[query_name] = sql_text

    return queries


def run_query(
    db_path: str,
    query_name: str,
    params: Optional[tuple] = None,
    sql_file_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute a named query from queries.sql against the database."""
    queries = load_sql_queries(sql_file_path)
    if query_name not in queries:
        raise KeyError(f"Query '{query_name}' not found in SQL definitions.")

    sql = queries[query_name]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        rows = [dict(row) for row in cur.fetchall()]
        return rows
    finally:
        conn.close()


def run_failure_rate_by_stage_and_week(db_path: str) -> List[Dict[str, Any]]:
    return run_query(db_path, "failure_rate_by_stage_and_week")


def run_slowest_designs(db_path: str) -> List[Dict[str, Any]]:
    return run_query(db_path, "slowest_designs")


def run_most_frequent_errors(db_path: str) -> List[Dict[str, Any]]:
    return run_query(db_path, "most_frequent_errors")


def run_runtime_percentiles_by_stage(db_path: str) -> List[Dict[str, Any]]:
    return run_query(db_path, "runtime_percentiles_by_stage")


def run_top_warning_ids_before_failures(db_path: str) -> List[Dict[str, Any]]:
    return run_query(db_path, "top_warning_ids_before_failures")
