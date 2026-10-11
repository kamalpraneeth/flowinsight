"""FastAPI REST API server for FlowInsight EDA Tool Log Analytics.

NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from flowinsight.store import init_db

DATA_SOURCE = "synthetic"
BANNER = "Synthetic data generated for demonstration. Not real EDA tool logs."
DB_PATH = os.environ.get("FLOWINSIGHT_DB_PATH", "data/flowinsight.db")
MODELS_DIR = os.environ.get("FLOWINSIGHT_MODELS_DIR", "models_saved")
API_KEY_ENV = os.environ.get("FLOWINSIGHT_API_KEY", "")
IS_PRODUCTION = os.environ.get("ENVIRONMENT", "development").lower() == "production"

app = FastAPI(
    title="FlowInsight EDA Analytics API",
    description="Analytics and prediction toolkit for EDA tool execution telemetry.\n\n"
    "NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.",
    version="0.1.0",
)

# CORS configuration
raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
)
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

# Prevent wildcard in production
if IS_PRODUCTION and "*" in allowed_origins:
    allowed_origins = [o for o in allowed_origins if o != "*"]
    if not allowed_origins:
        allowed_origins = ["https://flowinsight.example.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request size limit middleware (10MB max)
@app.middleware("http")
async def limit_request_size(request: Request, call_next: Any) -> Response:
    max_size = 10 * 1024 * 1024  # 10MB
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        return Response(
            content=json.dumps(
                {"detail": "Request payload exceeds 10MB limit.", "data_source": DATA_SOURCE}
            ),
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            media_type="application/json",
        )
    response = await call_next(request)
    return response


# Optional API Key authentication dependency
def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if API_KEY_ENV:
        if not x_api_key or x_api_key != API_KEY_ENV:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key authentication header.",
            )


# Pydantic Request / Response schemas
class PredictRuntimeRequest(BaseModel):
    design_size: int = Field(..., ge=1000, le=50_000_000, description="Standard cell/gate count")
    stage: str = Field(..., pattern="^(synthesis|place_and_route|verification)$")
    threads: int = Field(default=4, ge=1, le=128)
    clock_period_ns: float = Field(default=1.5, gt=0.1, le=50.0)
    max_routing_layers: int = Field(default=0, ge=0, le=20)
    effort: str = Field(default="medium", pattern="^(low|medium|high|extreme)$")
    target: str = Field(default="timing", pattern="^(timing|area|power)$")


class ClassifyFailureRequest(BaseModel):
    log_text: Optional[str] = None
    job_id: Optional[str] = None
    stage: Optional[str] = "synthesis"


class AnomalyDetectRequest(BaseModel):
    log_text: Optional[str] = None
    design_size: Optional[int] = 100_000
    runtime_s: Optional[float] = 100.0
    peak_memory_mb: Optional[float] = 800.0
    warnings_count: Optional[int] = 10


def get_db() -> sqlite3.Connection:
    conn = init_db(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/health")
def get_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "0.1.0",
        "banner": BANNER,
        "data_source": DATA_SOURCE,
    }


@app.get("/metrics")
def get_metrics(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM jobs;")
        total_jobs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM jobs WHERE status = 'failed';")
        failed_jobs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM jobs WHERE is_anomaly = 1;")
        anomalies_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM messages;")
        messages_count = cur.fetchone()[0]

        return {
            "total_jobs": total_jobs,
            "failed_jobs": failed_jobs,
            "anomalies_count": anomalies_count,
            "total_messages_indexed": messages_count,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_source": DATA_SOURCE,
        }
    finally:
        conn.close()


@app.get("/jobs")
def list_jobs(
    stage: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    failure_cause: Optional[str] = None,
    design_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("start_time"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    auth: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    conn = get_db()
    try:
        allowed_sort = {
            "start_time",
            "runtime_s",
            "peak_memory_mb",
            "design_size",
            "warnings_count",
            "errors_count",
        }
        sort_col = sort_by if sort_by in allowed_sort else "start_time"

        conditions = ["1=1"]
        params: List[Any] = []

        if stage:
            conditions.append("stage = ?")
            params.append(stage)
        if status_filter:
            conditions.append("status = ?")
            params.append(status_filter)
        if failure_cause:
            conditions.append("failure_cause = ?")
            params.append(failure_cause)
        if design_name:
            conditions.append("design_name LIKE ?")
            params.append(f"%{design_name}%")
        if date_from:
            conditions.append("start_time >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("start_time <= ?")
            params.append(date_to)

        where_clause = " AND ".join(conditions)

        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM jobs WHERE {where_clause}", params)
        total_count = cur.fetchone()[0]

        offset = (page - 1) * page_size
        query = f"""
            SELECT * FROM jobs
            WHERE {where_clause}
            ORDER BY {sort_col} {sort_order.upper()}
            LIMIT ? OFFSET ?
        """
        cur.execute(query, params + [page_size, offset])
        rows = [dict(r) for r in cur.fetchall()]

        return {
            "items": rows,
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size if page_size else 1,
            "data_source": DATA_SOURCE,
        }
    finally:
        conn.close()


@app.get("/jobs/{job_id}")
def get_job_detail(job_id: str, auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        job_row = cur.fetchone()
        if not job_row:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

        job_dict = dict(job_row)

        cur.execute(
            "SELECT * FROM step_timings WHERE job_id = ? ORDER BY step_index ASC", (job_id,)
        )
        steps = [dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT * FROM messages WHERE job_id = ? ORDER BY line_number ASC LIMIT 100", (job_id,)
        )
        messages = [dict(r) for r in cur.fetchall()]

        # Log lines formatted with line numbers, capped at 500 lines
        log_lines: List[Dict[str, Any]] = []
        log_path = job_dict.get("log_path")
        if log_path and os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                for idx, line in enumerate(f, 1):
                    if idx > 500:
                        log_lines.append(
                            {
                                "line_number": idx,
                                "text": "... [REMAINDER OF LOG TRUNCATED FOR DISPLAY] ...",
                            }
                        )
                        break
                    log_lines.append({"line_number": idx, "text": line.rstrip()})

        # Predicted runtime from model if model file exists
        predicted_runtime: Optional[Dict[str, Any]] = None
        model_p = os.path.join(MODELS_DIR, "runtime_s_model.joblib")
        if os.path.exists(model_p):
            try:
                bundle = joblib.load(model_p)
                cfg = json.loads(job_dict.get("config_json", "{}"))
                pred_input = {
                    "design_size": job_dict["design_size"],
                    "stage": job_dict["stage"],
                    "threads": cfg.get("threads", 4),
                    "clock_period_ns": cfg.get("clock_period_ns", 1.5),
                    "max_routing_layers": cfg.get("max_routing_layers", 0),
                    "effort": cfg.get("effort", "medium"),
                    "target": cfg.get("target", "timing"),
                }
                predicted_runtime = bundle.predict(pred_input)
            except Exception:
                pass

        return {
            "job": job_dict,
            "steps": steps,
            "messages": messages,
            "log_lines": log_lines,
            "predicted_runtime": predicted_runtime,
            "data_source": DATA_SOURCE,
        }
    finally:
        conn.close()


@app.get("/stats/overview")
def get_stats_overview(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM jobs;")
        total_jobs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM jobs WHERE status = 'failed';")
        failed_jobs = cur.fetchone()[0]

        failure_rate = round((failed_jobs / total_jobs) * 100.0, 2) if total_jobs else 0.0

        cur.execute("SELECT COUNT(*) FROM jobs WHERE is_anomaly = 1;")
        anomalies_count = cur.fetchone()[0]

        # Median and P90 runtime of completed jobs
        cur.execute("SELECT runtime_s FROM jobs WHERE status = 'completed' ORDER BY runtime_s ASC;")
        runtimes = [r[0] for r in cur.fetchall()]
        n = len(runtimes)
        median_runtime = runtimes[n // 2] if n else 0.0
        p90_runtime = runtimes[int(0.90 * n)] if n else 0.0

        # Failure rate by stage
        cur.execute("""
            SELECT stage,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
            FROM jobs
            GROUP BY stage;
        """)
        stages = []
        for r in cur.fetchall():
            s_total = r["total"]
            s_failed = r["failed"]
            stages.append(
                {
                    "stage": r["stage"],
                    "total": s_total,
                    "failed": s_failed,
                    "failure_rate_pct": round((s_failed / s_total) * 100.0, 2) if s_total else 0.0,
                }
            )

        return {
            "total_jobs": total_jobs,
            "failed_jobs": failed_jobs,
            "failure_rate_pct": failure_rate,
            "anomalies_flagged": anomalies_count,
            "median_runtime_s": median_runtime,
            "p90_runtime_s": p90_runtime,
            "by_stage": stages,
            "banner": BANNER,
            "data_source": DATA_SOURCE,
        }
    finally:
        conn.close()


@app.get("/stats/failures")
def get_stats_failures(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT failure_cause, COUNT(*) as count
            FROM jobs
            WHERE status = 'failed' AND failure_cause != 'none'
            GROUP BY failure_cause
            ORDER BY count DESC;
        """)
        causes = [dict(r) for r in cur.fetchall()]

        # Top warning IDs before failures
        cur.execute("""
            SELECT m.message_id, COUNT(*) as count
            FROM messages m
            JOIN jobs j ON m.job_id = j.job_id
            WHERE m.severity IN ('WARN', 'WARNING') AND j.status = 'failed'
            GROUP BY m.message_id
            ORDER BY count DESC
            LIMIT 10;
        """)
        top_warnings = [dict(r) for r in cur.fetchall()]

        return {
            "failure_causes": causes,
            "top_warning_ids": top_warnings,
            "data_source": DATA_SOURCE,
        }
    finally:
        conn.close()


@app.get("/stats/trends")
def get_stats_trends(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT strftime('%Y-W%W', start_time) AS week,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
            FROM jobs
            GROUP BY week
            ORDER BY week ASC;
        """)
        weekly = []
        for r in cur.fetchall():
            tot = r["total"]
            fail = r["failed"]
            weekly.append(
                {
                    "week": r["week"],
                    "total": tot,
                    "failed": fail,
                    "failure_rate_pct": round((fail / tot) * 100.0, 2) if tot else 0.0,
                }
            )

        # Tool version runtime comparison
        cur.execute("""
            SELECT tool_version,
                   COUNT(*) as run_count,
                   ROUND(AVG(runtime_s), 2) as avg_runtime_s,
                   ROUND(MIN(runtime_s), 2) as min_runtime_s,
                   ROUND(MAX(runtime_s), 2) as max_runtime_s
            FROM jobs
            WHERE status = 'completed'
            GROUP BY tool_version
            ORDER BY tool_version ASC;
        """)
        versions = [dict(r) for r in cur.fetchall()]

        return {
            "weekly_failure_rates": weekly,
            "tool_versions": versions,
            "data_source": DATA_SOURCE,
        }
    finally:
        conn.close()


@app.get("/stats/anomalies")
def get_stats_anomalies(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT anomaly_type, COUNT(*) as count
            FROM jobs
            WHERE is_anomaly = 1
            GROUP BY anomaly_type;
        """)
        type_breakdown = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT job_id, stage, design_name, design_size, runtime_s, peak_memory_mb, warnings_count, anomaly_type, start_time
            FROM jobs
            WHERE is_anomaly = 1
            ORDER BY start_time DESC
            LIMIT 50;
        """)
        recent_anomalies = [dict(r) for r in cur.fetchall()]

        return {
            "type_breakdown": type_breakdown,
            "recent_anomalies": recent_anomalies,
            "data_source": DATA_SOURCE,
        }
    finally:
        conn.close()


@app.get("/models")
def get_models(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Retrieve saved model cards and metadata."""
    cards: List[Dict[str, Any]] = []
    if os.path.exists(MODELS_DIR):
        for fname in sorted(os.listdir(MODELS_DIR)):
            if fname.endswith("_metadata.json"):
                fpath = os.path.join(MODELS_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    cards.append(meta)
                except Exception:
                    pass

    return {
        "models": cards,
        "count": len(cards),
        "data_source": DATA_SOURCE,
    }


@app.post("/predict/runtime")
def predict_runtime(
    req: PredictRuntimeRequest, auth: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    model_p = os.path.join(MODELS_DIR, "runtime_s_model.joblib")
    if not os.path.exists(model_p):
        raise HTTPException(
            status_code=503,
            detail="Runtime model is not trained yet. Run 'flowinsight train' or POST /train first.",
        )

    bundle = joblib.load(model_p)
    res = bundle.predict(req.model_dump())
    res["data_source"] = DATA_SOURCE
    return res


@app.post("/classify/failure")
def classify_failure(
    req: ClassifyFailureRequest, auth: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    model_p = os.path.join(MODELS_DIR, "failure_model.joblib")
    if not os.path.exists(model_p):
        raise HTTPException(status_code=503, detail="Failure classifier is not trained yet.")

    bundle = joblib.load(model_p)
    input_data = {
        "error_text": req.log_text or "General execution failure error",
        "stage": req.stage or "synthesis",
    }
    res = bundle.predict_top_k_causes(input_data, k=3)
    res["data_source"] = DATA_SOURCE
    return res


@app.post("/detect/anomaly")
def detect_anomaly(
    req: AnomalyDetectRequest, auth: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    model_p = os.path.join(MODELS_DIR, "anomaly_model.joblib")
    if not os.path.exists(model_p):
        raise HTTPException(status_code=503, detail="Anomaly detector is not trained yet.")

    bundle = joblib.load(model_p)
    input_data = {
        "design_size": req.design_size or 100_000,
        "runtime_s": req.runtime_s or 120.0,
        "peak_memory_mb": req.peak_memory_mb or 800.0,
        "warnings_count": req.warnings_count or 10,
        "error_text": req.log_text or "",
    }
    res = bundle.detect(input_data)
    res["data_source"] = DATA_SOURCE
    return res
