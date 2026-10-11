-- FlowInsight Analytical SQL Queries
-- NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.

-- Query 1: failure_rate_by_stage_and_week
-- Calculates weekly failure rate trends partitioned by EDA stage
-- name: failure_rate_by_stage_and_week
SELECT 
    stage,
    strftime('%Y-W%W', start_time) AS week_bucket,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_jobs,
    ROUND(CAST(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100.0, 2) AS failure_rate_pct
FROM jobs
GROUP BY stage, week_bucket
ORDER BY week_bucket ASC, stage ASC;

-- Query 2: slowest_designs
-- Ranks chip designs by mean and 95th percentile wallclock execution time
-- name: slowest_designs
SELECT 
    design_name,
    COUNT(*) AS run_count,
    ROUND(AVG(runtime_s), 2) AS avg_runtime_s,
    ROUND(MIN(runtime_s), 2) AS min_runtime_s,
    ROUND(MAX(runtime_s), 2) AS max_runtime_s,
    ROUND(AVG(peak_memory_mb), 1) AS avg_peak_memory_mb
FROM jobs
WHERE status = 'completed'
GROUP BY design_name
ORDER BY avg_runtime_s DESC;

-- Query 3: most_frequent_errors
-- Computes the highest frequency error and fatal message IDs across failed executions
-- name: most_frequent_errors
SELECT 
    m.message_id,
    COUNT(*) AS occurrence_count,
    COUNT(DISTINCT m.job_id) AS distinct_jobs_affected,
    j.stage,
    MAX(m.message_text) AS sample_message_text
FROM messages m
JOIN jobs j ON m.job_id = j.job_id
WHERE m.severity IN ('ERROR', 'FATAL') AND j.status = 'failed'
GROUP BY m.message_id, j.stage
ORDER BY occurrence_count DESC
LIMIT 20;

-- Query 4: runtime_percentiles_by_stage
-- Approximates median, 90th, and 99th runtime percentiles across stages using SQLite ordering
-- name: runtime_percentiles_by_stage
WITH ordered_runs AS (
    SELECT 
        stage,
        runtime_s,
        ROW_NUMBER() OVER (PARTITION BY stage ORDER BY runtime_s) AS row_num,
        COUNT(*) OVER (PARTITION BY stage) AS stage_count
    FROM jobs
    WHERE status = 'completed'
)
SELECT 
    stage,
    stage_count,
    ROUND(MAX(CASE WHEN row_num = CAST(0.50 * stage_count AS INT) THEN runtime_s END), 2) AS p50_runtime_s,
    ROUND(MAX(CASE WHEN row_num = CAST(0.90 * stage_count AS INT) THEN runtime_s END), 2) AS p90_runtime_s,
    ROUND(MAX(CASE WHEN row_num = CAST(0.99 * stage_count AS INT) THEN runtime_s END), 2) AS p99_runtime_s
FROM ordered_runs
GROUP BY stage;

-- Query 5: top_warning_ids_before_failures
-- Aggregates the most common warning message codes logged in executions that subsequently failed
-- name: top_warning_ids_before_failures
SELECT 
    m.message_id,
    j.failure_cause,
    COUNT(*) AS warning_occurrences,
    COUNT(DISTINCT m.job_id) AS failed_runs_affected
FROM messages m
JOIN jobs j ON m.job_id = j.job_id
WHERE m.severity IN ('WARN', 'WARNING') AND j.status = 'failed'
GROUP BY m.message_id, j.failure_cause
ORDER BY warning_occurrences DESC
LIMIT 25;
