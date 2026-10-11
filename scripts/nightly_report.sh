#!/usr/bin/env bash
# ==============================================================================
# FlowInsight Nightly Cron Analytics and Alerting Job
#
# NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
#
# Scheduled cron script to collect daily logs, run analytical queries,
# generate an updated nightly report, and send alert notifications if failure spikes occur.
# ==============================================================================

set -euo pipefail

DATA_DIR="${1:-data}"
REPORTS_DIR="${2:-reports}"
TODAY=$(date -u +"%Y%m%d")
REPORT_PATH="${REPORTS_DIR}/nightly_report_${TODAY}.md"

echo "========================================================================"
echo "          FlowInsight Nightly Analytics Automation"
echo "========================================================================"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs."
echo "========================================================================"

mkdir -p "${REPORTS_DIR}"

# Run end-to-end pipeline
bash scripts/run_pipeline.sh \
    "${DATA_DIR}/manifest.csv" \
    "${DATA_DIR}/flowinsight.db" \
    "models_saved" \
    "${REPORT_PATH}"

# Extract key statistics using grep and sed
echo ""
echo ">>> Checking failure rate in generated nightly report..."
FAIL_RATE=$(grep -E "\- \*\*Overall Job Failure Rate:\*\*" "${REPORT_PATH}" | sed -E 's/.*: ([0-9.]+)%.*/\1/' || echo "0.0")

echo "    Current measured failure rate: ${FAIL_RATE}%"

# Threshold check (alert if failure rate exceeds 30%)
ALERT_THRESHOLD=30.0
IS_SPIKE=$(awk -v fr="${FAIL_RATE}" -v th="${ALERT_THRESHOLD}" 'BEGIN { if (fr > th) print 1; else print 0 }')

if [[ "${IS_SPIKE}" -eq 1 ]]; then
    echo "========================================================================"
    echo "[!] ALERT: Failure rate (${FAIL_RATE}%) exceeds threshold (${ALERT_THRESHOLD}%)!"
    echo "    Review details in: ${REPORT_PATH}"
    echo "========================================================================"
else
    echo "[+] Failure rate is within normal operating boundaries."
fi

echo ">>> Nightly run complete. Report archived at: ${REPORT_PATH}"
exit 0
