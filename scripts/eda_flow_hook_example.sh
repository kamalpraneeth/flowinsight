#!/usr/bin/env bash
# ==============================================================================
# Illustrative Design-Flow Hook Example
#
# IMPORTANT NOTICE:
# This script is an illustrative EXAMPLE demonstrating how a chip design automation
# harness (e.g. Makefile or Tcl runner) could invoke FlowInsight CLI after an EDA job finishes.
# It is NOT connected to a real commercial EDA tool (Synopsys, Cadence, or Siemens).
# Synthetic data generated for demonstration. Not real EDA tool logs.
# ==============================================================================

set -euo pipefail

LOG_FILE="${1:-data/logs/sample_run.log}"
DESIGN="${2:-riscv_core}"
STAGE="${3:-place_and_route}"

echo "========================================================================"
echo " [EXAMPLE HOOK] Post-Execution Flow Hook for Design: ${DESIGN}"
echo " Stage: ${STAGE} | Log: ${LOG_FILE}"
echo " NOTICE: Illustrative demonstration only. Not connected to real EDA tool."
echo "========================================================================"

# Determine Python executable
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python &>/dev/null; then
        PYTHON_BIN="python"
    elif command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
    elif command -v py &>/dev/null; then
        PYTHON_BIN="py"
    else
        PYTHON_BIN="python"
    fi
fi

if [[ ! -f "${LOG_FILE}" ]]; then
    echo "[-] Error: Log file '${LOG_FILE}' not found. Exiting hook." >&2
    exit 0  # Non-blocking for tapeout flow
fi

# 1. Parse log with FlowInsight CLI
echo ">>> Invoking FlowInsight to parse log telemetry..."
PARSED_JSON=$("${PYTHON_BIN}" -m flowinsight.cli parse --log "${LOG_FILE}")
STATUS=$(echo "${PARSED_JSON}" | grep -o '"status": "[^"]*' | cut -d'"' -f4 || echo "unknown")

echo "    Detected Job Status: ${STATUS}"

# 2. If job failed, invoke root-cause ranker
if [[ "${STATUS}" == "failed" ]]; then
    echo ""
    echo "[!] Run failed. Querying FlowInsight Failure Classifier & Root-Cause Ranker..."
    CLASSIFICATION=$("${PYTHON_BIN}" -m flowinsight.cli classify-failure --log "${LOG_FILE}" --top-k 3 || true)
    
    echo "    Root Cause Diagnosis:"
    echo "${CLASSIFICATION}"
    
    # 3. Check for anomalies
    echo ""
    echo ">>> Running Anomaly Detection check on failed execution..."
    ANOMALY_INFO=$("${PYTHON_BIN}" -m flowinsight.cli detect-anomalies --log "${LOG_FILE}" || true)
    echo "${ANOMALY_INFO}"
    
    echo "========================================================================"
    echo "[!] Notification dispatched to engineer queue. Log diagnosis complete."
    echo "========================================================================"
else
    echo "[+] Run completed nominally. Querying runtime predictor for next step..."
    "${PYTHON_BIN}" -m flowinsight.cli predict-runtime --size 250000 --stage "${STAGE}" --threads 8 || true
fi

exit 0
