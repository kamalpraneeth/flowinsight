#!/usr/bin/env bash
# ==============================================================================
# FlowInsight End-to-End Pipeline Execution Script
#
# NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
#
# Sequences log parsing, database population, model training, and executive reporting.
# ==============================================================================

set -euo pipefail

MANIFEST_PATH="${1:-data/manifest.csv}"
DB_PATH="${2:-data/flowinsight.db}"
MODELS_DIR="${3:-models_saved}"
REPORT_FILE="${4:-reports/nightly_report.md}"

echo "========================================================================"
echo "          FlowInsight End-to-End Analytics Pipeline Run"
echo "========================================================================"
echo "NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs."
echo "Manifest:    ${MANIFEST_PATH}"
echo "Database:    ${DB_PATH}"
echo "Models Dir:  ${MODELS_DIR}"
echo "Report:      ${REPORT_FILE}"
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

# Step 1: Verify data exists or generate smoke batch
if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "[-] Manifest not found. Generating initial synthetic batch (100 jobs)..."
    "${PYTHON_BIN}" -m flowinsight.cli generate --jobs 100 --seed 42 --output-dir "data"
fi

# Step 2: Ingest into SQLite database
echo ""
echo ">>> [Step 1/3] Parsing logs and populating database..."
"${PYTHON_BIN}" -m flowinsight.cli parse --manifest "${MANIFEST_PATH}" --db "${DB_PATH}"

# Step 3: Train predictive & anomaly models
echo ""
echo ">>> [Step 2/3] Training regression, failure classification, and anomaly models..."
"${PYTHON_BIN}" -m flowinsight.cli train --manifest "${MANIFEST_PATH}" --seed 42 --output-dir "${MODELS_DIR}"

# Step 4: Generate executive analytical report and charts
echo ""
echo ">>> [Step 3/3] Generating executive Markdown report and visual charts..."
"${PYTHON_BIN}" -m flowinsight.cli report --manifest "${MANIFEST_PATH}" --output "${REPORT_FILE}"

echo ""
echo "========================================================================"
echo "[+] FlowInsight pipeline completed successfully."
echo "    Database: ${DB_PATH}"
echo "    Report:   ${REPORT_FILE}"
echo "========================================================================"
exit 0
