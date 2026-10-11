#!/usr/bin/env bash
# ==============================================================================
# FlowInsight Log Collection Automation Script
#
# NOTICE: Synthetic data generated for demonstration. Not real EDA tool logs.
#
# Traverses a source directory tree, identifies EDA tool log artifacts (.log, .txt),
# skips unreadable or corrupted files, and copies them to a standardized target destination.
# ==============================================================================

set -euo pipefail

# Default arguments or environment variables
SRC_DIR="${1:-data/logs}"
DEST_DIR="${2:-data/collected_logs}"
MIN_BYTES="${3:-50}"

echo ">>> [FlowInsight] Initiating log collection pipeline..."
echo "    Source directory: ${SRC_DIR}"
echo "    Target directory: ${DEST_DIR}"

if [[ ! -d "${SRC_DIR}" ]]; then
    echo "[-] Error: Source directory '${SRC_DIR}' does not exist." >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

# Find candidate log files, filter for readable and non-empty files
echo ">>> Scanning directory tree for EDA log files..."
TOTAL_FOUND=0
COPIED_COUNT=0
SKIPPED_COUNT=0

# Use find to locate .log and .txt files
while IFS= read -r -d '' logfile; do
    TOTAL_FOUND=$((TOTAL_FOUND + 1))
    
    # Check readability
    if [[ ! -r "${logfile}" ]]; then
        echo "[-] Skipping unreadable file: ${logfile}" >&2
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi
    
    # Check minimum file size
    FILE_SIZE=$(wc -c < "${logfile}" | awk '{print $1}')
    if [[ "${FILE_SIZE}" -lt "${MIN_BYTES}" ]]; then
        echo "[-] Skipping undersized/truncated stub (${FILE_SIZE} bytes): ${logfile}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi
    
    # Check for FlowInsight or EDA marker in first 10 lines
    if head -n 10 "${logfile}" | grep -Eq "(EDA Tool Suite|EXECUTION SUMMARY|NOTICE: Synthetic data)"; then
        BASENAME=$(basename "${logfile}")
        cp "${logfile}" "${DEST_DIR}/${BASENAME}"
        COPIED_COUNT=$((COPIED_COUNT + 1))
    else
        # Copy with fallback notice
        BASENAME=$(basename "${logfile}")
        cp "${logfile}" "${DEST_DIR}/${BASENAME}"
        COPIED_COUNT=$((COPIED_COUNT + 1))
    fi
done < <(find "${SRC_DIR}" -type f \( -name "*.log" -o -name "*.txt" \) -print0)

echo ">>> Log collection completed successfully:"
echo "    Total candidate files found: ${TOTAL_FOUND}"
echo "    Successfully copied:         ${COPIED_COUNT}"
echo "    Skipped (unreadable/empty):  ${SKIPPED_COUNT}"
echo "    Output directory:            ${DEST_DIR}"

exit 0
