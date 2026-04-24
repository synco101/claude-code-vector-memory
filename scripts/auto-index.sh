#!/bin/bash
# Auto-index script for Claude Code Hybrid Memory
# Runs incremental indexing of all memory sources
# Designed to be called by launchd every 6 hours

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$BASE_DIR/logs"
LOG_FILE="$LOG_DIR/auto-index-$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

echo "=== Auto-index started: $(date) ===" >> "$LOG_FILE"

if [ ! -d "$BASE_DIR/venv" ]; then
    echo "ERROR: venv not found at $BASE_DIR/venv" >> "$LOG_FILE"
    exit 1
fi

"$BASE_DIR/venv/bin/python" "$SCRIPT_DIR/index_summaries.py" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "Exit code: $EXIT_CODE" >> "$LOG_FILE"
echo "=== Auto-index finished: $(date) ===" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Cleanup logs older than 30 days
find "$LOG_DIR" -name "auto-index-*.log" -mtime +30 -delete 2>/dev/null

exit $EXIT_CODE
