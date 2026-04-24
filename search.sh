#!/bin/bash
# Search script for Claude Code Hybrid Memory
# Usage: ./search.sh <query> [--source summary|topic|daily] [--compact]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Error: Virtual environment not found. Run: cd $SCRIPT_DIR && python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

"$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/scripts/memory_search.py" "$@"
