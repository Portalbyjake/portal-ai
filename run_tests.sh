#!/bin/bash

# run_tests.sh - Run all tests for Portal AI

if [ ! -d "venv" ]; then
  echo "[ERROR] Python virtual environment not found. Run 'make setup' first."
  exit 1
fi

source venv/bin/activate
if command -v pytest &> /dev/null; then
  echo "[INFO] Running tests with pytest..."
  pytest
else
  echo "[INFO] Pytest not found. Running all test_*.py scripts..."
  for f in test_*.py; do
    echo "[INFO] Running $f"
    python "$f"
  done
fi 