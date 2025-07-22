#!/bin/bash

# start.sh - Start the Portal AI server

if [ ! -d "venv" ]; then
  echo "[ERROR] Python virtual environment not found. Run 'make setup' first."
  exit 1
fi

source venv/bin/activate
python main.py 