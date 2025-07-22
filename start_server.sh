#!/bin/bash

# Kill any existing Python processes on port 8081
echo "Checking for existing processes on port 8081..."
lsof -ti:8081 | xargs kill -9 2>/dev/null || true

# Wait a moment for processes to fully terminate
sleep 2

# Activate virtual environment and start server
echo "Starting Flask server..."
source venv/bin/activate
python main.py 