#!/bin/bash

# -------- STOP ALL SERVICES --------
# Stops Flask server and Celery worker cleanly
# ------------------------------------

echo "[INFO] Stopping all processes..."

# Option 1: Use PID file if it exists
PID_FILE=".process_pids"
if [ -f "$PID_FILE" ]; then
    echo "[INFO] Found PID file, stopping processes from file..."
    while read -r pid; do
        if ps -p "$pid" > /dev/null; then
            echo "[INFO] Stopping process $pid"
            kill "$pid"
            # Give it a moment to terminate gracefully
            sleep 2
            # Force kill if still running
            if ps -p "$pid" > /dev/null; then
                echo "[INFO] Process $pid still running, forcing termination..."
                kill -9 "$pid"
            fi
        else
            echo "[INFO] Process $pid is not running"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
else
    echo "[INFO] No PID file found, searching for processes by name..."
    
    # Option 2: Find processes by their command patterns
    echo "[INFO] Stopping Flask server..."
    pkill -f "python app.py" || echo "[INFO] No Flask server found"
    
    echo "[INFO] Stopping Celery worker..."
    pkill -f "celery -A celery_worker worker" || echo "[INFO] No Celery worker found"
    
    # Wait a moment
    sleep 2
    
    # Force kill any remaining processes
    if pgrep -f "python app.py" > /dev/null || pgrep -f "celery -A celery_worker worker" > /dev/null; then
        echo "[INFO] Some processes still running, forcing termination..."
        pkill -9 -f "python app.py" || true
        pkill -9 -f "celery -A celery_worker worker" || true
    fi
fi

echo "[✅] All services stopped" 