#!/bin/bash

# -------- START ALL SERVICES --------
# Starts Flask server (with HTTPS) and Celery worker
# ------------------------------------

# Store PIDs file
PID_FILE=".process_pids"

# Create logs directory if it doesn't exist
mkdir -p logs

# Remove PID file if it exists
rm -f "$PID_FILE"

# Note: Assuming venv is already activated

echo "[INFO] Running pip install to ensure all dependencies are up to date..."
pip install -r requirements.txt > logs/pip_install.log 2>&1

echo "[INFO] Starting Flask server with HTTPS (self-signed cert) on port 8080..."
nohup python app.py > logs/flask.log 2>&1 &
FLASK_PID=$!
echo $FLASK_PID > "$PID_FILE"
echo "[INFO] Flask server started with PID: $FLASK_PID"

echo "[INFO] Starting Celery worker..."
nohup celery -A celery_worker worker --loglevel=info > logs/celery.log 2>&1 &
CELERY_PID=$!
echo $CELERY_PID >> "$PID_FILE"
echo "[INFO] Celery worker started with PID: $CELERY_PID"

echo "[✅] All services are running."
echo "    - To check Flask log: tail -f logs/flask.log"
echo "    - To check Celery log: tail -f logs/celery.log"
echo "    - To stop services: ./stop_all.sh"
echo "    - Access the application at: https://192.168.167.88:8080"
echo "      (Accept the security warning about the self-signed certificate when prompted)"

# Don't wait - let the script complete while services run in background
# This is better for production use where you might run this from a system service 