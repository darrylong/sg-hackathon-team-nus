#!/usr/bin/env bash
# One command to set up (first run) and run the dashboard: API + built frontend on http://localhost:${PORT:-8000}
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
PY="${PYTHON:-python3}"

# 1. Python 3.9+
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found — install Python 3.9 or newer (https://www.python.org/downloads/)"
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "Python 3.9+ required, found $("$PY" --version 2>&1)"
  exit 1
fi

# 2. Data: ../05_channel_analytics/data by default, or CHANNEL_DATA_DIR
DATA_DIR="${CHANNEL_DATA_DIR:-../05_channel_analytics/data}"
for f in partner_sales.csv partner_master.csv targets.csv; do
  if [ ! -f "$DATA_DIR/$f" ]; then
    echo "Missing $DATA_DIR/$f"
    echo "Put the pack's 05_channel_analytics folder next to this folder, or run: CHANNEL_DATA_DIR=/path/to/data ./start.sh"
    exit 1
  fi
done
export CHANNEL_DATA_DIR="$DATA_DIR"

# 3. Virtual environment (recreated if it was copied from another machine and no longer runs)
if [ -d .venv ] && ! .venv/bin/python -c 'import sys' >/dev/null 2>&1; then
  echo "Existing .venv does not work on this machine — recreating it..."
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  echo "Creating Python environment (one-time)..."
  "$PY" -m venv .venv
fi
if ! .venv/bin/python -c "import fastapi, uvicorn, statsmodels, sklearn, pandas, pyarrow, matplotlib" >/dev/null 2>&1; then
  echo "Installing Python packages (one-time, needs internet, ~1-2 min)..."
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
fi

# 4. Frontend (prebuilt dist/ is shipped; Node is only needed if it is missing)
if [ ! -f frontend/dist/index.html ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "Building frontend (one-time, ~30s)..."
    (cd frontend && npm install --no-audit --no-fund && npm run build)
  else
    echo "frontend/dist missing and npm not found — install Node.js first"
    exit 1
  fi
fi

# 5. Run
exec .venv/bin/python -m src.serve --port "$PORT"
