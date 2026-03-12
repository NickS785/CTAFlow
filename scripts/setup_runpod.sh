Ok#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash setup_runpod.sh <FRED_API_KEY>"
    exit 1
fi

export ROOT_DIR=/workspace
cd "$ROOT_DIR/CTAFlow" && git stash && git pull
cd "$ROOT_DIR"
source venv/bin/activate
pip install -e CTAFlow
python -m ipykernel install --user --name venv --display-name "CTAVenv"

echo "Env set. Environment ready."
