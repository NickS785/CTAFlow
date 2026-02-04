#!/bin/bash
# Bootstrap script for CTAFlow environment on RunPod
# Root: /workspace/ containing CTAFlow/ and SierraPy/

set -e

ROOT_DIR="/workspace"
VENV_PATH="$ROOT_DIR/venv"

echo "============================================================"
echo "CTAFlow Environment Bootstrap"
echo "============================================================"

# Create venv
echo -e "\n[1/4] Creating virtual environment at $VENV_PATH..."
python3 -m venv "$VENV_PATH" --system-site-packages
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo -e "\n[2/4] Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Git pull in each package
echo -e "\n[3/4] Updating packages (git pull)..."
cd "$ROOT_DIR/SierraPy" && git pull
cd "$ROOT_DIR/CTAFlow" && git pull

# Install packages
echo -e "\n[4/4] Installing packages..."
pip install --ignore-installed -r "$ROOT_DIR/CTAFlow/requirements.txt" || true
pip install -e "$ROOT_DIR/SierraPy"
pip install -e "$ROOT_DIR/CTAFlow"

echo -e "\n============================================================"
echo "Done! Activate with: source $VENV_PATH/bin/activate"
echo "============================================================"
