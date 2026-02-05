#!/bin/bash
# Bootstrap script for CTAFlow environment on RunPod
# Root: /workspace/ containing CTAFlow/ and SierraPy/
#
# Usage:
#   ./bootstrap_env.sh          # Full setup or update
#   ./bootstrap_env.sh --check  # Just check venv status
#   source bootstrap_env.sh     # Setup and stay in venv

set -e

ROOT_DIR="/workspace"
VENV_PATH="$ROOT_DIR/venv"
VENV_PIP="$VENV_PATH/bin/pip"
VENV_PYTHON="$VENV_PATH/bin/python"
KERNEL_NAME="ctaflow"

echo "============================================================"
echo "CTAFlow Environment Bootstrap"
echo "============================================================"

# Function to activate venv
activate_venv() {
    source "$VENV_PATH/bin/activate"
    echo "Activated venv: $VENV_PATH"
}

# Function to check installed packages
check_packages() {
    echo -e "\n--- Installed Package Versions ---"
    "$VENV_PIP" show CTAFlow 2>/dev/null | grep -E "^(Name|Version):" || echo "CTAFlow: NOT INSTALLED"
    "$VENV_PIP" show SierraPy 2>/dev/null | grep -E "^(Name|Version):" || echo "SierraPy: NOT INSTALLED"
    "$VENV_PIP" show torch 2>/dev/null | grep -E "^(Name|Version):" || echo "torch: NOT INSTALLED"
    "$VENV_PIP" show mamba-ssm 2>/dev/null | grep -E "^(Name|Version):" || echo "mamba-ssm: NOT INSTALLED"
    "$VENV_PIP" show optuna 2>/dev/null | grep -E "^(Name|Version):" || echo "optuna: NOT INSTALLED"
    echo ""
}

# Check if just checking status
if [[ "$1" == "--check" ]]; then
    if [[ -d "$VENV_PATH" ]]; then
        echo "Venv exists at: $VENV_PATH"
        check_packages

        # Check kernel
        if "$VENV_PYTHON" -m jupyter kernelspec list 2>/dev/null | grep -q "$KERNEL_NAME"; then
            echo "Jupyter kernel '$KERNEL_NAME' is registered"
        else
            echo "Jupyter kernel '$KERNEL_NAME' is NOT registered"
        fi
    else
        echo "Venv does NOT exist at: $VENV_PATH"
    fi
    exit 0
fi

# Check if venv already exists
if [[ -d "$VENV_PATH" ]]; then
    echo -e "\nVenv already exists at $VENV_PATH"
    activate_venv

    # Update packages
    echo -e "\n[1/4] Updating pip..."
    "$VENV_PIP" install --upgrade pip wheel setuptools

    echo -e "\n[2/4] Pulling latest code..."
    cd "$ROOT_DIR/SierraPy" && git pull || echo "SierraPy git pull failed (may be ok)"
    cd "$ROOT_DIR/CTAFlow" && git pull || echo "CTAFlow git pull failed (may be ok)"

    echo -e "\n[3/4] Reinstalling local packages..."
    "$VENV_PIP" install -e "$ROOT_DIR/SierraPy" --no-deps
    "$VENV_PIP" install -e "$ROOT_DIR/CTAFlow" --no-deps

    echo -e "\n[4/4] Checking missing dependencies..."
    "$VENV_PIP" install -r "$ROOT_DIR/CTAFlow/requirements.txt" --quiet 2>/dev/null || true

else
    echo -e "\n[1/5] Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH" --system-site-packages
    activate_venv

    echo -e "\n[2/5] Upgrading pip (using venv pip explicitly)..."
    "$VENV_PIP" install --upgrade pip wheel setuptools

    echo -e "\n[3/5] Pulling latest code..."
    cd "$ROOT_DIR/SierraPy" && git pull || echo "SierraPy git pull failed (may be ok)"
    cd "$ROOT_DIR/CTAFlow" && git pull || echo "CTAFlow git pull failed (may be ok)"

    echo -e "\n[4/5] Installing requirements (into venv)..."
    # Use --target or explicit pip path to ensure venv installation
    "$VENV_PIP" install -r "$ROOT_DIR/CTAFlow/requirements.txt" || {
        echo "Some requirements failed, continuing..."
    }

    echo -e "\n[5/5] Installing local packages (editable)..."
    "$VENV_PIP" install -e "$ROOT_DIR/SierraPy"
    "$VENV_PIP" install -e "$ROOT_DIR/CTAFlow"
fi

# Install and register ipykernel
echo -e "\n--- Setting up Jupyter kernel ---"
"$VENV_PIP" install ipykernel --quiet

# Register kernel (use venv python to register)
"$VENV_PYTHON" -m ipykernel install \
    --user \
    --name="$KERNEL_NAME" \
    --display-name="Python (CTAFlow)"

echo "Jupyter kernel '$KERNEL_NAME' registered"

# Verify kernel is available
echo -e "\nAvailable Jupyter kernels:"
"$VENV_PYTHON" -m jupyter kernelspec list 2>/dev/null || jupyter kernelspec list 2>/dev/null || echo "(jupyter not found)"

# Show package status
check_packages

echo "============================================================"
echo "Setup complete!"
echo ""
echo "To activate manually:  source $VENV_PATH/bin/activate"
echo "Jupyter kernel:        $KERNEL_NAME (Python (CTAFlow))"
echo "============================================================"

# Return to root
cd "$ROOT_DIR"
