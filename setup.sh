#!/bin/bash
echo "==================================================="
echo "  ATS Resume Analyzer - Teammate Setup Script"
echo "==================================================="

if ! command -v python3 &> /dev/null
then
    echo "[ERROR] python3 could not be found. Please install it."
    exit 1
fi

echo "[1/3] Creating virtual environment (.venv)..."
python3 -m venv .venv

echo "[2/3] Activating virtual environment..."
source .venv/bin/activate

echo "[3/3] Installing required packages..."
pip install -r requirements.txt

echo ""
echo "==================================================="
echo "  Setup Complete!"
echo "==================================================="
echo ""
echo "Your environment is ready. To test the pipeline, run:"
echo ""
echo "  source .venv/bin/activate"
echo "  python run_submission.py"
echo ""
