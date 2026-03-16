#!/bin/bash

echo "================================="
echo "LLM Trading Agent - Project Runner"
echo "================================="

echo ""
echo "Checking Python..."
python3 --version

echo ""
echo "Installing dependencies..."
pip3 install -r requirements.txt

echo ""
echo "Select mode:"
echo "1 - Run Signal Generator"
echo "2 - Run Backtest"
echo "3 - Run Paper Trading"

read -p "Enter option (1/2/3): " mode

if [ "$mode" = "1" ]; then
    echo "Running live signal..."
    python3 scripts/run_live_signal.py
fi

if [ "$mode" = "2" ]; then
    echo "Running backtest..."
    python3 scripts/run_backtest.py
fi

if [ "$mode" = "3" ]; then
    echo "Running paper trading..."
    python3 scripts/run_paper_trade.py
fi

echo ""
echo "Done."