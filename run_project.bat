@echo off
echo =====================================
echo LLM Trading Agent - Project Launcher
echo =====================================

echo.
echo Checking Python installation...
python --version

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Select mode:
echo 1 - Run Signal Generator
echo 2 - Run Backtest
echo 3 - Run Paper Trading
set /p mode=Enter option (1/2/3):

if "%mode%"=="1" (
    echo Running live signal...
    python scripts\run_live_signal.py
)

if "%mode%"=="2" (
    echo Running backtest...
    python scripts\run_backtest.py
)

if "%mode%"=="3" (
    echo Running paper trading...
    python scripts\run_paper_trade.py
)

echo.
echo Finished.
pause