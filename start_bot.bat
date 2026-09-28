@echo off
rem Double-click to turn the trading bot ON. Close this window to turn it OFF.
title Trading bot (close this window to stop)
cd /d "%~dp0"
python trade.py --loop
pause
