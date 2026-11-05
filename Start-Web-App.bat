@echo off
title IE415 Visual-Inertial SLAM Web Studio
cd /d "%~dp0"
echo ==========================================================
echo    IE415: Visual-Inertial SLAM & Web Application Studio
echo ==========================================================
echo Starting Unified Web Server & Phone Receiver on http://127.0.0.1:8080 ...

REM Open default browser
start "" "http://127.0.0.1:8080"

REM Launch Python Server
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m slam.sensor_api --recordings recordings --host 0.0.0.0 --port 8080
) else (
    python -m slam.sensor_api --recordings recordings --host 0.0.0.0 --port 8080
)
pause

