@echo off
title SandStories Kinect Dashboard
cd /d "%~dp0"
set "PATH=C:\Program Files\Azure Kinect SDK v1.4.2\sdk\windows-desktop\amd64\release\bin;C:\Program Files\Azure Kinect SDK v1.4.2\tools;%PATH%"
set "PYTHONPATH=%~dp0"
"C:\Users\vijit\AppData\Local\Programs\Python\Python312\python.exe" run_dashboard.py --output-dir "C:\Users\vijit\Desktop\AzureKinectRecordings" --log-level INFO
if %errorlevel% neq 0 (
    echo.
    echo Dashboard exited with error code %errorlevel%.
    pause
)
