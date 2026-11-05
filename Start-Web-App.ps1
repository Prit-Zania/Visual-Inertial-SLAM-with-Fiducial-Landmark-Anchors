# Launch the Unified Visual-Inertial SLAM Web Application & Phone Receiver
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = "python"
if (Test-Path -LiteralPath "$PSScriptRoot\.venv\Scripts\python.exe") {
    $python = "$PSScriptRoot\.venv\Scripts\python.exe"
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   IE415: Visual-Inertial SLAM & Web Application Studio   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting Unified Web Server & Phone Receiver on http://127.0.0.1:8080 ..." -ForegroundColor Green

# Open browser after 1.5 seconds in background
Start-Job -ScriptBlock {
    Start-Sleep -Milliseconds 1500
    Start-Process "http://127.0.0.1:8080"
} | Out-Null

& $python -m slam.sensor_api --recordings recordings --host 127.0.0.1 --port 8080

