# Launch the Sensor Web Visualizer
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = "python"
if (Test-Path -LiteralPath "$PSScriptRoot\.venv\Scripts\python.exe") {
    $python = "$PSScriptRoot\.venv\Scripts\python.exe"
}

Write-Host "Starting Sensor API Server on http://127.0.0.1:8080 ..." -ForegroundColor Cyan
& $python -m slam.sensor_api --recordings recordings --host 127.0.0.1 --port 8080
