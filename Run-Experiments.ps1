# Execute all 5 IE415 Mandatory Course Experiments and generate Master Scorecard
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = "python"
if (Test-Path -LiteralPath "$PSScriptRoot\.venv\Scripts\python.exe") {
    $python = "$PSScriptRoot\.venv\Scripts\python.exe"
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "     IE415: Running All 5 Benchmark SLAM Experiments      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

& $python -m slam.experiments --scenario all --output demo_output
if ($LASTEXITCODE -ne 0) { throw 'Experiments execution failed.' }

Write-Host "`nAll 5 experiments completed successfully!" -ForegroundColor Green
Write-Host "Results saved to demo_output/ and viewable in http://127.0.0.1:8080" -ForegroundColor Yellow

