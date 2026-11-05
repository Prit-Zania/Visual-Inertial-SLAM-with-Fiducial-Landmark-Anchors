$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $projectPython -m slam.simulate --output demo_output/synthetic
if ($LASTEXITCODE -ne 0) { throw 'Synthetic data generation failed.' }
& $projectPython -m slam.cli process demo_output/synthetic --config demo_output/synthetic/config.json --map demo_output/synthetic/surveyed_map.json
if ($LASTEXITCODE -ne 0) { throw 'SLAM processing failed.' }
& $projectPython -m slam.cli process demo_output/synthetic --config demo_output/synthetic/config.json --baseline --no-smooth
if ($LASTEXITCODE -ne 0) { throw 'Baseline processing failed.' }
& $projectPython -m slam.cli evaluate demo_output/synthetic/processed/trajectory.csv demo_output/synthetic/ground_truth.csv
& $projectPython -m slam.report demo_output/synthetic
