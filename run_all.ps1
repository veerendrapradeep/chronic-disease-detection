param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/5] Generating sample dataset..."
& $PythonExe -m src.generate_sample_data --samples 1500

Write-Host "[2/5] Training models..."
& $PythonExe -m src.train

Write-Host "[3/5] Running sample prediction..."
& $PythonExe -m src.predict --input data/raw/sample_patient.json --output reports/sample_prediction.json

Write-Host "[4/5] Evaluating saved model..."
& $PythonExe -m src.evaluate --output reports/evaluation.json

Write-Host "[5/5] Generating explainability reports..."
& $PythonExe -m src.explain --repeats 10

Write-Host "Pipeline completed. Check models/ and reports/ for outputs."
