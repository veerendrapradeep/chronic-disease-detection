param(
    [string]$PythonExe = ".\\.venv\\Scripts\\python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$reportPath = "reports/review_validation.txt"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$pythonPath = (Resolve-Path $PythonExe).Path

function Invoke-LoggedProcess {
    param(
        [string]$StepName,
        [string]$FilePath,
        [string[]]$Arguments
    )

    Write-Host $StepName
    Add-Content -Path $reportPath -Value $StepName -Encoding utf8

    $tempStdOutPath = Join-Path $env:TEMP ("review_stdout_" + [Guid]::NewGuid().ToString() + ".log")
    $tempStdErrPath = Join-Path $env:TEMP ("review_stderr_" + [Guid]::NewGuid().ToString() + ".log")

    try {
        $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -NoNewWindow -PassThru -Wait -RedirectStandardOutput $tempStdOutPath -RedirectStandardError $tempStdErrPath

        if (Test-Path $tempStdOutPath) {
            $stdoutLines = Get-Content $tempStdOutPath
            foreach ($line in $stdoutLines) {
                Write-Host $line
                Add-Content -Path $reportPath -Value $line -Encoding utf8
            }
        }

        if (Test-Path $tempStdErrPath) {
            $stderrLines = Get-Content $tempStdErrPath
            if ($stderrLines.Count -gt 0) {
                Write-Host "[command output]"
                Add-Content -Path $reportPath -Value "[command output]" -Encoding utf8
                foreach ($line in $stderrLines) {
                    Write-Host $line
                    Add-Content -Path $reportPath -Value $line -Encoding utf8
                }
            }
        }

        if ($process.ExitCode -ne 0) {
            throw "Command failed with exit code ${process.ExitCode}: $FilePath $($Arguments -join ' ')"
        }
    }
    finally {
        if (Test-Path $tempStdOutPath) {
            Remove-Item $tempStdOutPath -Force
        }
        if (Test-Path $tempStdErrPath) {
            Remove-Item $tempStdErrPath -Force
        }
    }
}

"Review validation run started: $timestamp" | Out-File -FilePath $reportPath -Encoding utf8
"Python executable: $pythonPath" | Out-File -FilePath $reportPath -Append -Encoding utf8
"" | Out-File -FilePath $reportPath -Append -Encoding utf8

Invoke-LoggedProcess -StepName "[1/4] Running syntax checks for app entrypoints..." -FilePath $pythonPath -Arguments @("-m", "py_compile", "app/api.py", "app/streamlit_app.py")

Invoke-LoggedProcess -StepName "[2/4] Running smoke test..." -FilePath $pythonPath -Arguments @("smoke_test.py")

Invoke-LoggedProcess -StepName "[3/4] Running unittest contract suite..." -FilePath $pythonPath -Arguments @("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v")

Invoke-LoggedProcess -StepName "[4/4] Running demo batch predictions..." -FilePath $pythonPath -Arguments @("-m", "src.predict_batch", "--input-dir", "data/raw/demo_patients", "--output", "reports/demo_batch_predictions.json")

if (Test-Path "reports/demo_batch_predictions.json") {
    "Generated: reports/demo_batch_predictions.json" | Out-File -FilePath $reportPath -Append -Encoding utf8
} else {
    throw "Expected batch output not found: reports/demo_batch_predictions.json"
}

"" | Out-File -FilePath $reportPath -Append -Encoding utf8
"Review validation run finished." | Out-File -FilePath $reportPath -Append -Encoding utf8

Write-Host "Done. Validation report: $reportPath"
