# User Manual

## 1) What This Project Does

This system estimates chronic disease risk from patient data.
It gives:
- risk probability
- risk category (low or high)
- practical guidance (exercise, food, safety notes)

It is a support tool, not a final medical diagnosis.

## 2) Super Quick Start (2 Minutes)

1. Open PowerShell in the project folder.
2. Start API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

3. Open a second PowerShell window and start Streamlit:

```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

4. Open these URLs:
- Streamlit UI: http://127.0.0.1:8502
- API Home: http://127.0.0.1:8000/
- API Docs: http://127.0.0.1:8000/docs

5. You are ready to demo.

## 3) How To Use The Streamlit App

### Manual Entry

1. Open the "Manual Entry" tab.
2. Fill patient fields.
3. Click "Run Prediction".
4. Read:
- probability
- low/high risk badge
- recommendation tabs

### Upload Multiple Patient Files

1. Open "Upload JSON Cases" tab.
2. Upload one or many JSON files.
3. Click "Run Uploaded Predictions".
4. Review batch summary table.
5. Select a file to see detailed recommendations.
6. Export JSON or CSV if needed.

Demo files are available in:
- data/raw/demo_patients/

## 4) How To Use The API (No Coding Needed)

1. Open http://127.0.0.1:8000/docs
2. Expand POST /predict
3. Click "Try it out"
4. Paste sample input from:
- data/raw/sample_patient.json
5. Click "Execute"
6. Review response fields:
- predicted_probability
- risk_category
- recommendations

## 5) One Command Full Test

Run this before submission:

```powershell
.\run_review_checks.ps1 -PythonExe ".\.venv\Scripts\python.exe"
```

Expected result:
- syntax checks pass
- smoke test passes
- all unit tests pass
- batch prediction file is generated

Main output files:
- reports/review_validation.txt
- reports/demo_batch_predictions.json

## 6) Easy Test Checklist

- API Home opens at http://127.0.0.1:8000/
- API Health returns ok at http://127.0.0.1:8000/health
- Streamlit opens at http://127.0.0.1:8502
- Manual prediction works in Streamlit
- Upload prediction works in Streamlit
- Full validation script passes

## 7) Common Problems and Fast Fixes

### Problem: "Port 8501 is not available"
Use another port:

```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py --server.port 8502
```

### Problem: "Not Found" on API
- http://127.0.0.1:8000/ now shows API home
- http://127.0.0.1:8000/health is health check
- http://127.0.0.1:8000/docs is interactive docs

### Problem: PowerShell blocks script run
Run once in current terminal:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then run the test script again.

## 8) Optional Security Modes

- API key mode:
  - set API_ACCESS_KEY
  - send x-api-key header in /predict requests

- Streamlit password mode:
  - set APP_DEMO_PASSWORD
  - enter password in UI login box

If these environment variables are not set, app runs in open demo mode.
