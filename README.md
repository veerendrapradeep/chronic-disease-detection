# Chronic Disease Detection System

End-to-end machine learning project for chronic disease risk prediction with:
- model training and evaluation pipeline,
- FastAPI inference service,
- Streamlit demo interface,
- structured recommendation output for lifestyle and safety guidance,
- automated validation scripts for handoff confidence.

## Core Features
- Data preprocessing and feature engineering pipeline for numeric and categorical clinical inputs.
- Multi-model training with model selection and threshold tuning.
- Explainability artifact generation (permutation and coefficient-based importance).
- FastAPI endpoints for health and prediction.
- Streamlit UI for manual prediction and multi-file JSON uploads.
- Optional API key and optional Streamlit password protection.
- Contract tests, smoke tests, and one-command review validation.

## Technology Stack
- Python
- scikit-learn
- FastAPI
- Streamlit
- unittest
- PowerShell automation

## Repository Layout
- `src/`: ML training, evaluation, explainability, and prediction scripts
- `app/`: FastAPI app and Streamlit app
- `data/`: raw and processed data, including demo JSON patient files
- `tests/`: API, prediction, recommendations, and batch contract tests
- `reports/`: generated evaluation and validation outputs
- `docs/`: final report source (`final_report_full.tex`)
- `USER_MANUAL.md`: non-technical usage guide for group/demo users

## Quick Setup
1. Create and activate a virtual environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Generate data and train model.

```bash
python -m src.generate_sample_data
python -m src.train
```

4. Optional: generate explainability outputs.

```bash
python -m src.explain
```

## Run The Applications

### Run API
```bash
uvicorn app.api:app --reload
```

Useful API URLs:
- `/` API home
- `/health` health check
- `/docs` interactive Swagger docs
- `/predict` prediction endpoint (POST)

### Run Streamlit
```bash
streamlit run app/streamlit_app.py
```

If port 8501 is busy, run on another port:

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

## Optional Security Modes

Enable API key requirement:

```powershell
$env:API_ACCESS_KEY="your-api-key"
uvicorn app.api:app --reload
```

Enable Streamlit shared-password gate:

```powershell
$env:APP_DEMO_PASSWORD="your-demo-password"
streamlit run app/streamlit_app.py
```

## Prediction Usage

### Single JSON prediction (CLI)
```bash
python -m src.predict --input data/raw/sample_patient.json
```

### Batch prediction over demo files
```bash
python -m src.predict_batch --input-dir data/raw/demo_patients --output reports/demo_batch_predictions.json
```

## Validation and Testing

### Smoke test
```bash
python smoke_test.py
```

### Full contract test suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### One-command review validation
```powershell
./run_review_checks.ps1 -PythonExe ".\.venv\Scripts\python.exe"
```

This script runs:
- syntax checks for API and Streamlit entrypoints,
- smoke test,
- full test suite,
- batch prediction generation.

Primary validation outputs:
- `reports/review_validation.txt`
- `reports/demo_batch_predictions.json`

## Key Generated Artifacts
- `models/best_model.joblib`
- `reports/model_metrics.csv`
- `reports/model_metrics.json`
- `reports/confusion_matrix.png`
- `reports/roc_curves.png`
- `reports/feature_importance_permutation.csv`
- `reports/feature_importance_coefficients.csv`

## Notes
- This project is designed for educational and decision-support use.
- It does not replace professional clinical diagnosis or treatment.
