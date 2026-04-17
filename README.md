# Chronic Disease Detection System

A machine learning project for early chronic disease risk prediction using patient health indicators.

## Objective
Build a practical pipeline that:
- preprocesses patient records,
- trains multiple ML models,
- compares clinical metrics,
- explains predictions,
- provides a demo interface for risk prediction,
- and returns personalized safety-focused guidance (exercise, food, avoid list, daily actions, natural-support options, medication safety notes).

## 2-Day Sprint Plan
### Day 1
1. Finalize problem scope and target label.
2. Build data schema and preprocessing pipeline.
3. Train baseline and advanced models.
4. Save model artifacts and evaluation metrics.

### Day 2
1. Build demo app for prediction.
2. Add explainability and visual reports.
3. Prepare viva-ready documentation.
4. Add production-readiness checklist and architecture notes.

## Quick Start
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Generate sample dataset (for immediate demo):

```bash
python -m src.generate_sample_data
```

4. Train and evaluate models:

```bash
python -m src.train
```

5. Run single prediction from sample JSON:

```bash
python -m src.predict --input data/raw/sample_patient.json
```

6. Generate explainability reports:

```bash
python -m src.explain
```

7. Launch Streamlit demo:

```bash
streamlit run app/streamlit_app.py
```

Optional: protect Streamlit demo with a shared password:

```powershell
$env:APP_DEMO_PASSWORD="your-demo-password"
streamlit run app/streamlit_app.py
```

8. Optional one-command pipeline run (PowerShell):

```powershell
./run_all.ps1 -PythonExe "python"
```

9. Optional API run:

```bash
uvicorn app.api:app --reload
```

Optional: require API key header (`x-api-key`) for `/predict`:

```powershell
$env:API_ACCESS_KEY="your-api-key"
uvicorn app.api:app --reload
```

10. Run smoke test:

```bash
python smoke_test.py
```

11. Run full review validation pack (smoke + contract tests):

```powershell
./run_review_checks.ps1 -PythonExe ".\.venv\Scripts\python.exe"
```

This one command now validates:
- app syntax (`app/api.py`, `app/streamlit_app.py`)
- smoke test
- full unittest contract suite
- batch prediction generation (`reports/demo_batch_predictions.json`)

12. Run batch prediction for demo patient files:

```bash
python -m src.predict_batch --input-dir data/raw/demo_patients --output reports/demo_batch_predictions.json
```

## Project Structure
- `src/` core ML pipeline and scripts
- `app/` user-facing demo app
- `data/` raw and processed datasets
- `data/raw/demo_patients/` ready JSON patient cases for live upload demo
- `models/` trained model artifacts
- `reports/` metrics and visualizations
- `docs/` final report source (`final_report_full.tex`)
- `notebooks/` exploratory and presentation notebook

## Generated Outputs
- `reports/model_metrics.csv` model comparison table
- `reports/confusion_matrix.png` confusion matrix for best model
- `reports/roc_curves.png` ROC curves for candidate models
- `reports/feature_importance_permutation.csv` global permutation feature importance
- `reports/feature_importance_coefficients.csv` logistic coefficient importance

## Personalized Guidance Feature
The prediction output now includes a structured recommendations block designed for educational safety support. It contains:

- exercise plan
- foods to take more
- foods to limit or avoid
- daily action plan
- natural-support options
- medication safety notes
- red-flag escalation advice

This guidance is educational and does not replace professional diagnosis or treatment.

## Interactive Demo Workflow
For a stronger live demo than only smoke test:

1. Open Streamlit (`streamlit run app/streamlit_app.py`)
2. Go to **Upload JSON Cases** tab
3. Upload files from `data/raw/demo_patients/`
4. Run predictions and show:
	- batch summary table
	- detailed guidance per selected file
	- JSON/CSV result downloads

## Automated Validation
The project includes automated proof checks for review:

- `smoke_test.py` for quick end-to-end flow validation
- `tests/test_api_contract.py` for API contract checks
- `tests/test_predict_contract.py` for prediction output contract checks
- `tests/test_recommendations_contract.py` for recommendation structure and safety checks
- `tests/test_demo_patient_cases.py` for demo patient case probability ordering checks
- `tests/test_predict_batch_contract.py` for batch prediction contract checks
- `run_review_checks.ps1` to generate a combined validation log in `reports/review_validation.txt`

## Final Submission Docs
- `docs/final_report_full.tex`

## Clinical Warning
This project is for educational and decision-support purposes only. It is not a standalone medical diagnostic system.
