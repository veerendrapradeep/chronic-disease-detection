from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

DATA_FILE = DATA_RAW_DIR / "chronic_disease.csv"
SAMPLE_PATIENT_FILE = DATA_RAW_DIR / "sample_patient.json"
DEMO_PATIENTS_DIR = DATA_RAW_DIR / "demo_patients"
BEST_MODEL_FILE = MODEL_DIR / "best_model.joblib"
METRICS_CSV_FILE = REPORT_DIR / "model_metrics.csv"
METRICS_JSON_FILE = REPORT_DIR / "model_metrics.json"
CONFUSION_MATRIX_PLOT = REPORT_DIR / "confusion_matrix.png"
ROC_CURVE_PLOT = REPORT_DIR / "roc_curves.png"
PERMUTATION_IMPORTANCE_CSV = REPORT_DIR / "feature_importance_permutation.csv"
PERMUTATION_IMPORTANCE_PLOT = REPORT_DIR / "feature_importance_permutation.png"
COEFFICIENT_IMPORTANCE_CSV = REPORT_DIR / "feature_importance_coefficients.csv"
COEFFICIENT_IMPORTANCE_PLOT = REPORT_DIR / "feature_importance_coefficients.png"
DEMO_BATCH_PREDICTIONS_JSON = REPORT_DIR / "demo_batch_predictions.json"

TARGET_COLUMN = "has_disease"
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_TARGET_RECALL = 0.85

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "hba1c",
    "cholesterol",
    "creatinine",
    "egfr",
    "physical_activity_days",
    "sleep_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "smoking_status",
    "family_history",
    "diet_quality",
    "alcohol_intake",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
