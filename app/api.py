import hmac
from functools import lru_cache
import os
from pathlib import Path
import sys

import joblib
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_MODEL_FILE
from src.data_pipeline import build_inference_dataframe
from src.recommendations import build_personalized_recommendations


class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    bmi: float = Field(..., ge=10, le=60)
    systolic_bp: float = Field(..., ge=70, le=250)
    diastolic_bp: float = Field(..., ge=40, le=150)
    glucose: float = Field(..., ge=40, le=400)
    hba1c: float = Field(..., ge=3, le=18)
    cholesterol: float = Field(..., ge=80, le=450)
    creatinine: float = Field(..., ge=0.1, le=8)
    egfr: float = Field(..., ge=5, le=150)
    physical_activity_days: int = Field(..., ge=0, le=7)
    sleep_hours: float = Field(..., ge=2, le=12)
    gender: str
    smoking_status: str
    family_history: str
    diet_quality: str
    alcohol_intake: str


@lru_cache(maxsize=1)
def load_artifact():
    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {BEST_MODEL_FILE}")
    return joblib.load(BEST_MODEL_FILE)


app = FastAPI(title="Chronic Disease Detection API", version="1.0.0")


def require_api_key(x_api_key: str | None) -> None:
    expected_api_key = os.getenv("API_ACCESS_KEY", "").strip()
    if not expected_api_key:
        return

    if not x_api_key or not hmac.compare_digest(x_api_key, expected_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing x-api-key header")


@app.get("/")
def root():
    return {
        "message": "Chronic Disease Detection API is running.",
        "quick_links": {
            "health": "/health",
            "interactive_docs": "/docs",
            "predict_endpoint": "/predict",
        },
        "notes": {
            "predict_method": "POST",
            "api_key_required": bool(os.getenv("API_ACCESS_KEY", "").strip()),
            "api_key_header": "x-api-key",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PatientInput, x_api_key: str | None = Header(default=None, alias="x-api-key")):
    require_api_key(x_api_key)

    try:
        artifact = load_artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload_dict = payload.model_dump()

    try:
        sample_df = build_inference_dataframe(payload_dict)
        pipeline = artifact["pipeline"]
        threshold = float(artifact.get("threshold", 0.5))
        model_name = artifact.get("model_name", "unknown_model")

        probability = float(pipeline.predict_proba(sample_df)[0, 1])
        prediction = int(probability >= threshold)
        recommendations = build_personalized_recommendations(
            payload=payload_dict,
            predicted_probability=probability,
            threshold=threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed. Check model artifact and input schema.") from exc

    return {
        "model_name": model_name,
        "threshold": threshold,
        "predicted_probability": probability,
        "predicted_label": prediction,
        "risk_category": "high_risk" if prediction == 1 else "low_risk",
        "recommendations": recommendations,
    }
