import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import app
from src.config import BEST_MODEL_FILE, SAMPLE_PATIENT_FILE
from src.predict import predict


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_smoke_test() -> None:
    _assert(BEST_MODEL_FILE.exists(), f"Missing trained model artifact: {BEST_MODEL_FILE}")
    _assert(SAMPLE_PATIENT_FILE.exists(), f"Missing sample payload: {SAMPLE_PATIENT_FILE}")

    cli_result = predict(input_path=Path(SAMPLE_PATIENT_FILE), model_path=Path(BEST_MODEL_FILE))
    for key in [
        "model_name",
        "threshold",
        "predicted_probability",
        "predicted_label",
        "risk_category",
        "recommendations",
    ]:
        _assert(key in cli_result, f"CLI prediction missing key: {key}")

    _assert("exercise_plan" in cli_result["recommendations"], "Recommendations missing exercise_plan")
    _assert("foods_to_take_more" in cli_result["recommendations"], "Recommendations missing foods_to_take_more")
    _assert(
        "medication_safety_notes" in cli_result["recommendations"],
        "Recommendations missing medication_safety_notes",
    )

    client = TestClient(app)

    health_response = client.get("/health")
    _assert(health_response.status_code == 200, f"Unexpected /health status: {health_response.status_code}")

    payload = json.loads(Path(SAMPLE_PATIENT_FILE).read_text(encoding="utf-8"))
    expected_api_key = os.getenv("API_ACCESS_KEY", "").strip()
    request_headers = {"x-api-key": expected_api_key} if expected_api_key else {}

    predict_response = client.post("/predict", json=payload, headers=request_headers)
    _assert(predict_response.status_code == 200, f"Unexpected /predict status: {predict_response.status_code}")

    api_result = predict_response.json()
    _assert("recommendations" in api_result, "API response missing recommendations")
    _assert("summary" in api_result["recommendations"], "Recommendations missing summary")

    print("Smoke test passed: CLI prediction and API endpoints are working.")


if __name__ == "__main__":
    run_smoke_test()
