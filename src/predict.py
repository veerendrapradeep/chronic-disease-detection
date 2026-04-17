import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib

from .config import BEST_MODEL_FILE, FEATURE_COLUMNS, SAMPLE_PATIENT_FILE
from .data_pipeline import build_inference_dataframe
from .recommendations import build_personalized_recommendations


def load_payload(input_path: Path) -> Dict[str, Any]:
    return json.loads(input_path.read_text(encoding="utf-8"))


def validate_payload(payload: Dict[str, Any]) -> None:
    missing_fields = [feature for feature in FEATURE_COLUMNS if feature not in payload]
    if missing_fields:
        raise ValueError(f"Input payload is missing required fields: {', '.join(missing_fields)}")


def predict(input_path: Path, model_path: Path) -> Dict[str, Any]:
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    threshold = float(artifact.get("threshold", 0.5))
    model_name = artifact.get("model_name", "unknown_model")

    payload = load_payload(input_path=input_path)
    validate_payload(payload)
    sample_df = build_inference_dataframe(payload=payload)

    probability = float(pipeline.predict_proba(sample_df)[0, 1])
    prediction = int(probability >= threshold)
    recommendations = build_personalized_recommendations(
        payload=payload,
        predicted_probability=probability,
        threshold=threshold,
    )

    return {
        "model_name": model_name,
        "threshold": threshold,
        "predicted_probability": probability,
        "predicted_label": prediction,
        "risk_category": "high_risk" if prediction == 1 else "low_risk",
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chronic disease risk prediction.")
    parser.add_argument(
        "--input",
        type=Path,
        default=SAMPLE_PATIENT_FILE,
        help="Path to patient JSON input.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=BEST_MODEL_FILE,
        help="Path to saved model artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save prediction result JSON.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    result = predict(input_path=args.input, model_path=args.model)
    print(json.dumps(result, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
