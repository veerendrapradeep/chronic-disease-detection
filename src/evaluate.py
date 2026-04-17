import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import BEST_MODEL_FILE, DATA_FILE, FEATURE_COLUMNS, TARGET_COLUMN
from .data_pipeline import compute_metrics, validate_training_columns


def evaluate_model(model_path: Path, data_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    threshold = float(artifact.get("threshold", 0.5))

    df = pd.read_csv(data_path)
    validate_training_columns(df)

    X = df[FEATURE_COLUMNS]
    y_true = df[TARGET_COLUMN].astype(int).to_numpy()
    y_prob = pipeline.predict_proba(X)[:, 1]

    metrics = compute_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    metrics["threshold"] = threshold
    metrics["model_name"] = artifact.get("model_name", "unknown_model")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved model on a dataset.")
    parser.add_argument("--model", type=Path, default=BEST_MODEL_FILE, help="Saved model artifact path.")
    parser.add_argument("--data", type=Path, default=DATA_FILE, help="Dataset CSV path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON file path.")
    args = parser.parse_args()

    metrics = evaluate_model(model_path=args.model, data_path=args.data)
    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
