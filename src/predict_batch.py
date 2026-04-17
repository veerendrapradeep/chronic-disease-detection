import argparse
import json
from pathlib import Path
from typing import Dict, List

from .config import BEST_MODEL_FILE, DEMO_BATCH_PREDICTIONS_JSON, DEMO_PATIENTS_DIR
from .predict import predict


def get_patient_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted([path for path in input_dir.glob("*.json") if path.is_file()])
    if not json_files:
        raise ValueError(f"No JSON files found in directory: {input_dir}")

    return json_files


def run_batch_predictions(input_dir: Path, model_path: Path) -> List[Dict]:
    files = get_patient_files(input_dir)
    results: List[Dict] = []

    for file_path in files:
        prediction = predict(input_path=file_path, model_path=model_path)
        results.append(
            {
                "input_file": str(file_path),
                **prediction,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch risk prediction for JSON files in a directory.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEMO_PATIENTS_DIR,
        help="Directory containing patient JSON files.",
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
        default=DEMO_BATCH_PREDICTIONS_JSON,
        help="Path to save combined batch prediction JSON.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    batch_results = run_batch_predictions(input_dir=args.input_dir, model_path=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(batch_results, indent=2), encoding="utf-8")

    print(f"Batch predictions completed for {len(batch_results)} files.")
    print(f"Saved output: {args.output}")


if __name__ == "__main__":
    main()
