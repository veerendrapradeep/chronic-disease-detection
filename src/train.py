import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from .config import (
    BEST_MODEL_FILE,
    CONFUSION_MATRIX_PLOT,
    DATA_FILE,
    DEFAULT_TARGET_RECALL,
    FEATURE_COLUMNS,
    METRICS_CSV_FILE,
    METRICS_JSON_FILE,
    RANDOM_STATE,
    REPORT_DIR,
    ROC_CURVE_PLOT,
    SAMPLE_PATIENT_FILE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .data_pipeline import (
    build_model_pipeline,
    choose_threshold_for_recall,
    compute_metrics,
    ensure_project_directories,
    get_model_candidates,
    validate_training_columns,
)
from .generate_sample_data import generate_and_save_dataset


def load_or_generate_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Generating synthetic dataset...")
        generate_and_save_dataset(csv_path=data_path, sample_patient_path=SAMPLE_PATIENT_FILE)

    df = pd.read_csv(data_path)
    validate_training_columns(df)
    return df


def save_metrics(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    metrics_rows = []

    for model_name, result in results.items():
        row = {
            "model": model_name,
            "threshold": result["threshold"],
            **result["metrics"],
        }
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["recall", "roc_auc", "f1", "precision"],
        ascending=False,
    )

    METRICS_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV_FILE, index=False)
    METRICS_JSON_FILE.write_text(
        json.dumps(metrics_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    return metrics_df


def plot_confusion_matrix(y_true, y_prob, threshold: float, output_path: Path) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Disease", "Disease"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Best Model Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true, results: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, result in results.items():
        y_prob = result["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_training_summary(best_model_name: str, best_metrics: Dict[str, float], data_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    summary_lines = [
        f"Training timestamp (UTC): {timestamp}",
        f"Dataset path: {data_path}",
        f"Best model: {best_model_name}",
        f"Accuracy: {best_metrics['accuracy']:.4f}",
        f"Precision: {best_metrics['precision']:.4f}",
        f"Recall: {best_metrics['recall']:.4f}",
        f"F1: {best_metrics['f1']:.4f}",
        f"ROC-AUC: {best_metrics['roc_auc']:.4f}",
    ]
    (REPORT_DIR / "training_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def train(data_path: Path, target_recall: float) -> None:
    ensure_project_directories()
    df = load_or_generate_dataset(data_path=data_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_candidates = get_model_candidates(random_state=RANDOM_STATE)
    results: Dict[str, Dict[str, Any]] = {}

    print("Training models...")
    for model_name, estimator in model_candidates.items():
        pipeline = build_model_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        threshold = choose_threshold_for_recall(
            y_true=y_test.to_numpy(),
            y_prob=y_prob,
            target_recall=target_recall,
        )
        metrics = compute_metrics(y_true=y_test.to_numpy(), y_prob=y_prob, threshold=threshold)

        results[model_name] = {
            "pipeline": pipeline,
            "threshold": threshold,
            "metrics": metrics,
            "y_prob": y_prob,
        }

        print(
            f"{model_name}: recall={metrics['recall']:.3f}, "
            f"precision={metrics['precision']:.3f}, roc_auc={metrics['roc_auc']:.3f}, "
            f"threshold={threshold:.3f}"
        )

    best_model_name = max(
        results,
        key=lambda name: (
            results[name]["metrics"]["recall"],
            results[name]["metrics"]["roc_auc"],
            results[name]["metrics"]["f1"],
            results[name]["metrics"]["precision"],
        ),
    )

    best_result = results[best_model_name]
    best_pipeline = best_result["pipeline"]
    best_threshold = float(best_result["threshold"])
    best_metrics = best_result["metrics"]

    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "threshold": best_threshold,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": best_metrics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    BEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, BEST_MODEL_FILE)

    metrics_df = save_metrics(results=results)
    plot_confusion_matrix(
        y_true=y_test.to_numpy(),
        y_prob=best_result["y_prob"],
        threshold=best_threshold,
        output_path=CONFUSION_MATRIX_PLOT,
    )
    plot_roc_curves(
        y_true=y_test.to_numpy(),
        results=results,
        output_path=ROC_CURVE_PLOT,
    )
    save_training_summary(
        best_model_name=best_model_name,
        best_metrics=best_metrics,
        data_path=data_path,
    )

    print("\nTraining complete.")
    print(f"Best model: {best_model_name}")
    print(f"Saved artifact: {BEST_MODEL_FILE}")
    print(f"Saved metrics table: {METRICS_CSV_FILE}")
    print(f"Saved metrics JSON: {METRICS_JSON_FILE}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PLOT}")
    print(f"Saved ROC plot: {ROC_CURVE_PLOT}")
    print("\nModel ranking:")
    print(metrics_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train chronic disease risk prediction models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_FILE,
        help="Path to CSV dataset.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
        help="Minimum desired recall when choosing probability threshold.",
    )
    args = parser.parse_args()

    train(data_path=args.data, target_recall=args.target_recall)


if __name__ == "__main__":
    main()
