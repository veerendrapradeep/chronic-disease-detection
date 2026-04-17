import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import (
    BEST_MODEL_FILE,
    COEFFICIENT_IMPORTANCE_CSV,
    COEFFICIENT_IMPORTANCE_PLOT,
    DATA_FILE,
    FEATURE_COLUMNS,
    PERMUTATION_IMPORTANCE_CSV,
    PERMUTATION_IMPORTANCE_PLOT,
    TARGET_COLUMN,
)
from .data_pipeline import ensure_project_directories, validate_training_columns


def _save_horizontal_bar_plot(df: pd.DataFrame, value_column: str, title: str, output_path: Path, top_k: int = 12) -> None:
    top_df = df.head(top_k).copy()
    fig_height = max(4, 0.45 * len(top_df))

    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(top_df["feature"], top_df[value_column], color="#2563eb")
    ax.invert_yaxis()
    ax.set_xlabel(value_column.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_permutation_importance(model_artifact: dict, data_path: Path, repeats: int) -> pd.DataFrame:
    pipeline = model_artifact["pipeline"]

    df = pd.read_csv(data_path)
    validate_training_columns(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    result = permutation_importance(
        estimator=pipeline,
        X=X,
        y=y,
        scoring="roc_auc",
        n_repeats=repeats,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)

    return importance_df


def build_logistic_coefficients(model_artifact: dict) -> pd.DataFrame:
    pipeline = model_artifact["pipeline"]
    model = pipeline.named_steps["model"]

    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "absolute_coefficient"])

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_features = preprocessor.get_feature_names_out().tolist()

    coefficients = model.coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": transformed_features,
            "coefficient": coefficients,
        }
    )
    coef_df["absolute_coefficient"] = coef_df["coefficient"].abs()

    coef_df = coef_df.sort_values(by="absolute_coefficient", ascending=False)
    return coef_df


def generate_explainability_reports(model_path: Path, data_path: Path, repeats: int) -> Tuple[List[Path], List[str]]:
    ensure_project_directories()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    artifact = joblib.load(model_path)
    generated_files: List[Path] = []
    notes: List[str] = []

    permutation_df = build_permutation_importance(artifact, data_path=data_path, repeats=repeats)
    permutation_df.to_csv(PERMUTATION_IMPORTANCE_CSV, index=False)
    generated_files.append(PERMUTATION_IMPORTANCE_CSV)

    _save_horizontal_bar_plot(
        df=permutation_df,
        value_column="importance_mean",
        title="Permutation Feature Importance (AUC impact)",
        output_path=PERMUTATION_IMPORTANCE_PLOT,
        top_k=12,
    )
    generated_files.append(PERMUTATION_IMPORTANCE_PLOT)

    coef_df = build_logistic_coefficients(artifact)
    if coef_df.empty:
        notes.append("Coefficient report skipped because the selected model has no linear coefficients.")
    else:
        coef_df.to_csv(COEFFICIENT_IMPORTANCE_CSV, index=False)
        generated_files.append(COEFFICIENT_IMPORTANCE_CSV)

        _save_horizontal_bar_plot(
            df=coef_df,
            value_column="absolute_coefficient",
            title="Top Absolute Logistic Coefficients",
            output_path=COEFFICIENT_IMPORTANCE_PLOT,
            top_k=12,
        )
        generated_files.append(COEFFICIENT_IMPORTANCE_PLOT)

    return generated_files, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate explainability reports for the trained model.")
    parser.add_argument("--model", type=Path, default=BEST_MODEL_FILE, help="Path to saved model artifact.")
    parser.add_argument("--data", type=Path, default=DATA_FILE, help="Path to dataset CSV.")
    parser.add_argument("--repeats", type=int, default=10, help="Permutation importance repeats.")
    args = parser.parse_args()

    files, notes = generate_explainability_reports(
        model_path=args.model,
        data_path=args.data,
        repeats=args.repeats,
    )

    print("Explainability artifacts generated:")
    for output_file in files:
        print(f"- {output_file}")

    for note in notes:
        print(f"Note: {note}")


if __name__ == "__main__":
    main()
