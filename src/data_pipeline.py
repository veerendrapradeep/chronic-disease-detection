from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DEFAULT_TARGET_RECALL,
    FEATURE_COLUMNS,
    MODEL_DIR,
    NUMERIC_FEATURES,
    REPORT_DIR,
    TARGET_COLUMN,
)


def ensure_project_directories() -> None:
    for folder in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def validate_training_columns(df: pd.DataFrame) -> None:
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {joined}")


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_model_candidates(random_state: int) -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=450,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def build_model_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def choose_threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = DEFAULT_TARGET_RECALL,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if thresholds.size == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    valid_indices = np.where(recall >= target_recall)[0]

    if valid_indices.size == 0:
        return 0.5

    best_index = valid_indices[np.argmax(precision[valid_indices])]
    return float(thresholds[best_index])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def build_inference_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    ordered_payload = {feature: payload.get(feature, None) for feature in FEATURE_COLUMNS}
    return pd.DataFrame([ordered_payload])
