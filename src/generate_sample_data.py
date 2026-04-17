import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DATA_FILE,
    DATA_RAW_DIR,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    SAMPLE_PATIENT_FILE,
    TARGET_COLUMN,
)


def _to_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def generate_dataset(n_samples: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(20, 86, size=n_samples)
    bmi = np.clip(rng.normal(28.0, 6.0, size=n_samples), 15.0, 55.0)
    systolic_bp = np.clip(rng.normal(130.0, 20.0, size=n_samples), 80.0, 220.0)
    diastolic_bp = np.clip(rng.normal(82.0, 12.0, size=n_samples), 50.0, 130.0)
    glucose = np.clip(rng.normal(125.0, 40.0, size=n_samples), 60.0, 320.0)
    hba1c = np.clip(rng.normal(6.1, 1.5, size=n_samples), 4.0, 14.0)
    cholesterol = np.clip(rng.normal(205.0, 45.0, size=n_samples), 100.0, 400.0)
    creatinine = np.clip(rng.normal(1.0, 0.4, size=n_samples), 0.4, 6.0)
    egfr = np.clip(rng.normal(85.0, 25.0, size=n_samples), 10.0, 140.0)
    physical_activity_days = rng.integers(0, 8, size=n_samples)
    sleep_hours = np.clip(rng.normal(6.8, 1.3, size=n_samples), 3.0, 10.0)

    gender = rng.choice(["female", "male"], size=n_samples, p=[0.52, 0.48])
    smoking_status = rng.choice(
        ["never", "former", "current"],
        size=n_samples,
        p=[0.55, 0.25, 0.20],
    )
    family_history = rng.choice(["no", "yes"], size=n_samples, p=[0.63, 0.37])
    diet_quality = rng.choice(
        ["good", "average", "poor"],
        size=n_samples,
        p=[0.28, 0.50, 0.22],
    )
    alcohol_intake = rng.choice(
        ["low", "moderate", "high"],
        size=n_samples,
        p=[0.40, 0.45, 0.15],
    )

    linear_score = (
        -4.5
        + 0.045 * (age - 45)
        + 0.08 * (bmi - 26)
        + 0.03 * (systolic_bp - 120)
        + 0.022 * (glucose - 100)
        + 0.7 * (hba1c - 5.7)
        + 0.015 * (cholesterol - 180)
        + 0.9 * (family_history == "yes")
        + 0.85 * (smoking_status == "current")
        + 0.35 * (smoking_status == "former")
        + 0.45 * (diet_quality == "poor")
        + 0.2 * (alcohol_intake == "high")
        - 0.25 * physical_activity_days
        - 0.18 * (diet_quality == "good")
        - 0.15 * (sleep_hours - 7.0)
        - 0.02 * (egfr - 90)
        + rng.normal(0.0, 1.0, size=n_samples)
    )

    probability = 1.0 / (1.0 + np.exp(-linear_score))
    probability = np.clip(probability, 0.01, 0.99)
    target = rng.binomial(1, probability, size=n_samples)

    df = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "glucose": glucose,
            "hba1c": hba1c,
            "cholesterol": cholesterol,
            "creatinine": creatinine,
            "egfr": egfr,
            "physical_activity_days": physical_activity_days,
            "sleep_hours": sleep_hours,
            "gender": gender,
            "smoking_status": smoking_status,
            "family_history": family_history,
            "diet_quality": diet_quality,
            "alcohol_intake": alcohol_intake,
            TARGET_COLUMN: target,
        }
    )

    # Inject light missingness so the preprocessing pipeline is tested.
    for column in df.columns:
        if column == TARGET_COLUMN:
            continue
        missing_rate = 0.03 if column in df.select_dtypes(include=[np.number]).columns else 0.02
        missing_mask = rng.random(n_samples) < missing_rate
        df.loc[missing_mask, column] = np.nan

    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_sample_patient(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidate = df[FEATURE_COLUMNS].dropna().head(1)
    if not candidate.empty:
        payload = {k: _to_native(v) for k, v in candidate.iloc[0].to_dict().items()}
    else:
        payload = {}
        for feature in FEATURE_COLUMNS:
            if pd.api.types.is_numeric_dtype(df[feature]):
                payload[feature] = float(df[feature].median())
            else:
                payload[feature] = str(df[feature].mode(dropna=True).iloc[0])

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_and_save_dataset(
    csv_path: Path = DATA_FILE,
    n_samples: int = 1500,
    random_state: int = RANDOM_STATE,
    sample_patient_path: Path = SAMPLE_PATIENT_FILE,
) -> pd.DataFrame:
    df = generate_dataset(n_samples=n_samples, random_state=random_state)
    save_dataset(df=df, output_path=csv_path)
    save_sample_patient(df=df, output_path=sample_patient_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic chronic disease dataset.")
    parser.add_argument("--samples", type=int, default=1500, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_FILE,
        help="CSV path for generated dataset.",
    )
    parser.add_argument(
        "--sample-patient",
        type=Path,
        default=SAMPLE_PATIENT_FILE,
        help="JSON path for one sample patient record.",
    )
    args = parser.parse_args()

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_and_save_dataset(
        csv_path=args.output,
        n_samples=args.samples,
        random_state=args.seed,
        sample_patient_path=args.sample_patient,
    )

    prevalence = float(df[TARGET_COLUMN].mean())
    print(f"Generated dataset: {args.output}")
    print(f"Rows: {len(df)}")
    print(f"Positive class prevalence: {prevalence:.3f}")


if __name__ == "__main__":
    main()
