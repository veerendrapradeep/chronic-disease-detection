from src.config import BEST_MODEL_FILE, DATA_FILE, DEFAULT_TARGET_RECALL, SAMPLE_PATIENT_FILE
from src.generate_sample_data import generate_and_save_dataset
from src.train import train


def ensure_ready_artifacts() -> None:
    if not DATA_FILE.exists() or not SAMPLE_PATIENT_FILE.exists():
        generate_and_save_dataset(csv_path=DATA_FILE, sample_patient_path=SAMPLE_PATIENT_FILE)

    if not BEST_MODEL_FILE.exists():
        train(data_path=DATA_FILE, target_recall=DEFAULT_TARGET_RECALL)
