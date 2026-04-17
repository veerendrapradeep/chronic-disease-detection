import unittest

from src.config import BEST_MODEL_FILE, DEMO_PATIENTS_DIR
from src.predict import predict
from tests.test_utils import ensure_ready_artifacts


class DemoPatientCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_ready_artifacts()

    def test_demo_patient_files_exist(self) -> None:
        expected_files = [
            DEMO_PATIENTS_DIR / "low_risk_case.json",
            DEMO_PATIENTS_DIR / "moderate_risk_case.json",
            DEMO_PATIENTS_DIR / "high_risk_case.json",
            DEMO_PATIENTS_DIR / "critical_risk_case.json",
        ]

        for file_path in expected_files:
            self.assertTrue(file_path.exists(), f"Missing demo patient file: {file_path}")

    def test_demo_case_probability_ordering(self) -> None:
        low = predict(DEMO_PATIENTS_DIR / "low_risk_case.json", BEST_MODEL_FILE)
        moderate = predict(DEMO_PATIENTS_DIR / "moderate_risk_case.json", BEST_MODEL_FILE)
        high = predict(DEMO_PATIENTS_DIR / "high_risk_case.json", BEST_MODEL_FILE)
        critical = predict(DEMO_PATIENTS_DIR / "critical_risk_case.json", BEST_MODEL_FILE)

        self.assertLess(low["predicted_probability"], moderate["predicted_probability"])
        self.assertLess(moderate["predicted_probability"], high["predicted_probability"])
        self.assertLessEqual(high["predicted_probability"], critical["predicted_probability"])


if __name__ == "__main__":
    unittest.main()
