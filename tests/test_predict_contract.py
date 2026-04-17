import unittest

from src.config import BEST_MODEL_FILE, SAMPLE_PATIENT_FILE
from src.predict import predict
from tests.test_utils import ensure_ready_artifacts


class PredictContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_ready_artifacts()

    def test_predict_output_contract(self) -> None:
        result = predict(input_path=SAMPLE_PATIENT_FILE, model_path=BEST_MODEL_FILE)

        self.assertIn("model_name", result)
        self.assertIn("threshold", result)
        self.assertIn("predicted_probability", result)
        self.assertIn("predicted_label", result)
        self.assertIn("risk_category", result)
        self.assertIn("recommendations", result)

        self.assertGreaterEqual(result["predicted_probability"], 0.0)
        self.assertLessEqual(result["predicted_probability"], 1.0)
        self.assertIn(result["predicted_label"], [0, 1])

        recommendations = result["recommendations"]
        self.assertIn("exercise_plan", recommendations)
        self.assertIn("foods_to_take_more", recommendations)
        self.assertIn("medication_safety_notes", recommendations)


if __name__ == "__main__":
    unittest.main()
