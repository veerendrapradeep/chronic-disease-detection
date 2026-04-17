import unittest

from src.config import BEST_MODEL_FILE, DEMO_PATIENTS_DIR
from src.predict_batch import run_batch_predictions
from tests.test_utils import ensure_ready_artifacts


class PredictBatchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_ready_artifacts()

    def test_batch_predictions_contract(self) -> None:
        results = run_batch_predictions(input_dir=DEMO_PATIENTS_DIR, model_path=BEST_MODEL_FILE)

        self.assertGreaterEqual(len(results), 4)
        for result in results:
            self.assertIn("input_file", result)
            self.assertIn("predicted_probability", result)
            self.assertIn("risk_category", result)
            self.assertIn("recommendations", result)
            self.assertIn("summary", result["recommendations"])


if __name__ == "__main__":
    unittest.main()
