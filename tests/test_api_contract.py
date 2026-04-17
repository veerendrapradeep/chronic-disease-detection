import json
import os
import unittest

from fastapi.testclient import TestClient

from app.api import app
from src.config import SAMPLE_PATIENT_FILE
from tests.test_utils import ensure_ready_artifacts


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_ready_artifacts()
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "ok")

    def test_root_endpoint(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("message", body)
        self.assertIn("quick_links", body)
        self.assertIn("notes", body)
        self.assertEqual(body["quick_links"].get("health"), "/health")

    def test_predict_endpoint_contract(self) -> None:
        payload = json.loads(SAMPLE_PATIENT_FILE.read_text(encoding="utf-8"))
        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertIn("predicted_probability", body)
        self.assertIn("risk_category", body)
        self.assertIn("recommendations", body)
        self.assertIn("summary", body["recommendations"])
        self.assertIn("exercise_plan", body["recommendations"])
        self.assertIn("foods_to_limit_or_avoid", body["recommendations"])

    def test_predict_requires_api_key_when_enabled(self) -> None:
        payload = json.loads(SAMPLE_PATIENT_FILE.read_text(encoding="utf-8"))
        previous_key = os.environ.get("API_ACCESS_KEY")
        os.environ["API_ACCESS_KEY"] = "demo-contract-key"

        try:
            missing_header_response = self.client.post("/predict", json=payload)
            self.assertEqual(missing_header_response.status_code, 401)

            invalid_header_response = self.client.post(
                "/predict",
                json=payload,
                headers={"x-api-key": "wrong-key"},
            )
            self.assertEqual(invalid_header_response.status_code, 401)

            valid_header_response = self.client.post(
                "/predict",
                json=payload,
                headers={"x-api-key": "demo-contract-key"},
            )
            self.assertEqual(valid_header_response.status_code, 200)
        finally:
            if previous_key is None:
                os.environ.pop("API_ACCESS_KEY", None)
            else:
                os.environ["API_ACCESS_KEY"] = previous_key


if __name__ == "__main__":
    unittest.main()
