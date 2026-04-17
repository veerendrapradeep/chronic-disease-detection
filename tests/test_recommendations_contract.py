import unittest

from src.recommendations import build_personalized_recommendations


class RecommendationsContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "age": 58,
            "bmi": 33.2,
            "systolic_bp": 152,
            "diastolic_bp": 96,
            "glucose": 190,
            "hba1c": 7.8,
            "cholesterol": 238,
            "creatinine": 1.2,
            "egfr": 66,
            "physical_activity_days": 1,
            "sleep_hours": 5.9,
            "gender": "male",
            "smoking_status": "current",
            "family_history": "yes",
            "diet_quality": "poor",
            "alcohol_intake": "high",
        }

    def test_recommendations_has_required_sections(self) -> None:
        result = build_personalized_recommendations(
            payload=self.payload,
            predicted_probability=0.82,
            threshold=0.50,
        )

        expected_keys = {
            "summary",
            "risk_factors_detected",
            "exercise_plan",
            "foods_to_take_more",
            "foods_to_limit_or_avoid",
            "natural_support_options",
            "medication_safety_notes",
            "daily_action_plan",
            "when_to_seek_urgent_care",
            "evidence_sources",
            "safety_disclaimer",
        }
        self.assertTrue(expected_keys.issubset(result.keys()))

    def test_high_risk_summary_and_disclaimer_present(self) -> None:
        result = build_personalized_recommendations(
            payload=self.payload,
            predicted_probability=0.82,
            threshold=0.50,
        )

        self.assertIn("above the active threshold", result["summary"])
        self.assertIn("Educational guidance only", result["safety_disclaimer"])


if __name__ == "__main__":
    unittest.main()
