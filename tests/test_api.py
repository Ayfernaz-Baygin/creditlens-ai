import unittest
from unittest.mock import patch

import pandas as pd

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


class TestAPI(unittest.TestCase):

    def test_root(self):
        response = client.get("/")

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.json()

        self.assertEqual(
            data["name"],
            "CreditLens AI API"
        )

        self.assertEqual(
            data["version"],
            "0.3.0"
        )

        self.assertEqual(
            data["status"],
            "running"
        )

    def test_health(self):
        response = client.get("/health")

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.json()

        self.assertEqual(
            data["status"],
            "healthy"
        )

        self.assertTrue(
            data["model_exists"]
        )

        self.assertTrue(
            data["schema_exists"]
        )

        self.assertTrue(
            data["metadata_exists"]
        )

    def test_model_info(self):
        response = client.get(
            "/model-info"
        )

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.json()

        self.assertEqual(
            data["model_type"],
            "CatBoostClassifier"
        )

        self.assertEqual(
            data["feature_count"],
            194
        )

        self.assertEqual(
            data["categorical_feature_count"],
            15
        )

        self.assertEqual(
            data["numeric_feature_count"],
            179
        )

        self.assertEqual(
            data["excluded_sensitive_feature"],
            "CODE_GENDER"
        )

        self.assertEqual(
            data["iterations"],
            1470
        )

    @patch(
        "src.api.predict_risk_scores"
    )
    def test_predict_success(
        self,
        mock_predict
    ):
        mock_predict.return_value = (
            pd.DataFrame(
                {
                    "SK_ID_CURR": [
                        100001
                    ],
                    "risk_score": [
                        0.346927880706
                    ]
                }
            )
        )

        payload = {
            "application": {
                "SK_ID_CURR": 100001
            },
            "bureau": {
                "SK_ID_CURR": 100001
            },
            "previous": {
                "SK_ID_CURR": 100001
            },
            "installments": {
                "SK_ID_CURR": 100001
            }
        }

        response = client.post(
            "/predict",
            json=payload
        )

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.json()

        self.assertEqual(
            data["SK_ID_CURR"],
            100001
        )

        self.assertAlmostEqual(
            data["risk_score"],
            0.346927880706,
            places=12
        )

        self.assertIn(
            "not a calibrated",
            data["interpretation"]
        )

        mock_predict.assert_called_once()

    @patch(
        "src.api.predict_risk_scores"
    )
    def test_predict_rejects_customer_id_mismatch(
        self,
        mock_predict
    ):
        payload = {
            "application": {
                "SK_ID_CURR": 100001
            },
            "bureau": {
                "SK_ID_CURR": 999999
            },
            "previous": {
                "SK_ID_CURR": 100001
            },
            "installments": {
                "SK_ID_CURR": 100001
            }
        }

        response = client.post(
            "/predict",
            json=payload
        )

        self.assertEqual(
            response.status_code,
            422
        )

        self.assertIn(
            "does not match",
            response.json()["detail"]
        )

        mock_predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()