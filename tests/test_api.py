import unittest
from unittest.mock import patch

import pandas as pd

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def build_mock_demo_data(
    days_employed=-637,
    education_type="Secondary / secondary special",
):
    application = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "AMT_INCOME_TOTAL": [202500.0],
            "AMT_CREDIT": [406597.5],
            "AMT_ANNUITY": [24700.5],
            "AMT_GOODS_PRICE": [351000.0],
            "DAYS_BIRTH": [-9461],
            "DAYS_EMPLOYED": [days_employed],
            "NAME_EDUCATION_TYPE": [education_type],
            "NAME_INCOME_TYPE": ["Working"],
            "NAME_FAMILY_STATUS": ["Single / not married"],
        }
    )

    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "BUREAU_LOAN_COUNT": [7],
        }
    )

    previous = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "PREV_APPLICATION_COUNT": [3],
        }
    )

    installments = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "INST_PAYMENT_RECORD_COUNT": [15],
        }
    )

    common_ids = [100001]

    return {
        "application": application,
        "bureau": bureau,
        "previous": previous,
        "installments": installments,
        "common_ids": common_ids,
        "common_id_set": set(common_ids),
    }


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

    @patch(
        "src.api.get_demo_data"
    )
    def test_demo_customer_summary_success(
        self,
        mock_get_demo_data
    ):
        mock_get_demo_data.return_value = (
            build_mock_demo_data()
        )

        response = client.get(
            "/demo/customers/100001"
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
            data["income_total"],
            202500.0
        )

        self.assertAlmostEqual(
            data["credit_amount"],
            406597.5
        )

        self.assertAlmostEqual(
            data["annuity_amount"],
            24700.5
        )

        self.assertAlmostEqual(
            data["goods_price"],
            351000.0
        )

        self.assertAlmostEqual(
            data["age_years"],
            25.9,
            places=1
        )

        self.assertAlmostEqual(
            data["employment_years"],
            1.7,
            places=1
        )

        self.assertEqual(
            data["education_type"],
            "Secondary / secondary special"
        )

        self.assertEqual(
            data["income_type"],
            "Working"
        )

        self.assertEqual(
            data["family_status"],
            "Single / not married"
        )

        self.assertTrue(
            data["has_bureau_history"]
        )

        self.assertTrue(
            data["has_previous_application_history"]
        )

        self.assertTrue(
            data["has_installment_history"]
        )

        self.assertNotIn(
            "CODE_GENDER",
            data
        )

    @patch(
        "src.api.get_demo_data"
    )
    def test_demo_customer_summary_not_found(
        self,
        mock_get_demo_data
    ):
        mock_get_demo_data.return_value = (
            build_mock_demo_data()
        )

        response = client.get(
            "/demo/customers/999999"
        )

        self.assertEqual(
            response.status_code,
            404
        )

    @patch(
        "src.api.get_demo_data"
    )
    def test_demo_customer_summary_handles_employment_sentinel(
        self,
        mock_get_demo_data
    ):
        mock_get_demo_data.return_value = (
            build_mock_demo_data(
                days_employed=365243
            )
        )

        response = client.get(
            "/demo/customers/100001"
        )

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.json()

        self.assertIsNone(
            data["employment_years"]
        )


if __name__ == "__main__":
    unittest.main()