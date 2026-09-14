import unittest
from unittest.mock import MagicMock, patch

import numpy as np
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


def build_mock_explanation_matrix():
    """
    A small, fixed model-ready feature matrix and a matching
    fake CatBoost ShapValues output used to test the
    explanation endpoint's sorting/direction logic without
    running real SHAP computation.
    """

    feature_names = [
        "EXT_SOURCE_2",
        "AMT_CREDIT",
        "NAME_EDUCATION_TYPE",
        "INST_LATE_PAYMENT_RATE",
    ]

    X = pd.DataFrame(
        [
            {
                "EXT_SOURCE_2": 0.79,
                "AMT_CREDIT": 568800.0,
                "NAME_EDUCATION_TYPE": "Higher education",
                "INST_LATE_PAYMENT_RATE": 0.18,
            }
        ]
    )

    # Last column is CatBoost's expected/base value.
    shap_matrix = np.array(
        [[-0.42, 0.06, -0.05, 0.21, -0.123]]
    )

    return feature_names, X, shap_matrix


def configure_demo_explain_mocks(
    mock_get_demo_data,
    mock_get_runtime_schema,
    mock_get_runtime_model,
    mock_predict_risk_scores,
    mock_build_model_matrix,
    common_ids=(100001,),
):
    _, X, shap_matrix = (
        build_mock_explanation_matrix()
    )

    placeholder_frame = pd.DataFrame(
        {"SK_ID_CURR": list(common_ids)}
    )

    mock_get_demo_data.return_value = {
        "application": placeholder_frame,
        "bureau": placeholder_frame,
        "previous": placeholder_frame,
        "installments": placeholder_frame,
        "common_id_set": set(common_ids),
    }

    mock_get_runtime_schema.return_value = {
        "categorical_features": [
            "NAME_EDUCATION_TYPE"
        ]
    }

    mock_model = MagicMock()

    mock_model.get_feature_importance.return_value = (
        shap_matrix
    )

    mock_get_runtime_model.return_value = mock_model

    mock_predict_risk_scores.return_value = (
        pd.DataFrame(
            {
                "SK_ID_CURR": [100001],
                "risk_score": [0.346927880706],
            }
        )
    )

    mock_build_model_matrix.return_value = (
        pd.Series([100001]),
        X,
    )


def build_mock_fairness_report():
    """
    A small, fixed fairness-report dataset matching the
    real CSV column names, used to test the fairness
    summary endpoint's contract without reading the
    actual reports/ CSV files.
    """

    major_groups = pd.DataFrame(
        {
            "group": ["F", "M"],
            "n": [40561, 20940],
            "positive_rate_actual": [
                0.0699193806858805,
                0.1016714422158548,
            ],
            "selection_rate": [
                0.238973398091763,
                0.3219675262655205,
            ],
            "precision": [
                0.1813679975239863,
                0.2174428952832987,
            ],
            "recall_tpr": [
                0.6198871650211566,
                0.688586190699859,
            ],
            "fpr": [
                0.210337972166998,
                0.2804741906331401,
            ],
            "fnr": [
                0.3801128349788434,
                0.3114138093001409,
            ],
            "roc_auc": [
                0.7767740430945124,
                0.7814378068816805,
            ],
        }
    )

    major_group_gaps = pd.DataFrame(
        {
            "metric": [
                "positive_rate_actual",
                "selection_rate",
                "precision",
                "recall_tpr",
                "fpr",
                "fnr",
                "roc_auc",
            ],
            "min_group": [
                "F", "F", "F", "F", "F", "M", "F"
            ],
            "min_value": [
                0.0699193806858805,
                0.238973398091763,
                0.1813679975239863,
                0.6198871650211566,
                0.210337972166998,
                0.3114138093001409,
                0.7767740430945124,
            ],
            "max_group": [
                "M", "M", "M", "M", "M", "F", "M"
            ],
            "max_value": [
                0.1016714422158548,
                0.3219675262655205,
                0.2174428952832987,
                0.688586190699859,
                0.2804741906331401,
                0.3801128349788434,
                0.7814378068816805,
            ],
            "absolute_gap": [
                0.0317520615299743,
                0.0829941281737575,
                0.0360748977593123,
                0.0686990256787024,
                0.0701362184661421,
                0.0686990256787025,
                0.0046637637871681,
            ],
        }
    )

    audit = pd.DataFrame(
        {
            "group": ["F", "M", "XNA"],
            "n": [40561, 20940, 2],
        }
    )

    return {
        "major_groups": major_groups,
        "major_group_gaps": major_group_gaps,
        "audit": audit,
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

    @patch(
        "src.api.build_model_matrix"
    )
    @patch(
        "src.api.predict_risk_scores"
    )
    @patch(
        "src.api.get_runtime_model"
    )
    @patch(
        "src.api.get_runtime_schema"
    )
    @patch(
        "src.api.get_demo_data"
    )
    def test_demo_explain_success(
        self,
        mock_get_demo_data,
        mock_get_runtime_schema,
        mock_get_runtime_model,
        mock_predict_risk_scores,
        mock_build_model_matrix,
    ):
        configure_demo_explain_mocks(
            mock_get_demo_data,
            mock_get_runtime_schema,
            mock_get_runtime_model,
            mock_predict_risk_scores,
            mock_build_model_matrix,
        )

        response = client.get(
            "/demo/explain/100001"
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

        self.assertAlmostEqual(
            data["base_value"],
            -0.123,
            places=6
        )

        self.assertEqual(
            len(data["top_features"]),
            4
        )

        self.assertIn(
            "do not represent causal effects",
            data["disclaimer"]
        )

        mock_predict_risk_scores.assert_called_once()
        mock_build_model_matrix.assert_called_once()

    @patch(
        "src.api.build_model_matrix"
    )
    @patch(
        "src.api.predict_risk_scores"
    )
    @patch(
        "src.api.get_runtime_model"
    )
    @patch(
        "src.api.get_runtime_schema"
    )
    @patch(
        "src.api.get_demo_data"
    )
    def test_demo_explain_not_found(
        self,
        mock_get_demo_data,
        mock_get_runtime_schema,
        mock_get_runtime_model,
        mock_predict_risk_scores,
        mock_build_model_matrix,
    ):
        configure_demo_explain_mocks(
            mock_get_demo_data,
            mock_get_runtime_schema,
            mock_get_runtime_model,
            mock_predict_risk_scores,
            mock_build_model_matrix,
        )

        response = client.get(
            "/demo/explain/999999"
        )

        self.assertEqual(
            response.status_code,
            404
        )

        mock_predict_risk_scores.assert_not_called()
        mock_build_model_matrix.assert_not_called()

    @patch(
        "src.api.build_model_matrix"
    )
    @patch(
        "src.api.predict_risk_scores"
    )
    @patch(
        "src.api.get_runtime_model"
    )
    @patch(
        "src.api.get_runtime_schema"
    )
    @patch(
        "src.api.get_demo_data"
    )
    def test_demo_explain_top_features_sorted_by_absolute_shap(
        self,
        mock_get_demo_data,
        mock_get_runtime_schema,
        mock_get_runtime_model,
        mock_predict_risk_scores,
        mock_build_model_matrix,
    ):
        configure_demo_explain_mocks(
            mock_get_demo_data,
            mock_get_runtime_schema,
            mock_get_runtime_model,
            mock_predict_risk_scores,
            mock_build_model_matrix,
        )

        response = client.get(
            "/demo/explain/100001"
        )

        data = response.json()

        feature_order = [
            item["feature"]
            for item in data["top_features"]
        ]

        self.assertEqual(
            feature_order,
            [
                "EXT_SOURCE_2",
                "INST_LATE_PAYMENT_RATE",
                "AMT_CREDIT",
                "NAME_EDUCATION_TYPE",
            ]
        )

    @patch(
        "src.api.build_model_matrix"
    )
    @patch(
        "src.api.predict_risk_scores"
    )
    @patch(
        "src.api.get_runtime_model"
    )
    @patch(
        "src.api.get_runtime_schema"
    )
    @patch(
        "src.api.get_demo_data"
    )
    def test_demo_explain_positive_shap_increases_score(
        self,
        mock_get_demo_data,
        mock_get_runtime_schema,
        mock_get_runtime_model,
        mock_predict_risk_scores,
        mock_build_model_matrix,
    ):
        configure_demo_explain_mocks(
            mock_get_demo_data,
            mock_get_runtime_schema,
            mock_get_runtime_model,
            mock_predict_risk_scores,
            mock_build_model_matrix,
        )

        response = client.get(
            "/demo/explain/100001"
        )

        data = response.json()

        positive_feature = next(
            item
            for item in data["top_features"]
            if item["feature"]
            == "INST_LATE_PAYMENT_RATE"
        )

        self.assertEqual(
            positive_feature["direction"],
            "increases_score"
        )

    @patch(
        "src.api.build_model_matrix"
    )
    @patch(
        "src.api.predict_risk_scores"
    )
    @patch(
        "src.api.get_runtime_model"
    )
    @patch(
        "src.api.get_runtime_schema"
    )
    @patch(
        "src.api.get_demo_data"
    )
    def test_demo_explain_negative_shap_decreases_score(
        self,
        mock_get_demo_data,
        mock_get_runtime_schema,
        mock_get_runtime_model,
        mock_predict_risk_scores,
        mock_build_model_matrix,
    ):
        configure_demo_explain_mocks(
            mock_get_demo_data,
            mock_get_runtime_schema,
            mock_get_runtime_model,
            mock_predict_risk_scores,
            mock_build_model_matrix,
        )

        response = client.get(
            "/demo/explain/100001"
        )

        data = response.json()

        negative_feature = next(
            item
            for item in data["top_features"]
            if item["feature"] == "EXT_SOURCE_2"
        )

        self.assertEqual(
            negative_feature["direction"],
            "decreases_score"
        )

    @patch(
        "src.api.get_fairness_report"
    )
    def test_fairness_summary_success(
        self,
        mock_get_fairness_report
    ):
        mock_get_fairness_report.return_value = (
            build_mock_fairness_report()
        )

        response = client.get(
            "/fairness-summary"
        )

        self.assertEqual(
            response.status_code,
            200
        )

        data = response.json()

        self.assertEqual(
            len(data["groups"]),
            2
        )

        self.assertEqual(
            data["groups"][0]["group"],
            "F"
        )

        self.assertEqual(
            data["groups"][0]["sample_count"],
            40561
        )

        self.assertAlmostEqual(
            data["groups"][0]["recall"],
            0.6198871650211566,
            places=6
        )

        self.assertEqual(
            data["groups"][1]["group"],
            "M"
        )

        self.assertAlmostEqual(
            data["gaps"]["recall"],
            0.0686990256787024,
            places=6
        )

        self.assertAlmostEqual(
            data["gaps"]["false_positive_rate"],
            0.0701362184661421,
            places=6
        )

        self.assertAlmostEqual(
            data["gaps"]["false_negative_rate"],
            0.0686990256787025,
            places=6
        )

        self.assertAlmostEqual(
            data["gaps"]["roc_auc"],
            0.0046637637871681,
            places=6
        )

        self.assertEqual(
            len(data["excluded_groups"]),
            1
        )

        self.assertEqual(
            data["excluded_groups"][0]["group"],
            "XNA"
        )

        self.assertEqual(
            data["excluded_groups"][0][
                "sample_count"
            ],
            2
        )

        self.assertIn(
            "do not establish",
            data["note"]
        )

    @patch(
        "src.api.get_fairness_report"
    )
    def test_fairness_summary_missing_report_returns_503(
        self,
        mock_get_fairness_report
    ):
        mock_get_fairness_report.side_effect = (
            FileNotFoundError(
                "Fairness report files are missing: "
                "reports/catboost_gender_fairness_major_groups.csv"
            )
        )

        response = client.get(
            "/fairness-summary"
        )

        self.assertEqual(
            response.status_code,
            503
        )

        self.assertIn(
            "Fairness report unavailable",
            response.json()["detail"]
        )


if __name__ == "__main__":
    unittest.main()
