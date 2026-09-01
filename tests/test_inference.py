import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.inference import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCHEMA_PATH,
    load_model,
    load_schema,
    predict_risk_scores,
)


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(
            probabilities,
            dtype=float
        )

    def predict_proba(self, X):
        return np.column_stack(
            [
                1.0 - self.probabilities,
                self.probabilities
            ]
        )


class TestInference(unittest.TestCase):

    def test_default_artifacts_exist(self):
        self.assertTrue(
            DEFAULT_MODEL_PATH.exists(),
            f"Model not found: {DEFAULT_MODEL_PATH}"
        )

        self.assertTrue(
            DEFAULT_SCHEMA_PATH.exists(),
            f"Schema not found: {DEFAULT_SCHEMA_PATH}"
        )

    def test_schema_contract_is_valid(self):
        schema = load_schema(
            DEFAULT_SCHEMA_PATH
        )

        self.assertEqual(
            schema["feature_count"],
            194
        )

        self.assertEqual(
            len(schema["features"]),
            194
        )

        self.assertEqual(
            len(set(schema["features"])),
            194
        )

        self.assertEqual(
            len(schema["categorical_features"]),
            15
        )

        self.assertNotIn(
            "CODE_GENDER",
            schema["features"]
        )

        self.assertNotIn(
            "TARGET",
            schema["features"]
        )

        self.assertNotIn(
            "SK_ID_CURR",
            schema["features"]
        )

    def test_saved_model_loads(self):
        model = load_model(
            DEFAULT_MODEL_PATH
        )

        self.assertEqual(
            model.tree_count_,
            1470
        )

    @patch("src.inference.build_model_matrix")
    @patch("src.inference.load_model")
    @patch("src.inference.load_schema")
    def test_predict_risk_scores_output_contract(
        self,
        mock_load_schema,
        mock_load_model,
        mock_build_model_matrix
    ):
        mock_load_schema.return_value = {
            "features": ["A"]
        }

        mock_load_model.return_value = FakeModel(
            [0.20, 0.80]
        )

        mock_build_model_matrix.return_value = (
            pd.Series(
                [100001, 100002]
            ),
            pd.DataFrame(
                {
                    "A": [1.0, 2.0]
                }
            )
        )

        dummy_df = pd.DataFrame(
            {
                "SK_ID_CURR": [100001, 100002]
            }
        )

        result = predict_risk_scores(
            application_df=dummy_df,
            bureau_df=dummy_df,
            previous_df=dummy_df,
            installments_df=dummy_df
        )

        self.assertEqual(
            result.columns.tolist(),
            [
                "SK_ID_CURR",
                "risk_score"
            ]
        )

        self.assertEqual(
            result["SK_ID_CURR"].tolist(),
            [100001, 100002]
        )

        np.testing.assert_allclose(
            result["risk_score"].values,
            [0.20, 0.80]
        )

    @patch("src.inference.build_model_matrix")
    @patch("src.inference.load_model")
    @patch("src.inference.load_schema")
    def test_nan_predictions_raise_error(
        self,
        mock_load_schema,
        mock_load_model,
        mock_build_model_matrix
    ):
        mock_load_schema.return_value = {
            "features": ["A"]
        }

        mock_load_model.return_value = FakeModel(
            [0.25, np.nan]
        )

        mock_build_model_matrix.return_value = (
            pd.Series(
                [100001, 100002]
            ),
            pd.DataFrame(
                {
                    "A": [1.0, 2.0]
                }
            )
        )

        dummy_df = pd.DataFrame(
            {
                "SK_ID_CURR": [100001, 100002]
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "NaN risk scores"
        ):
            predict_risk_scores(
                application_df=dummy_df,
                bureau_df=dummy_df,
                previous_df=dummy_df,
                installments_df=dummy_df
            )


if __name__ == "__main__":
    unittest.main()