import unittest

import numpy as np
import pandas as pd

from src.feature_engineering import (
    merge_customer_features,
    add_engineered_features,
    build_model_matrix,
)


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.application_df = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "TARGET": [0, 1],
                "CODE_GENDER": ["F", "M"],
                "DAYS_EMPLOYED": [365243, -1000],
                "CAT_FEATURE": ["A", None],
                "NUM_FEATURE": [10.0, 20.0],
            }
        )

        self.bureau_df = pd.DataFrame(
            {
                "SK_ID_CURR": [1],
                "BUREAU_LOAN_COUNT": [2.0],
            }
        )

        self.previous_df = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "PREV_APPLICATION_COUNT": [3.0, 1.0],
            }
        )

        self.installments_df = pd.DataFrame(
            {
                "SK_ID_CURR": [2],
                "INST_PAYMENT_RECORD_COUNT": [4.0],
            }
        )

        self.expected_features = [
            "DAYS_EMPLOYED",
            "CAT_FEATURE",
            "NUM_FEATURE",
            "BUREAU_LOAN_COUNT",
            "PREV_APPLICATION_COUNT",
            "INST_PAYMENT_RECORD_COUNT",
            "BUREAU_HAS_HISTORY",
            "PREV_HAS_HISTORY",
            "INST_HAS_HISTORY",
            "DAYS_EMPLOYED_ANOMALY",
        ]

        self.schema = {
            "features": self.expected_features,
            "categorical_features": [
                "CAT_FEATURE"
            ],
            "numeric_features": [
                feature
                for feature in self.expected_features
                if feature != "CAT_FEATURE"
            ],
            "excluded_columns": [
                "TARGET",
                "SK_ID_CURR",
                "CODE_GENDER",
            ],
            "categorical_missing_value": "__MISSING__",
            "days_employed_sentinel": 365243,
        }

    def test_merge_customer_features_preserves_application_rows(self):
        result = merge_customer_features(
            application_df=self.application_df,
            bureau_df=self.bureau_df,
            previous_df=self.previous_df,
            installments_df=self.installments_df,
        )

        self.assertEqual(
            len(result),
            len(self.application_df)
        )

        self.assertEqual(
            result["SK_ID_CURR"].tolist(),
            [1, 2]
        )

    def test_engineered_history_flags_are_correct(self):
        merged = merge_customer_features(
            application_df=self.application_df,
            bureau_df=self.bureau_df,
            previous_df=self.previous_df,
            installments_df=self.installments_df,
        )

        result = add_engineered_features(
            merged
        )

        customer_1 = result.loc[
            result["SK_ID_CURR"] == 1
        ].iloc[0]

        customer_2 = result.loc[
            result["SK_ID_CURR"] == 2
        ].iloc[0]

        self.assertEqual(
            customer_1["BUREAU_HAS_HISTORY"],
            1
        )

        self.assertEqual(
            customer_2["BUREAU_HAS_HISTORY"],
            0
        )

        self.assertEqual(
            customer_1["PREV_HAS_HISTORY"],
            1
        )

        self.assertEqual(
            customer_2["PREV_HAS_HISTORY"],
            1
        )

        self.assertEqual(
            customer_1["INST_HAS_HISTORY"],
            0
        )

        self.assertEqual(
            customer_2["INST_HAS_HISTORY"],
            1
        )

    def test_days_employed_sentinel_is_replaced(self):
        merged = merge_customer_features(
            application_df=self.application_df,
            bureau_df=self.bureau_df,
            previous_df=self.previous_df,
            installments_df=self.installments_df,
        )

        result = add_engineered_features(
            merged
        )

        customer_1 = result.loc[
            result["SK_ID_CURR"] == 1
        ].iloc[0]

        customer_2 = result.loc[
            result["SK_ID_CURR"] == 2
        ].iloc[0]

        self.assertTrue(
            np.isnan(
                customer_1["DAYS_EMPLOYED"]
            )
        )

        self.assertEqual(
            customer_1["DAYS_EMPLOYED_ANOMALY"],
            1
        )

        self.assertEqual(
            customer_2["DAYS_EMPLOYED_ANOMALY"],
            0
        )

        self.assertEqual(
            customer_2["DAYS_EMPLOYED"],
            -1000
        )

    def test_build_model_matrix_matches_schema(self):
        customer_ids, X = build_model_matrix(
            application_df=self.application_df,
            bureau_df=self.bureau_df,
            previous_df=self.previous_df,
            installments_df=self.installments_df,
            schema=self.schema,
        )

        self.assertEqual(
            customer_ids.tolist(),
            [1, 2]
        )

        self.assertEqual(
            X.columns.tolist(),
            self.expected_features
        )

        self.assertNotIn(
            "TARGET",
            X.columns
        )

        self.assertNotIn(
            "SK_ID_CURR",
            X.columns
        )

        self.assertNotIn(
            "CODE_GENDER",
            X.columns
        )

    def test_categorical_missing_value_is_encoded(self):
        _, X = build_model_matrix(
            application_df=self.application_df,
            bureau_df=self.bureau_df,
            previous_df=self.previous_df,
            installments_df=self.installments_df,
            schema=self.schema,
        )

        self.assertEqual(
            X.loc[1, "CAT_FEATURE"],
            "__MISSING__"
        )

        self.assertEqual(
            X["CAT_FEATURE"].isna().sum(),
            0
        )

    def test_missing_required_engineering_column_raises_error(self):
        invalid_data = pd.DataFrame(
            {
                "SK_ID_CURR": [1],
                "DAYS_EMPLOYED": [-100],
            }
        )

        with self.assertRaises(ValueError):
            add_engineered_features(
                invalid_data
            )

    def test_duplicate_customer_feature_rows_raise_error(self):
        duplicate_bureau = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 1],
                "BUREAU_LOAN_COUNT": [2, 3],
            }
        )

        with self.assertRaises(pd.errors.MergeError):
            merge_customer_features(
                application_df=self.application_df,
                bureau_df=duplicate_bureau,
                previous_df=self.previous_df,
                installments_df=self.installments_df,
            )
    def test_missing_model_feature_raises_error(self):
        invalid_schema = self.schema.copy()

        invalid_schema["features"] = (
            self.expected_features
            + ["NON_EXISTENT_FEATURE"]
        )

        with self.assertRaisesRegex(
            ValueError,
            "Missing model features"
        ):
            build_model_matrix(
                application_df=self.application_df,
                bureau_df=self.bureau_df,
                previous_df=self.previous_df,
                installments_df=self.installments_df,
                schema=invalid_schema,
            )

    def test_unexpected_model_feature_raises_error(self):
        application_with_extra = (
            self.application_df.copy()
        )

        application_with_extra[
            "UNEXPECTED_FEATURE"
        ] = [100, 200]

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected model features"
        ):
            build_model_matrix(
                application_df=application_with_extra,
                bureau_df=self.bureau_df,
                previous_df=self.previous_df,
                installments_df=self.installments_df,
                schema=self.schema,
            )

    def test_missing_customer_id_raises_error(self):
        bureau_without_id = (
            self.bureau_df.drop(
                columns=["SK_ID_CURR"]
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "bureau dataframe is missing SK_ID_CURR"
        ):
            merge_customer_features(
                application_df=self.application_df,
                bureau_df=bureau_without_id,
                previous_df=self.previous_df,
                installments_df=self.installments_df,
            )

if __name__ == "__main__":
    unittest.main()

    