import numpy as np
import pandas as pd


def merge_customer_features(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    installments_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge application-level data with customer-level relational features.
    """

    required_frames = {
        "application": application_df,
        "bureau": bureau_df,
        "previous": previous_df,
        "installments": installments_df,
    }

    for name, frame in required_frames.items():
        if "SK_ID_CURR" not in frame.columns:
            raise ValueError(
                f"{name} dataframe is missing SK_ID_CURR."
            )

    data = (
        application_df
        .merge(
            bureau_df,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one"
        )
        .merge(
            previous_df,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one"
        )
        .merge(
            installments_df,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one"
        )
    )

    return data


def add_engineered_features(
    data: pd.DataFrame,
    days_employed_sentinel: int = 365243
) -> pd.DataFrame:
    """
    Add engineered features shared by training and inference.
    """

    required_columns = [
        "BUREAU_LOAN_COUNT",
        "PREV_APPLICATION_COUNT",
        "INST_PAYMENT_RECORD_COUNT",
        "DAYS_EMPLOYED",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot build engineered features. "
            f"Missing columns: {missing_columns}"
        )

    engineered_flags = pd.DataFrame(
        {
            "BUREAU_HAS_HISTORY": (
                data["BUREAU_LOAN_COUNT"]
                .notna()
                .astype("int8")
            ),
            "PREV_HAS_HISTORY": (
                data["PREV_APPLICATION_COUNT"]
                .notna()
                .astype("int8")
            ),
            "INST_HAS_HISTORY": (
                data["INST_PAYMENT_RECORD_COUNT"]
                .notna()
                .astype("int8")
            ),
            "DAYS_EMPLOYED_ANOMALY": (
                data["DAYS_EMPLOYED"]
                == days_employed_sentinel
            ).astype("int8"),
        },
        index=data.index,
    )

    result = pd.concat(
        [
            data.copy(),
            engineered_flags
        ],
        axis=1
    )

    result.loc[
        result["DAYS_EMPLOYED"]
        == days_employed_sentinel,
        "DAYS_EMPLOYED"
    ] = np.nan

    return result


def build_model_matrix(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    installments_df: pd.DataFrame,
    schema: dict
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build the exact model input matrix required by the saved schema.
    """

    data = merge_customer_features(
        application_df=application_df,
        bureau_df=bureau_df,
        previous_df=previous_df,
        installments_df=installments_df,
    )

    data = add_engineered_features(
        data=data,
        days_employed_sentinel=schema[
            "days_employed_sentinel"
        ],
    )

    if "SK_ID_CURR" not in data.columns:
        raise ValueError(
            "SK_ID_CURR is required for inference."
        )

    customer_ids = data["SK_ID_CURR"].copy()

    excluded_columns = [
        column
        for column in schema["excluded_columns"]
        if column in data.columns
    ]

    X = data.drop(
        columns=excluded_columns
    ).copy()

    expected_features = schema["features"]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X.columns
    ]

    extra_features = [
        feature
        for feature in X.columns
        if feature not in expected_features
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            f"{missing_features}"
        )

    if extra_features:
        raise ValueError(
            "Unexpected model features: "
            f"{extra_features}"
        )

    # Enforce exact training feature order.
    X = X[
        expected_features
    ].copy()

    missing_token = schema[
        "categorical_missing_value"
    ]

    for column in schema["categorical_features"]:

        if column not in X.columns:
            raise ValueError(
                "Categorical feature missing from "
                f"model matrix: {column}"
            )

        X[column] = (
            X[column]
            .astype("object")
            .where(
                X[column].notna(),
                missing_token
            )
            .astype(str)
        )

    return customer_ids, X