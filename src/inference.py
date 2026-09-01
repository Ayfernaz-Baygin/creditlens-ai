from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "creditlens_catboost_gender_free.cbm"
)

DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "models"
    / "creditlens_feature_schema.json"
)


def load_schema(schema_path: Path) -> dict:
    with open(
        schema_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_model(model_path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def build_model_matrix(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    installments_df: pd.DataFrame,
    schema: dict
) -> tuple[pd.Series, pd.DataFrame]:

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
                == schema["days_employed_sentinel"]
            ).astype("int8")
        },
        index=data.index
    )

    data = pd.concat(
        [data, engineered_flags],
        axis=1
    )

    data.loc[
        data["DAYS_EMPLOYED"]
        == schema["days_employed_sentinel"],
        "DAYS_EMPLOYED"
    ] = np.nan

    customer_ids = data["SK_ID_CURR"].copy()

    columns_to_drop = [
        column
        for column in schema["excluded_columns"]
        if column in data.columns
    ]

    X = data.drop(
        columns=columns_to_drop
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

    X = X[
        expected_features
    ].copy()

    missing_token = schema[
        "categorical_missing_value"
    ]

    for column in schema["categorical_features"]:
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


def predict_risk_scores(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    installments_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH
) -> pd.DataFrame:

    schema = load_schema(schema_path)
    model = load_model(model_path)

    customer_ids, X = build_model_matrix(
        application_df=application_df,
        bureau_df=bureau_df,
        previous_df=previous_df,
        installments_df=installments_df,
        schema=schema
    )

    risk_scores = model.predict_proba(X)[:, 1]

    if np.isnan(risk_scores).any():
        raise ValueError(
            "Model produced NaN risk scores."
        )

    return pd.DataFrame(
        {
            "SK_ID_CURR": customer_ids.values,
            "risk_score": risk_scores
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CreditLens risk scores."
        )
    )

    parser.add_argument(
        "--application",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--bureau",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--previous",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--installments",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path
    )

    args = parser.parse_args()

    application_df = pd.read_csv(
        args.application,
        low_memory=False
    )

    bureau_df = pd.read_csv(args.bureau)
    previous_df = pd.read_csv(args.previous)
    installments_df = pd.read_csv(
        args.installments
    )

    predictions = predict_risk_scores(
        application_df=application_df,
        bureau_df=bureau_df,
        previous_df=previous_df,
        installments_df=installments_df,
        model_path=args.model,
        schema_path=args.schema
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions.to_csv(
        args.output,
        index=False
    )

    print(
        f"Rows scored: "
        f"{len(predictions):,}"
    )

    print(
        f"Risk score min: "
        f"{predictions['risk_score'].min():.6f}"
    )

    print(
        f"Risk score mean: "
        f"{predictions['risk_score'].mean():.6f}"
    )

    print(
        f"Risk score max: "
        f"{predictions['risk_score'].max():.6f}"
    )

    print(
        f"Saved predictions: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()