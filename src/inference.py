from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier


try:
    from src.feature_engineering import build_model_matrix
except ModuleNotFoundError:
    from feature_engineering import build_model_matrix


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


def load_schema(
    schema_path: Path
) -> dict:
    """
    Load the saved model feature schema.
    """

    with open(
        schema_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_model(
    model_path: Path
) -> CatBoostClassifier:
    """
    Load the trained CatBoost model artifact.
    """

    model = CatBoostClassifier()

    model.load_model(
        str(model_path)
    )

    return model


def predict_risk_scores(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    installments_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    model: CatBoostClassifier | None = None,
    schema: dict | None = None,
) -> pd.DataFrame:
    """
    Generate CreditLens risk scores.

    A preloaded model and schema can optionally be supplied.
    This allows API requests to reuse artifacts already held
    in memory instead of reloading them from disk.
    """

    if schema is None:
        schema = load_schema(
            schema_path
        )

    if model is None:
        model = load_model(
            model_path
        )

    customer_ids, X = build_model_matrix(
        application_df=application_df,
        bureau_df=bureau_df,
        previous_df=previous_df,
        installments_df=installments_df,
        schema=schema
    )

    risk_scores = model.predict_proba(
        X
    )[:, 1]

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
        type=Path,
        help="Path to application-level CSV data."
    )

    parser.add_argument(
        "--bureau",
        required=True,
        type=Path,
        help=(
            "Path to customer-level bureau "
            "feature CSV."
        )
    )

    parser.add_argument(
        "--previous",
        required=True,
        type=Path,
        help=(
            "Path to customer-level previous "
            "application feature CSV."
        )
    )

    parser.add_argument(
        "--installments",
        required=True,
        type=Path,
        help=(
            "Path to customer-level installment "
            "feature CSV."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained CatBoost model."
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to model feature schema JSON."
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for generated risk-score CSV."
    )

    args = parser.parse_args()

    application_df = pd.read_csv(
        args.application,
        low_memory=False
    )

    bureau_df = pd.read_csv(
        args.bureau
    )

    previous_df = pd.read_csv(
        args.previous
    )

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