import json
from functools import lru_cache
from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCHEMA_PATH,
    load_model,
    load_schema,
    predict_risk_scores,
)


METADATA_PATH = (
    DEFAULT_MODEL_PATH.parent
    / "creditlens_model_metadata.json"
)


app = FastAPI(
    title="CreditLens AI API",
    description=(
        "Credit risk research and "
        "decision-support API."
    ),
    version="0.3.0",
)


class PredictionRequest(BaseModel):
    application: dict[str, Any] = Field(
        ...,
        description=(
            "Application-level customer features."
        )
    )

    bureau: dict[str, Any] = Field(
        ...,
        description=(
            "Aggregated bureau customer features."
        )
    )

    previous: dict[str, Any] = Field(
        ...,
        description=(
            "Aggregated previous-application features."
        )
    )

    installments: dict[str, Any] = Field(
        ...,
        description=(
            "Aggregated installment-payment features."
        )
    )


class PredictionResponse(BaseModel):
    SK_ID_CURR: int
    risk_score: float
    interpretation: str


@lru_cache(maxsize=1)
def get_runtime_model():
    """
    Load the CatBoost model once and reuse it
    for subsequent API requests.
    """

    return load_model(
        DEFAULT_MODEL_PATH
    )


@lru_cache(maxsize=1)
def get_runtime_schema():
    """
    Load the feature schema once and reuse it
    for subsequent API requests.
    """

    return load_schema(
        DEFAULT_SCHEMA_PATH
    )


@lru_cache(maxsize=1)
def get_runtime_metadata():
    """
    Load model metadata once.
    """

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def validate_customer_ids(
    payload: PredictionRequest
) -> int:
    application_id = payload.application.get(
        "SK_ID_CURR"
    )

    if application_id is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "application.SK_ID_CURR "
                "is required."
            ),
        )

    customer_id = int(
        application_id
    )

    relational_sections = {
        "bureau": payload.bureau,
        "previous": payload.previous,
        "installments": payload.installments,
    }

    for section_name, section in (
        relational_sections.items()
    ):
        section_id = section.get(
            "SK_ID_CURR"
        )

        if section_id is None:
            section["SK_ID_CURR"] = (
                customer_id
            )
            continue

        if int(section_id) != customer_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{section_name}.SK_ID_CURR "
                    "does not match "
                    "application.SK_ID_CURR."
                ),
            )

    return customer_id


@app.get("/")
def root():
    return {
        "name": "CreditLens AI API",
        "version": "0.3.0",
        "status": "running",
    }


@app.get("/health")
def health():
    model_exists = (
        DEFAULT_MODEL_PATH.exists()
    )

    schema_exists = (
        DEFAULT_SCHEMA_PATH.exists()
    )

    metadata_exists = (
        METADATA_PATH.exists()
    )

    if not (
        model_exists
        and schema_exists
        and metadata_exists
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "model_exists": model_exists,
                "schema_exists": schema_exists,
                "metadata_exists": metadata_exists,
            },
        )

    return {
        "status": "healthy",
        "model_exists": True,
        "schema_exists": True,
        "metadata_exists": True,
    }


@app.get("/model-info")
def model_info():
    try:
        schema = get_runtime_schema()
        metadata = get_runtime_metadata()

        return {
            "model_name": metadata.get(
                "model_name"
            ),
            "model_type": metadata.get(
                "model_type"
            ),
            "training_rows": metadata.get(
                "training_rows"
            ),
            "feature_count": schema.get(
                "feature_count"
            ),
            "categorical_feature_count": (
                metadata.get(
                    "categorical_feature_count"
                )
            ),
            "numeric_feature_count": (
                metadata.get(
                    "numeric_feature_count"
                )
            ),
            "excluded_sensitive_feature": (
                metadata.get(
                    "excluded_sensitive_feature"
                )
            ),
            "iterations": metadata.get(
                "iterations"
            ),
            "development_validation": (
                metadata.get(
                    "development_validation"
                )
            ),
            "stability_cv_oof": metadata.get(
                "stability_cv_oof"
            ),
            "probability_note": metadata.get(
                "probability_note"
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load model "
                "information: "
                f"{exc}"
            ),
        ) from exc


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    payload: PredictionRequest
):
    customer_id = validate_customer_ids(
        payload
    )

    try:
        application_df = pd.DataFrame(
            [payload.application]
        )

        bureau_df = pd.DataFrame(
            [payload.bureau]
        )

        previous_df = pd.DataFrame(
            [payload.previous]
        )

        installments_df = pd.DataFrame(
            [payload.installments]
        )

        model = get_runtime_model()
        schema = get_runtime_schema()

        prediction = predict_risk_scores(
            application_df=application_df,
            bureau_df=bureau_df,
            previous_df=previous_df,
            installments_df=installments_df,
            model=model,
            schema=schema,
        )

        risk_score = float(
            prediction.iloc[0][
                "risk_score"
            ]
        )

        return PredictionResponse(
            SK_ID_CURR=customer_id,
            risk_score=risk_score,
            interpretation=(
                "Risk score from the "
                "class-balanced CatBoost model. "
                "This value is not a calibrated "
                "probability of default."
            ),
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{exc}"
            ),
        ) from exc