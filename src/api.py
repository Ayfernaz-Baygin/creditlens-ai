import json
from functools import lru_cache
from typing import Any

import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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

PROJECT_ROOT = DEFAULT_MODEL_PATH.parent.parent

APPLICATION_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "application_test.csv"
)

BUREAU_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "bureau_customer_features.csv"
)

PREVIOUS_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "previous_application_customer_features.csv"
)

INSTALLMENTS_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "installments_customer_features.csv"
)


app = FastAPI(
    title="CreditLens AI API",
    description=(
        "Credit risk research and "
        "decision-support API."
    ),
    version="0.3.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    application: dict[str, Any] = Field(
        ...,
        description=(
            "Application-level customer features."
        ),
    )

    bureau: dict[str, Any] = Field(
        ...,
        description=(
            "Aggregated bureau customer features."
        ),
    )

    previous: dict[str, Any] = Field(
        ...,
        description=(
            "Aggregated previous-application features."
        ),
    )

    installments: dict[str, Any] = Field(
        ...,
        description=(
            "Aggregated installment-payment features."
        ),
    )


class PredictionResponse(BaseModel):
    SK_ID_CURR: int
    risk_score: float
    interpretation: str


class CustomerSummaryResponse(BaseModel):
    SK_ID_CURR: int
    income_total: float | None
    credit_amount: float | None
    annuity_amount: float | None
    goods_price: float | None
    age_years: float | None
    employment_years: float | None
    education_type: str | None
    income_type: str | None
    family_status: str | None
    has_bureau_history: bool
    has_previous_application_history: bool
    has_installment_history: bool


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
        encoding="utf-8",
    ) as file:
        return json.load(file)


@lru_cache(maxsize=1)
def get_demo_data():
    """
    Load local Home Credit test and aggregated
    feature data once for dashboard demo use.
    """

    required_files = [
        APPLICATION_TEST_PATH,
        BUREAU_FEATURES_PATH,
        PREVIOUS_FEATURES_PATH,
        INSTALLMENTS_FEATURES_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Demo data files are missing: "
            + ", ".join(missing_files)
        )

    application = pd.read_csv(
        APPLICATION_TEST_PATH,
        low_memory=False,
    )

    bureau = pd.read_csv(
        BUREAU_FEATURES_PATH
    )

    previous = pd.read_csv(
        PREVIOUS_FEATURES_PATH
    )

    installments = pd.read_csv(
        INSTALLMENTS_FEATURES_PATH
    )

    common_ids = sorted(
        set(application["SK_ID_CURR"])
        & set(bureau["SK_ID_CURR"])
        & set(previous["SK_ID_CURR"])
        & set(installments["SK_ID_CURR"])
    )

    return {
        "application": application,
        "bureau": bureau,
        "previous": previous,
        "installments": installments,
        "common_ids": common_ids,
        "common_id_set": set(common_ids),
    }


def _safe_float(value: Any) -> float | None:
    """
    Convert a raw feature value to a JSON-safe float,
    returning None for missing values.
    """

    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(numeric_value):
        return None

    return numeric_value


def _safe_str(value: Any) -> str | None:
    """
    Convert a raw feature value to a JSON-safe string,
    returning None for missing values.
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    return str(value)


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


@app.get("/demo/customers")
def demo_customers(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    )
):
    """
    Return customer IDs available in every
    feature source required by the demo model.
    """

    try:
        demo_data = get_demo_data()

        customer_ids = (
            demo_data["common_ids"][:limit]
        )

        return {
            "customers": [
                int(customer_id)
                for customer_id
                in customer_ids
            ],
            "returned": len(
                customer_ids
            ),
            "available": len(
                demo_data["common_ids"]
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load demo customers: "
                f"{exc}"
            ),
        ) from exc


@app.get(
    "/demo/customers/{customer_id}",
    response_model=CustomerSummaryResponse,
)
def demo_customer_summary(
    customer_id: int
):
    """
    Return a human-readable feature summary for a
    single demo customer.
    """

    try:
        demo_data = get_demo_data()

        if (
            customer_id
            not in demo_data[
                "common_id_set"
            ]
        ):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Customer was not found "
                    "in all required demo "
                    "feature sources."
                ),
            )

        application_row = (
            demo_data["application"]
            .loc[
                demo_data[
                    "application"
                ]["SK_ID_CURR"]
                == customer_id
            ]
            .iloc[0]
        )

        bureau_row = (
            demo_data["bureau"]
            .loc[
                demo_data[
                    "bureau"
                ]["SK_ID_CURR"]
                == customer_id
            ]
            .iloc[0]
        )

        previous_row = (
            demo_data["previous"]
            .loc[
                demo_data[
                    "previous"
                ]["SK_ID_CURR"]
                == customer_id
            ]
            .iloc[0]
        )

        installments_row = (
            demo_data["installments"]
            .loc[
                demo_data[
                    "installments"
                ]["SK_ID_CURR"]
                == customer_id
            ]
            .iloc[0]
        )

        schema = get_runtime_schema()

        days_employed_sentinel = schema.get(
            "days_employed_sentinel",
            365243,
        )

        days_birth = application_row.get(
            "DAYS_BIRTH"
        )

        age_years = None

        if pd.notna(days_birth):
            age_years = round(
                abs(float(days_birth)) / 365.25,
                1,
            )

        days_employed = application_row.get(
            "DAYS_EMPLOYED"
        )

        employment_years = None

        if (
            pd.notna(days_employed)
            and float(days_employed)
            != days_employed_sentinel
        ):
            employment_years = round(
                abs(float(days_employed)) / 365.25,
                1,
            )

        return CustomerSummaryResponse(
            SK_ID_CURR=customer_id,
            income_total=_safe_float(
                application_row.get(
                    "AMT_INCOME_TOTAL"
                )
            ),
            credit_amount=_safe_float(
                application_row.get(
                    "AMT_CREDIT"
                )
            ),
            annuity_amount=_safe_float(
                application_row.get(
                    "AMT_ANNUITY"
                )
            ),
            goods_price=_safe_float(
                application_row.get(
                    "AMT_GOODS_PRICE"
                )
            ),
            age_years=age_years,
            employment_years=employment_years,
            education_type=_safe_str(
                application_row.get(
                    "NAME_EDUCATION_TYPE"
                )
            ),
            income_type=_safe_str(
                application_row.get(
                    "NAME_INCOME_TYPE"
                )
            ),
            family_status=_safe_str(
                application_row.get(
                    "NAME_FAMILY_STATUS"
                )
            ),
            has_bureau_history=bool(
                pd.notna(
                    bureau_row.get(
                        "BUREAU_LOAN_COUNT"
                    )
                )
            ),
            has_previous_application_history=bool(
                pd.notna(
                    previous_row.get(
                        "PREV_APPLICATION_COUNT"
                    )
                )
            ),
            has_installment_history=bool(
                pd.notna(
                    installments_row.get(
                        "INST_PAYMENT_RECORD_COUNT"
                    )
                )
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load customer summary: "
                f"{exc}"
            ),
        ) from exc


@app.post(
    "/demo/predict/{customer_id}",
    response_model=PredictionResponse,
)
def demo_predict(
    customer_id: int
):
    """
    Generate a prediction for a customer from
    the local Home Credit demo dataset.
    """

    try:
        demo_data = get_demo_data()

        if (
            customer_id
            not in demo_data[
                "common_id_set"
            ]
        ):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Customer was not found "
                    "in all required demo "
                    "feature sources."
                ),
            )

        application_df = (
            demo_data["application"]
            .loc[
                demo_data[
                    "application"
                ]["SK_ID_CURR"]
                == customer_id
            ]
            .copy()
        )

        bureau_df = (
            demo_data["bureau"]
            .loc[
                demo_data[
                    "bureau"
                ]["SK_ID_CURR"]
                == customer_id
            ]
            .copy()
        )

        previous_df = (
            demo_data["previous"]
            .loc[
                demo_data[
                    "previous"
                ]["SK_ID_CURR"]
                == customer_id
            ]
            .copy()
        )

        installments_df = (
            demo_data["installments"]
            .loc[
                demo_data[
                    "installments"
                ]["SK_ID_CURR"]
                == customer_id
            ]
            .copy()
        )

        prediction = predict_risk_scores(
            application_df=application_df,
            bureau_df=bureau_df,
            previous_df=previous_df,
            installments_df=installments_df,
            model=get_runtime_model(),
            schema=get_runtime_schema(),
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
                "Demo prediction failed: "
                f"{exc}"
            ),
        ) from exc


@app.post(
    "/predict",
    response_model=PredictionResponse,
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