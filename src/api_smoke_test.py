from pathlib import Path
import json
import urllib.request

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
REPORTS_DIR = PROJECT_ROOT / "reports"

API_URL = "http://127.0.0.1:8000/predict"


def clean_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def row_to_dict(row):
    return {
        key: clean_value(value)
        for key, value in row.to_dict().items()
    }


# Load input data
application = pd.read_csv(
    RAW_DIR / "application_test.csv",
    low_memory=False
)

bureau = pd.read_csv(
    INTERIM_DIR / "bureau_customer_features.csv"
)

previous = pd.read_csv(
    INTERIM_DIR
    / "previous_application_customer_features.csv"
)

installments = pd.read_csv(
    INTERIM_DIR
    / "installments_customer_features.csv"
)


# Find a customer available in all four datasets
common_ids = (
    set(application["SK_ID_CURR"])
    & set(bureau["SK_ID_CURR"])
    & set(previous["SK_ID_CURR"])
    & set(installments["SK_ID_CURR"])
)

if not common_ids:
    raise RuntimeError(
        "No customer exists in all four datasets."
    )


customer_id = sorted(common_ids)[0]


# Select one customer from every feature source
application_row = application.loc[
    application["SK_ID_CURR"] == customer_id
].iloc[0]

bureau_row = bureau.loc[
    bureau["SK_ID_CURR"] == customer_id
].iloc[0]

previous_row = previous.loc[
    previous["SK_ID_CURR"] == customer_id
].iloc[0]

installments_row = installments.loc[
    installments["SK_ID_CURR"] == customer_id
].iloc[0]


# Build API payload
payload = {
    "application": row_to_dict(
        application_row
    ),
    "bureau": row_to_dict(
        bureau_row
    ),
    "previous": row_to_dict(
        previous_row
    ),
    "installments": row_to_dict(
        installments_row
    ),
}


encoded_payload = json.dumps(
    payload
).encode("utf-8")


request = urllib.request.Request(
    API_URL,
    data=encoded_payload,
    headers={
        "Content-Type": "application/json"
    },
    method="POST",
)


# Send request to FastAPI
with urllib.request.urlopen(
    request
) as response:
    result = json.loads(
        response.read().decode("utf-8")
    )


print(
    f"Customer: "
    f"{result['SK_ID_CURR']}"
)

print(
    f"Risk score: "
    f"{result['risk_score']:.6f}"
)

print(
    f"Interpretation: "
    f"{result['interpretation']}"
)


# Load previously validated notebook/CLI predictions
expected_predictions = pd.read_csv(
    REPORTS_DIR / "final_test_risk_scores.csv"
)


matching_rows = expected_predictions.loc[
    expected_predictions["SK_ID_CURR"]
    == customer_id
]


if matching_rows.empty:
    raise RuntimeError(
        f"Customer {customer_id} "
        "was not found in final_test_risk_scores.csv"
    )


expected_score = float(
    matching_rows[
        "risk_score"
    ].iloc[0]
)

api_score = float(
    result["risk_score"]
)


score_difference = abs(
    api_score - expected_score
)


scores_match = np.isclose(
    api_score,
    expected_score,
    rtol=0,
    atol=1e-12
)


print()

print(
    f"Expected score: "
    f"{expected_score:.12f}"
)

print(
    f"API score: "
    f"{api_score:.12f}"
)

print(
    f"Difference: "
    f"{score_difference:.12f}"
)

print(
    f"Scores match: "
    f"{scores_match}"
)


assert result["SK_ID_CURR"] == customer_id
assert scores_match


print()
print(
    "API regression check PASSED."
)