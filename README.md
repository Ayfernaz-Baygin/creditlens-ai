# CreditLens AI

CreditLens AI is an explainable credit risk modeling and decision-support research project built on the Home Credit Default Risk dataset.

The project implements an end-to-end machine learning workflow covering data understanding, relational feature engineering, baseline modeling, gradient boosting, explainability, fairness auditing, cross-validation, model artifact generation, reusable inference, automated testing, and FastAPI-based model serving.

The current primary research model is a gender-free CatBoost classifier that excludes `CODE_GENDER` from the predictive feature set.

> CreditLens AI is designed as a research and decision-support system. It is not intended to autonomously approve or reject credit applications.

---

## Project Objectives

- Build a reproducible credit default risk modeling pipeline
- Analyze relational financial datasets
- Engineer customer-level features from credit history tables
- Compare linear and gradient-boosting models
- Evaluate imbalanced classification performance using appropriate metrics
- Perform threshold and cost-sensitive analysis
- Explain model behavior using feature importance and SHAP values
- Audit model performance across demographic groups
- Reduce training-serving skew through shared feature-engineering code
- Export a reusable trained model and inference pipeline
- Serve model predictions through a FastAPI API
- Support future interactive dashboard integration

---

## Dataset

The project uses the **Home Credit Default Risk** dataset.

The dataset contains multiple relational tables, including:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`
- `POS_CASH_balance.csv`

Raw datasets are intentionally excluded from the repository.

The main training dataset contains:

- **307,511 labeled customers**
- Approximately **8.07% positive/default class**
- Strong class imbalance of approximately **11.4:1**

---

## Relational Feature Engineering

Customer-level features were generated from multiple relational data sources.

### Bureau History

Examples include:

- bureau loan count
- average and maximum debt
- total and maximum credit amounts
- credit activity status
- credit history timing information

### Previous Applications

Examples include:

- previous application count
- approval and refusal rates
- previous annuity statistics
- credit-to-application ratios
- payment term statistics

### Installment Payments

Examples include:

- late payment rates
- days past due statistics
- underpayment rates
- payment shortfalls
- payment-to-installment ratios

The final predictive matrix contains:

- **194 model features**
- **179 numeric features**
- **15 categorical features**

`CODE_GENDER`, `TARGET`, and `SK_ID_CURR` are excluded from the final predictive feature set.

---

## Model Development

Three major model configurations were evaluated using the same development-validation framework.

| Model | ROC-AUC | PR-AUC | F1 @ 0.50 |
|---|---:|---:|---:|
| Logistic Regression | 0.7655 | 0.2473 | 0.2734 |
| LightGBM | 0.7815 | 0.2779 | 0.2970 |
| CatBoost | **0.7842** | **0.2828** | **0.3040** |

CatBoost achieved the strongest development-validation performance.

---

## Fairness-Oriented Ablation

`CODE_GENDER` was found to have meaningful model influence during explainability analysis.

A second CatBoost model was therefore trained without `CODE_GENDER`.

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| Full CatBoost | 0.7842 | 0.2828 | 0.3040 |
| Gender-Free CatBoost | 0.7821 | 0.2797 | 0.3013 |

The predictive performance loss was limited, so the gender-free model was retained as the primary research candidate.

`CODE_GENDER` is retained only for post-hoc fairness auditing and is not used as a predictive input.

---

## Cross-Validation

The primary gender-free CatBoost model was evaluated using 3-fold stratified cross-validation.

### Mean CV Performance

- ROC-AUC: **0.7779 ± 0.0012**
- PR-AUC: **0.2685 ± 0.0050**
- F1 @ 0.50: **0.3003 ± 0.0010**
- Recall @ 0.50: **0.6395 ± 0.0071**

### Out-of-Fold Performance

- ROC-AUC: **0.7779**
- PR-AUC: **0.2684**
- F1: **0.3003**
- Recall: **0.6395**

The small variation between folds indicates relatively stable model performance across different stratified partitions.

The out-of-fold evaluation is used as a stability estimate rather than as a completely independent untouched test set because model architecture and training configuration were selected during development.

---

## Threshold Analysis

The model output is treated as a **risk score** rather than a calibrated probability of default.

Different operating thresholds produce different precision-recall trade-offs.

Development experiments showed that:

- lower thresholds increase recall
- higher thresholds increase precision
- the threshold that maximizes F1 is not necessarily appropriate for credit-risk screening
- the preferred threshold depends on the relative cost of false positives and false negatives

No universal or demographic-group-specific lending threshold is hard-coded into the project.

---

## Explainability

The project includes both CatBoost feature importance and SHAP-based analysis.

Important predictors include:

- `EXT_SOURCE_1`
- `EXT_SOURCE_2`
- `EXT_SOURCE_3`
- `AMT_CREDIT`
- `AMT_GOODS_PRICE`
- `DAYS_EMPLOYED`
- previous application features
- bureau debt features
- installment late-payment features

Relational financial-history features account for a substantial portion of total model importance.

SHAP analysis is used to evaluate both feature importance and the direction of model effects.

These explanations describe model behavior and should not be interpreted as causal relationships.

---

## Fairness Audit

The gender-free model was audited post-hoc using `CODE_GENDER`.

The two sufficiently represented groups showed similar ranking performance:

- ROC-AUC gap: approximately **0.0047**

However, threshold-dependent differences were observed:

- Recall gap: approximately **6.9 percentage points**
- False-positive-rate gap: approximately **7.0 percentage points**
- False-negative-rate gap: approximately **6.9 percentage points**

A very small `XNA` group was preserved in raw audit outputs but excluded from headline fairness-gap calculations because of insufficient sample size.

These results are treated as diagnostic measurements rather than proof that the model is fair or unfair.

---

## Final Model Artifact

The final research model is stored as:

```text
models/creditlens_catboost_gender_free.cbm
```

The final model was trained on all **307,511 labeled training rows**.

Model configuration:

- CatBoostClassifier
- 1,470 trees
- learning rate: 0.03
- depth: 7
- L2 leaf regularization: 5
- balanced class weighting
- random seed: 42
- gender-free predictive feature set

Additional model artifacts:

```text
models/creditlens_feature_schema.json
models/creditlens_model_metadata.json
```

The schema stores the exact ordered model feature contract, categorical and numeric features, excluded fields, missing-value handling, and engineered feature settings.

The metadata file stores training configuration, model information, validation metrics, and inference metadata.

---

## Shared Feature Engineering

Production inference uses the same shared feature-engineering implementation as the validated model workflow.

The main feature-engineering module is:

```text
src/feature_engineering.py
```

It is responsible for:

- merging application and customer-level relational features
- checking customer identifiers
- detecting duplicate relational rows
- creating history-presence flags
- handling the `DAYS_EMPLOYED` sentinel value
- enforcing the saved feature schema
- preparing categorical missing values
- preserving exact feature order

This reduces training-serving skew between experimentation and inference.

---

## Command-Line Inference

The project includes a reusable command-line inference pipeline:

```text
src/inference.py
```

Example:

```bash
python src/inference.py \
  --application data/raw/application_test.csv \
  --bureau data/interim/bureau_customer_features.csv \
  --previous data/interim/previous_application_customer_features.csv \
  --installments data/interim/installments_customer_features.csv \
  --output reports/predictions.csv
```

The inference pipeline expects:

- application-level customer data
- already aggregated customer-level bureau features
- already aggregated previous-application features
- already aggregated installment-payment features

The current production inference layer does not aggregate the original multi-million-row relational raw tables during each prediction request.

---

## FastAPI Inference API

CreditLens AI includes a FastAPI-based model-serving layer.

The API entry point is:

```text
src/api.py
```

Start the API locally with:

```bash
uvicorn src.api:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

#### `GET /`

Returns basic API status and version information.

#### `GET /health`

Checks whether the required model, schema, and metadata artifacts are available.

#### `GET /model-info`

Returns information about the deployed research model, including:

- model type
- training row count
- feature count
- categorical and numeric feature counts
- excluded sensitive feature
- training iterations
- development-validation metrics
- out-of-fold metrics
- risk-score interpretation information

#### `POST /predict`

Generates a customer-level credit risk score from:

- application-level features
- aggregated bureau features
- aggregated previous-application features
- aggregated installment-payment features

Example response:

```json
{
  "SK_ID_CURR": 100001,
  "risk_score": 0.346927880706,
  "interpretation": "Risk score from the class-balanced CatBoost model. This value is not a calibrated probability of default."
}
```

The API validates that relational feature records belong to the same customer.

The trained CatBoost model and feature schema are cached in memory and reused across requests instead of being reloaded from disk for every prediction.

The returned `risk_score` must not be interpreted as a calibrated default probability.

---

## API Regression Check

A smoke-test script is included:

```text
src/api_smoke_test.py
```

It selects a customer present in all required feature sources, sends the customer data to the live FastAPI `/predict` endpoint, and compares the returned risk score against the previously validated inference output.

Validated result:

```text
Customer: 100001

Expected score: 0.346927880706
API score:      0.346927880706
Difference:     0.000000000000
Scores match:   True

API regression check PASSED.
```

This verifies that the API-serving layer preserves the validated model output.

---

## Automated Tests

The project currently contains **20 automated tests**.

The test suite covers:

- feature merging
- customer-history flags
- `DAYS_EMPLOYED` sentinel handling
- model feature-schema enforcement
- categorical missing-value handling
- missing-feature validation
- unexpected-feature validation
- duplicate customer-row detection
- customer-ID validation
- saved model loading
- model artifact availability
- inference output contracts
- NaN prediction protection
- FastAPI root endpoint
- FastAPI health endpoint
- model-information endpoint
- prediction endpoint behavior
- mismatched customer-ID rejection

Run all tests with:

```bash
python -m unittest discover -s tests -v
```

Current validated result:

```text
Ran 20 tests

OK
```

---

## Project Structure

```text
creditlens-ai/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── creditlens_catboost_gender_free.cbm
│   ├── creditlens_feature_schema.json
│   └── creditlens_model_metadata.json
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_relational_feature_engineering.ipynb
│   ├── 04_boosting_model.ipynb
│   ├── 05_final_validation.ipynb
│   └── 06_final_model.ipynb
│
├── reports/
│
├── src/
│   ├── api.py
│   ├── api_smoke_test.py
│   ├── feature_engineering.py
│   └── inference.py
│
├── tests/
│   ├── test_api.py
│   ├── test_feature_engineering.py
│   └── test_inference.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Ayfernaz-Baygin/creditlens-ai.git
cd creditlens-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Technology Stack

The project currently uses:

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- CatBoost
- SHAP
- Jupyter
- FastAPI
- Pydantic
- Uvicorn
- HTTPX
- unittest
- Git
- GitHub

---

## Current Status

The following major components are complete:

- data understanding
- baseline logistic regression
- relational feature engineering
- LightGBM modeling
- CatBoost modeling
- threshold analysis
- cost-sensitive analysis
- feature importance analysis
- SHAP analysis
- fairness-oriented gender ablation
- post-hoc fairness audit
- stratified cross-validation
- out-of-fold evaluation
- final gender-free model training
- model artifact export
- reusable feature-engineering pipeline
- reusable command-line inference
- automated test suite
- FastAPI inference service
- Swagger API documentation
- API regression verification

---

## Future Work

Planned extensions include:

- interactive frontend/dashboard
- human-vs-model decision analysis
- probability calibration
- extended fairness analysis
- Docker containerization
- continuous integration with GitHub Actions
- experiment tracking
- model and data versioning
- production monitoring
- deployment to a hosted environment

---

## Research and Responsible-Use Note

CreditLens AI is an educational and research project.

Credit decisions can have significant consequences for individuals and may be subject to legal, regulatory, ethical, and organizational requirements.

The project therefore treats machine-learning output as decision-support information rather than an autonomous lending decision.

Sensitive attributes are not used by the primary predictive model, but their exclusion alone does not guarantee fairness because correlated proxy variables and threshold-dependent disparities may still exist.

Model outputs should be evaluated together with calibration, fairness analysis, human oversight, domain expertise, and applicable regulatory requirements.