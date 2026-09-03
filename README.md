# CreditLens AI

CreditLens AI is an explainable credit risk modeling and decision-support research project built on the Home Credit Default Risk dataset.

The project implements an end-to-end machine learning workflow covering data understanding, relational feature engineering, baseline modeling, gradient boosting, explainability, fairness auditing, cross-validation, model artifact generation, and inference.

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
- Support future API and interactive dashboard integration

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

Customer-level features were generated from:

### Bureau history

Examples:

- bureau loan count
- average and maximum debt
- total and maximum credit amounts
- credit activity status
- credit history timing information

### Previous applications

Examples:

- previous application count
- approval and refusal rates
- previous annuity statistics
- credit-to-application ratios
- payment term statistics

### Installment payments

Examples:

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

Three major model configurations were evaluated using the same development validation framework.

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

### Mean CV performance

- ROC-AUC: **0.7779 ± 0.0012**
- PR-AUC: **0.2685 ± 0.0050**
- F1 @ 0.50: **0.3003 ± 0.0010**
- Recall @ 0.50: **0.6395 ± 0.0071**

### Out-of-Fold performance

- ROC-AUC: **0.7779**
- PR-AUC: **0.2684**
- F1: **0.3003**
- Recall: **0.6395**

The small variation between folds indicates relatively stable model performance across different stratified partitions.

---

## Threshold Analysis

The model output is treated as a **risk score** rather than a calibrated probability of default.

Different operating thresholds produce different precision-recall trade-offs.

For example, development experiments showed:

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

---

## Fairness Audit

The gender-free model was audited post-hoc using `CODE_GENDER`.

The two sufficiently represented groups showed similar ranking performance:

- ROC-AUC gap: approximately **0.0047**

However, threshold-dependent differences were observed:

- Recall gap: approximately **6.9 percentage points**
- False-positive-rate gap: approximately **7.0 percentage points**
- False-negative-rate gap: approximately **6.9 percentage points**

A very small `XNA` group was preserved in raw audit outputs but excluded from headline fairness gap calculations because of insufficient sample size.

These results are treated as diagnostic measurements rather than proof that the model is fair or unfair.

---

## Final Model Artifact

The final research model is stored as:

```text
models/creditlens_catboost_gender_free.cbm