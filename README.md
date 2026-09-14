# CreditLens AI

Machine-learning based credit risk decision-support system built on the Home Credit Default Risk dataset.

> **CreditLens AI is not an automatic credit approval/rejection system.** It is a decision-support and research prototype: a FastAPI inference service and React dashboard around an explainable CatBoost risk model, with post-hoc fairness diagnostics. All final lending decisions remain a human and organizational responsibility.

[![CreditLens CI](https://github.com/Ayfernaz-Baygin/creditlens-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Ayfernaz-Baygin/creditlens-ai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)
![React](https://img.shields.io/badge/React-19-61DAFB)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

---

## Overview

CreditLens AI:

- Models credit default risk on the Home Credit Default Risk dataset using relational, customer-level features.
- Serves predictions through a FastAPI inference layer backed by a trained CatBoost classifier.
- Ships a React/Vite dashboard for exploring demo customers, risk scores, and explanations.
- Produces local SHAP explanations for individual predictions.
- Reports post-hoc fairness diagnostics across demographic groups.
- Runs in a Docker + Nginx production-like local stack.
- Is validated on every push/PR by a GitHub Actions CI pipeline.

What it deliberately does **not** do:

- It does not make an automatic credit approval/rejection decision.
- Its output (`risk_score`) is **not** a calibrated probability of default.
- Its fairness diagnostics do not amount to a legal or normative "fair" / "unfair" judgment.

---

## Architecture

```text
Browser
   │
   ▼
Nginx (:8080)
   ├── React / Vite static frontend
   └── /api/*  ───────────────►  FastAPI (:8000)
                                       │
                                       ▼
                             Feature Engineering
                                       │
                                       ▼
                                  CatBoost
                                       │
                                       ▼
                        Risk Score + SHAP Explanation
```

In local development, the React dev server (`:5173`) talks directly to FastAPI (`:8000`) over CORS — there is no Nginx in that loop. The Docker stack described above is what turns that into a single-origin (`:8080`) deployment where the browser only ever calls same-origin `/api/...` paths.

---

## Features

- Customer-level risk scoring from a shared, production-identical feature-engineering pipeline (no train/serve skew).
- 194 predictive features (179 numeric, 15 categorical) built from application data plus aggregated bureau, previous-application, and installment-payment history.
- CatBoost gender-free final model — `CODE_GENDER` is **not** a predictive input; it is used only for post-hoc fairness auditing.
- Local SHAP explanations for individual predictions, with feature-level direction (`increases_score` / `decreases_score`).
- Customer profile view (income, credit terms, demographics, credit-history flags).
- Model Information dashboard sourced from the live `/model-info` endpoint.
- Post-hoc fairness diagnostics across major demographic groups, sourced from the live `/fairness-summary` endpoint.
- Dockerized production-like stack (Nginx + FastAPI + React) with the dataset mounted read-only, never baked into the image.
- GitHub Actions CI validating backend tests, frontend build, and Docker build on every push/PR.

---

## Model Development

The model was developed through six notebooks, in order:

| Notebook | Purpose |
|---|---|
| `01_data_understanding.ipynb` | Dataset structure, class balance, missing values |
| `02_baseline_model.ipynb` | Baseline logistic regression |
| `03_relational_feature_engineering.ipynb` | Customer-level features from bureau/previous-application/installment tables |
| `04_boosting_model.ipynb` | LightGBM vs. CatBoost comparison |
| `05_final_validation.ipynb` | Cross-validation, gender ablation, fairness analysis |
| `06_final_model.ipynb` | Final full-data model training and artifact export |

### Model Comparison (full feature set)

| Model | ROC-AUC | PR-AUC | F1 @ 0.50 |
|---|---:|---:|---:|
| Logistic Regression | 0.7655 | 0.2473 | 0.2734 |
| LightGBM | 0.7815 | 0.2779 | 0.2970 |
| CatBoost | **0.7842** | **0.2828** | **0.3040** |

### Gender-Free Ablation

`CODE_GENDER` showed meaningful influence during explainability analysis, so a second CatBoost model was trained without it:

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| CatBoost (full, incl. `CODE_GENDER`) | 0.7842 | 0.2828 | 0.3040 |
| CatBoost (gender-free) | 0.7821 | 0.2797 | 0.3013 |

The predictive-performance loss from removing `CODE_GENDER` was small, so the gender-free model was kept as the primary research candidate. `CODE_GENDER` is retained in the dataset only for the post-hoc fairness audit below.

---

## Final Model

Model artifact:

```text
models/creditlens_catboost_gender_free.cbm
models/creditlens_feature_schema.json
models/creditlens_model_metadata.json
```

Configuration (from `creditlens_model_metadata.json`):

| | |
|---|---|
| Model | CatBoostClassifier |
| Training rows | 307,511 |
| Features | 194 total — 179 numeric, 15 categorical |
| Excluded from predictive inputs | `CODE_GENDER`, `TARGET`, `SK_ID_CURR` |
| Iterations | 1,470 |
| Learning rate | 0.03 |
| Depth | 7 |
| L2 leaf regularization | 5 |
| Class weighting | Balanced |
| Random seed | 42 |

The feature schema (`creditlens_feature_schema.json`) stores the exact ordered feature contract, categorical/numeric splits, excluded columns, and missing-value handling used by both training and inference — this is the mechanism that keeps `src/feature_engineering.py` train/serve-consistent.

---

## Model Performance

### Development Validation vs. Out-of-Fold (final gender-free model)

| Metric | Development Validation | Out-of-Fold (3-fold CV) |
|---|---:|---:|
| ROC-AUC | 0.7821 | 0.7779 |
| PR-AUC | 0.2797 | 0.2684 |
| F1 @ 0.50 | 0.3013 | 0.3003 |
| Recall @ 0.50 | — | 0.6395 |

### 3-Fold Cross-Validation (mean ± std)

- ROC-AUC: **0.7779 ± 0.0012**
- PR-AUC: **0.2685 ± 0.0050**
- F1 @ 0.50: **0.3003 ± 0.0010**
- Recall @ 0.50: **0.6395 ± 0.0071**

The small spread across folds indicates relatively stable performance. Out-of-fold results are used as a stability estimate rather than a fully independent test, since the model architecture and training configuration were selected during development.

---

## Risk Score Interpretation

**The model output is a risk score produced by a class-balanced model. It must NOT be interpreted as a calibrated probability of default.**

The dashboard's descriptive UI bands — *Lower*, *Moderate*, *Higher* — are visualization groupings only. They are:

- **not** a lending threshold,
- **not** an approval/rejection threshold,
- **not** a regulatory decision rule.

A statement such as *"34% probability of default"* is an incorrect reading of `risk_score`. Turning this score into a probability would require a separate, explicit probability-calibration step that this project does not currently perform (see [Limitations](#limitations)).

---

## Explainability

`GET /demo/explain/{customer_id}` returns a local SHAP-based explanation for one customer's risk score, computed with CatBoost's native SHAP implementation.

Each top feature contribution includes:

- the feature's raw value for that customer,
- its SHAP contribution in the model's raw margin space,
- a direction label: `increases_score` or `decreases_score`.

Representative predictive features include `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `AMT_GOODS_PRICE`, and `BUREAU_DAYS_CREDIT_MAX`, among the full 194-feature set.

**SHAP contributions describe model behavior, not causality.** A feature that increases the score is not necessarily a "cause" of higher risk in any real-world sense — it reflects the pattern the model learned from historical data.

---

## Fairness Diagnostics

`CODE_GENDER` is **not** a predictive input to the model. It is used exclusively for the post-hoc audit below, sourced from `GET /fairness-summary` (backed by `reports/catboost_gender_fairness_*.csv`).

| Group | Samples | ROC-AUC | Recall | FPR |
|---|---:|---:|---:|---:|
| F | 40,561 | 0.7768 | 0.6199 | 0.2103 |
| M | 20,940 | 0.7814 | 0.6886 | 0.2805 |

**Major-group gaps:**

| Metric | Gap |
|---|---:|
| ROC-AUC | 0.0047 |
| Recall | 0.0687 |
| False Positive Rate | 0.0701 |

A third group (`XNA`, n=2) exists in the raw audit output but is excluded from the headline gap comparison above because its sample size is too small to be meaningful.

**These are post-hoc diagnostic group differences and do not establish whether the system is fair or unfair.** Ranking performance (ROC-AUC) is close between groups; recall and false-positive-rate gaps at the default 0.50 operating point are more pronounced and would warrant further threshold-level analysis before any real-world use.

---

## API

FastAPI base URL (local development): `http://127.0.0.1:8000`
Interactive Swagger docs: `http://127.0.0.1:8000/docs`

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API name, version, and status |
| `GET` | `/health` | Checks that the model, schema, and metadata artifacts are available |
| `GET` | `/model-info` | Model configuration and validation/stability metrics |
| `GET` | `/fairness-summary` | Post-hoc fairness diagnostics by demographic group |
| `GET` | `/demo/customers` | Lists demo customer IDs available in the local dataset |
| `GET` | `/demo/customers/{customer_id}` | Readable profile summary for one demo customer |
| `POST` | `/demo/predict/{customer_id}` | Generates a risk score for a demo customer |
| `GET` | `/demo/explain/{customer_id}` | Local SHAP explanation for a demo customer's risk score |
| `POST` | `/predict` | Generates a risk score from caller-supplied feature payloads |

---

## Dashboard

The React dashboard has four pages:

- **Dashboard** — model summary cards, a *Last Analysis* card (reflects the most recent Risk Analysis run in the current session), a *Model Snapshot*, a *Fairness Diagnostics* summary, and *Quick Actions* shortcuts to the other pages.
- **Risk Analysis** — demo customer selection, customer profile, the live Model Risk Score with its descriptive band, and the SHAP explanation for that prediction.
- **Model Information** — model configuration and validation/stability metrics, from `/model-info`.
- **Fairness** — the full fairness diagnostics table and major-group gaps, from `/fairness-summary`.

*(No UI screenshots are included in this README.)*

---

## Dataset

CreditLens AI uses the **Home Credit Default Risk** dataset (originally distributed via the Kaggle competition of the same name).

The raw and interim dataset files are **not** stored in this Git repository. This is intentional:

- the raw tables are multiple gigabytes in total, which does not belong in a Git history;
- the dataset carries its own licensing/distribution terms via Kaggle, so it is not redistributed here.

To run the demo endpoints locally, the following files must exist on disk (paths relative to the repository root):

```text
data/raw/application_test.csv
data/interim/bureau_customer_features.csv
data/interim/previous_application_customer_features.csv
data/interim/installments_customer_features.csv
```

There is no automated download step for these files, and no Kaggle credentials are required by any script in this repository — obtaining and placing the dataset is a manual prerequisite.

---

## Local Development

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn src.api:app --reload
```

In a second terminal, start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

| Service | URL |
|---|---|
| Frontend | `http://localhost:5173` |
| FastAPI | `http://127.0.0.1:8000` |
| Swagger | `http://127.0.0.1:8000/docs` |

---

## Docker

Build and start the production-like local stack:

```bash
docker compose build
docker compose up -d
```

| Service | URL |
|---|---|
| Application (Nginx + React, proxying `/api`) | `http://localhost:8080` |
| API through the Nginx proxy | `http://localhost:8080/api/health` |
| FastAPI directly (debug/Swagger) | `http://localhost:8000/docs` |

Architecture: an Nginx container serves the built React frontend and proxies `/api/*` to the FastAPI container. `data/raw` and `data/interim` are mounted **read-only** from the host into the backend container (`docker-compose.yml`) — the dataset is never copied into the image (see `.dockerignore`).

Stop the stack:

```bash
docker compose down
```

This does not remove the host `data/` directory — the mounts are bind mounts, not managed volumes.

---

## Testing

Backend tests:

```bash
python -m unittest discover -s tests -v
```

Current result: **30 tests, all passing.**

Frontend production build:

```bash
npm --prefix frontend run build
```

Docker validation:

```bash
docker compose config
docker compose build
```

---

## Continuous Integration

Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — **CreditLens CI**

Triggers: `push` and `pull_request` on `main`.

Three independent jobs run in parallel:

| Job | Validates |
|---|---|
| `backend-tests` | Installs `requirements-backend.txt` + `httpx`, runs the full `unittest` suite |
| `frontend-build` | `npm ci` + `npm run build` for the Vite frontend |
| `docker-build` | `docker compose config` + `docker compose build` for both images |

The CI pipeline does **not** download the raw/interim dataset and does **not** run `docker compose up` — `data/raw` and `data/interim` are absent in CI by design, and image build/config resolution does not require them (only `docker compose up` would).

---

## Project Structure

```text
creditlens-ai/
├── src/                       # FastAPI app, inference, feature engineering
├── tests/                     # Backend automated test suite
├── notebooks/                 # Model development notebooks (01-06)
├── models/                    # Trained model artifact, schema, metadata
├── reports/                   # Validation, SHAP, and fairness CSV reports
├── frontend/                  # React + Vite dashboard, Dockerfile, nginx.conf
├── data/                      # raw/ and interim/ are git-ignored (see Dataset)
├── .github/workflows/ci.yml
├── Dockerfile.backend
├── docker-compose.yml
├── requirements.txt
├── requirements-backend.txt
└── README.md
```

---

## Design Principles

- Gender-free predictive model, with sensitive-attribute use limited to post-hoc auditing.
- Explainability as a first-class output (SHAP), not an afterthought.
- Reproducibility: a fixed feature schema and shared feature-engineering code between training and serving.
- Clean separation between the model, the API, and the UI.
- Decision support over autonomous decision-making.
- No calibrated-probability claim without an explicit calibration step.
- Fairness treated as ongoing post-hoc monitoring, not a one-time pass/fail check.

---

## Limitations

- Built on a Kaggle research dataset, not real-time production banking data.
- Class-balanced training affects how the raw score should be read; it is not directly comparable to an unbalanced model's output.
- No probability calibration has been applied — `risk_score` is not a probability.
- No external, prospective (out-of-time or out-of-population) validation has been performed.
- Fairness analysis is limited to the demographic groups and sample sizes available in this dataset (the `XNA` group, n=2, is too small to analyze).
- SHAP explanations are descriptive of model behavior, not causal claims.
- This is not a lending approval/rejection system and is not fit for that purpose as-is.

---

## Future Work

- Probability calibration of the risk score.
- Decision-curve / cost-sensitive threshold analysis.
- Model monitoring and drift detection.
- Stronger integration testing against fixture datasets (independent of the real Home Credit files).
- Deployment to a hosted cloud environment.
- API authentication / rate limiting if the service is ever exposed publicly.
- Automated model/version registry.
- Accessibility improvements to the dashboard.

---

## Research and Responsible-Use Note

CreditLens AI is an educational and research project. Running it in Docker behind Nginx makes it easier to demo end-to-end — it does not change its research/decision-support nature.

Credit decisions can have significant consequences for individuals and may be subject to legal, regulatory, ethical, and organizational requirements. This project treats machine-learning output as decision-support information, not an autonomous lending decision.

`CODE_GENDER` is not used by the primary predictive model, but excluding it does not by itself guarantee fairness — correlated proxy variables and threshold-dependent disparities (as observed in the fairness diagnostics above) may still exist. Model outputs should be evaluated together with calibration, fairness analysis, human oversight, domain expertise, and applicable regulatory requirements.
