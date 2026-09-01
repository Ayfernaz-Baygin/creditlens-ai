# CreditLens AI

CreditLens AI is an explainable credit risk analysis and human-vs-model decision research platform.

The project focuses on building an end-to-end machine learning pipeline for predicting credit default risk using historical financial data, explaining model predictions, evaluating fairness, and comparing AI predictions with human assessments.

## Project Goals

- Build a reproducible credit risk machine learning pipeline
- Perform exploratory data analysis and feature engineering on relational financial datasets
- Compare multiple machine learning models
- Explain individual and global predictions using Explainable AI techniques
- Evaluate model performance across different groups
- Compare AI predictions with human decisions
- Expose the trained model through an API and interactive dashboard

## Planned Tech Stack

### Data & Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost / LightGBM / CatBoost
- SHAP

### MLOps

- MLflow
- DVC
- Docker
- GitHub Actions

### Backend

- FastAPI
- Pydantic

### Frontend

- React

## Dataset

The project will use the Home Credit Default Risk dataset.

The dataset contains multiple relational tables including:

- application data
- previous loan applications
- credit bureau history
- installment payments
- credit card balances
- POS cash balances

Raw datasets are not stored in this repository.

## Project Structure

```text
creditlens-ai/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- models/
|-- notebooks/
|-- reports/
|   `-- figures/
|-- src/
|   `-- creditlens/
|       |-- data/
|       |-- evaluation/
|       |-- features/
|       `-- models/
|-- tests/
|-- .gitignore
|-- requirements.txt
`-- README.md