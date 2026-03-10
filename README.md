# Seoul Bike Sharing Demand Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)

## Overview

A modular ML pipeline for predicting hourly bike rental demand in Seoul, South Korea using the [UCI Seoul Bike Sharing dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand). The pipeline predicts demand based on weather conditions (temperature, humidity, wind speed) and temporal features (hour, season, holiday).

## Project Structure

```
Seoul-Bike-Prediction/
├── data/               # Raw data (CSV)
├── models/             # Serialized .joblib models
├── src/
│   ├── data_loader.py         # Data ingestion & cleaning
│   ├── feature_engineering.py # Scikit-learn pipelines & transformers
│   ├── model_trainer.py       # Model training (Linear Regression / XGBoost)
│   └── config.py              # Centralized configuration
├── tests/              # Unit & integration tests
├── main.py             # Pipeline entry point
└── requirements.txt
```

Key design choices:
- Separated data loading, feature engineering, and model training into independent modules
- Used scikit-learn pipelines for reproducible preprocessing
- Included unit tests for data integrity and pipeline logic
- Handled encoding edge cases (BOM/unicode in the raw CSV)

## Model Performance

Results on 20% held-out test set (80/20 split, `random_state=42`):

| Model | R² | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| Linear Regression (baseline) | — | — | — |
| **XGBoost** | **0.936** | **163.69** | **96.77** |

Performance benefits from engineered interaction features between temperature and seasonality.

## Usage

```bash
git clone https://github.com/Jiaweisun274/Seoul-Bike-Prediction.git
cd Seoul-Bike-Prediction
pip install -r requirements.txt

# Train and evaluate
python main.py

# Run tests
python -m pytest tests/
```

Trained models are saved to `models/`.

## Possible Extensions

- SHAP analysis for feature importance and model interpretability
- Time-series aware train/test split (chronological instead of random)
- REST API deployment with FastAPI
- CI/CD with GitHub Actions
