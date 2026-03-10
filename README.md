# Seoul Bike Sharing Demand Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)

## Project Overview

This project builds a modular machine learning pipeline to predict hourly bike rental demand in Seoul, South Korea.

By predicting demand based on weather conditions (Temperature, Humidity, Wind Speed) and temporal features, this system can help bike-sharing operators optimize fleet distribution and reduce operational costs.

## Key Technical Highlights

- **Modular Architecture**: Code is separated into `data_loader`, `feature_engineering`, and `model_trainer` for maintainability and testability.
- **XGBoost Regressor**: Achieves R² > 0.93 compared to a Linear Regression baseline.
- **Reproducible Validation**: Uses an 80/20 Train-Test split with fixed random seeds (`random_state=42`).
- **Data Pipeline**: Handles encoding issues (BOM/unicode), automated temporal feature extraction, and defensive checks against data corruption.

## Model Performance

XGBoost performance on the 20% held-out test set:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **R² Score** | 0.936 | The model explains 93.6% of the variance in rental demand. |
| **RMSE** | 163.69 | Average prediction error in number of bikes per hour. |
| **MAE** | 96.77 | Mean Absolute Error. |

*> Note: Performance is improved by incorporating non-linear interactions between temperature and seasonality.*

## Project Structure

```bash
Seoul-Bike-Prediction/
├── data/               # Raw data storage (CSV)
├── models/             # Serialized .joblib models
├── src/                # Source code
│   ├── data_loader.py       # Data ingestion & cleaning
│   ├── feature_engineering.py # Scikit-learn pipelines & transformers
│   ├── model_trainer.py     # Model factory (Linear/XGBoost)
│   └── config.py            # Centralized configuration
├── tests/              # Unit & integration tests
├── main.py             # Pipeline entry point
└── requirements.txt    # Project dependencies
```

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Jiaweisun274/Seoul-Bike-Prediction.git
cd Seoul-Bike-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline

To train the model and generate metrics:

```bash
python main.py
```

*Trained models will be saved to the `models/` directory.*

### 4. Run Tests

```bash
python -m pytest tests/
```

## Future Improvements

- [ ] SHAP analysis for model interpretability
- [ ] Time-series aware train/test split (chronological rather than random)
- [ ] Deploy as a REST API using FastAPI
- [ ] CI/CD workflows using GitHub Actions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
