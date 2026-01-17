# 🚲 Seoul Bike Sharing Demand Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

## 📌 Project Overview

This project engineers a robust, production-ready machine learning pipeline to predict hourly bike rental demand in Seoul, South Korea. Unlike standard analysis notebooks, this repository demonstrates a **full-cycle engineering approach**, featuring modular architecture, automated testing, and advanced ensemble modeling.

By accurately predicting demand based on weather conditions (Temperature, Humidity, Wind Speed) and temporal features, this system helps bike-sharing operators optimize fleet distribution and reduce operational costs.

## 🚀 Key Technical Highlights

- **Production-Grade Architecture**: Code is modularized into `data_loader`, `feature_engineering`, and `model_trainer` for scalability and maintainability.
- **Advanced Modeling**: Implements **XGBoost Regressor**, achieving superior performance (R² > 0.93) compared to traditional Linear Regression baselines.
- **Rigorous Validation**: Uses a strict **80/20 Train-Test split** with fixed random seeds (`random_state=42`) to ensure reproducibility and prevent data leakage.
- **Robust Data Pipeline**: Handles complex encoding issues (BOM/unicode), automated temporal feature extraction, and defensive programming against data corruption.

## 📊 Model Performance

The current production model (XGBoost) achieves the following metrics on the **20% held-out test set** (unseen data):

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **R² Score** | **0.936** | The model explains **93.6%** of the variance in rental demand. |
| **RMSE** | **163.69** | Average prediction error in number of bikes per hour. |
| **MAE** | **96.77** | Mean Absolute Error. |

*> Note: Performance is significantly improved by incorporating non-linear interactions between temperature and seasonality.*

## 🛠️ Project Structure

```bash
Seoul-Bike-Prediction/
├── data/               # Raw data storage (CSV)
├── models/             # Serialized .joblib models for deployment
├── src/                # Source code
│   ├── data_loader.py       # Robust data ingestion & cleaning
│   ├── feature_engineering.py # Scikit-learn pipelines & transformers
│   ├── model_trainer.py     # Model factory (Linear/XGBoost)
│   └── config.py            # Centralized configuration
├── tests/              # Automated unit & integration tests
├── main.py             # Pipeline entry point
└── requirements.txt    # Project dependencies
```

## 💻 Installation & Usage

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

*Artifacts (trained models) will be saved to the `models/` directory.*

### 4. Run Tests (Engineering Standard)

To verify data integrity and logic:

```bash
python -m pytest tests/
```

## 📈 Future Improvements

- [ ] Deploy the model as a REST API using **FastAPI**.
- [ ] Implement **SHAP** (SHapley Additive exPlanations) for model interpretability.
- [ ] Add CI/CD workflows using GitHub Actions.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.