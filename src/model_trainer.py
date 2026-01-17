import joblib
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from typing import Dict, Any

from src.config import MODEL_DIR

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Factory class to train and evaluate different models.
    """
    
    def __init__(self, model_type: str = 'xgboost', params: Dict[str, Any] = None):
        self.model_type = model_type
        self.params = params if params else {}
        self.model = self._get_model()

    def _get_model(self):
        """Selects the model architecture."""
        if self.model_type == 'linear':
            logger.info("Initializing Linear Regression (Baseline)...")
            return LinearRegression()
        elif self.model_type == 'xgboost':
            logger.info("Initializing XGBoost Regressor (Advanced)...")
            return XGBRegressor(**self.params)
        else:
            raise ValueError(f"Model type {self.model_type} not supported.")

    def train(self, X_train, y_train):
        """Fits the model."""
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Returns performance metrics."""
        logger.info("Evaluating model performance...")
        y_pred = self.model.predict(X_test)
        
        # Ensure non-negative predictions for bike counts
        y_pred = np.maximum(y_pred, 0)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
        logger.info(f"Metrics: {metrics}")
        return metrics

    def save_model(self, filename: str):
        """Persists the model to disk."""
        path = MODEL_DIR / filename
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")