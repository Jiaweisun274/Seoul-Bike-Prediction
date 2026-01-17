import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Module Imports
from src.config import CATEGORICAL_COLS, TEST_SIZE, RANDOM_STATE, XGB_PARAMS
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info(">>> STARTING ENGINEERING PIPELINE <<<")

    # 1. Data Loading & Cleaning
    loader = DataLoader()
    loader.load_data()
    df = loader.clean_and_feature_extract()
    X, y = loader.get_features_target()

    # 2. Train/Test Split
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Feature Engineering Pipeline
    # Identify numerical columns dynamically
    numerical_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
    
    engineer = FeatureEngineer()
    preprocessor = engineer.create_pipeline(CATEGORICAL_COLS, numerical_cols)
    
    # Fit-Transform on Train, Transform on Test (Prevent Data Leakage)
    logger.info("Applying feature transformations...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 4. Model Training (Here we choose XGBoost for maximum impact)
    trainer = ModelTrainer(model_type='xgboost', params=XGB_PARAMS)
    trainer.train(X_train_processed, y_train)

    # 5. Evaluation
    metrics = trainer.evaluate(X_test_processed, y_test)

    # 6. Save Artifacts
    trainer.save_model("xgboost_bike_predictor.joblib")
    
    logger.info(">>> PIPELINE COMPLETED SUCCESSFULLY <<<")

if __name__ == "__main__":
    run_pipeline()