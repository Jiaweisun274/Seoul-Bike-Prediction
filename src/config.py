from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "SeoulBikeData.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

CATEGORICAL_COLS = ['Seasons', 'Holiday', 'Functioning Day']
TARGET_COL = 'Rented Bike Count'
DROP_COLS = ['Date']

XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE
}