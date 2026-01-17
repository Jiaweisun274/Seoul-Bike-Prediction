import pandas as pd
import logging
from typing import Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_FILE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles data loading, cleaning, and initial temporal feature extraction.
    """
    
    def __init__(self, filepath: str = str(DATA_FILE)):
        self.filepath = filepath
        self.df = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from CSV handling specific encoding and BOM issues.
        """
        try:
            logger.info(f"Loading data from {self.filepath}...")
            self.df = pd.read_csv(self.filepath, encoding='unicode_escape')

            self.df.columns = [c.replace('ï»¿', '').strip() for c in self.df.columns]

            if 'Date' not in self.df.columns:
                for col in self.df.columns:
                    if 'Date' in col:
                        logger.warning(f"Renaming corrupted column '{col}' to 'Date'")
                        self.df.rename(columns={col: 'Date'}, inplace=True)
                        break

            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found at {self.filepath}. Please check the 'data/' directory.")
            raise

    def clean_and_feature_extract(self) -> pd.DataFrame:
        """
        Parses dates, cleans column names, and creates temporal features (Month, Day, Weekend).
        """
        if self.df.empty:
            self.load_data()

        logger.info("Preprocessing dates and extracting temporal features...")
        
        # 1. Date Conversion
        try:
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
        except KeyError:
            logger.error(f"Columns found: {self.df.columns.tolist()}")
            raise KeyError("Column 'Date' not found even after cleaning!")
        
        # 2. Feature Extraction
        self.df['Month'] = self.df['Date'].dt.month
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['IsWeekend'] = self.df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # 3. Column Name Cleaning (Removing units like (C), (%), etc.)
        self.df.columns = [c.split('(')[0].strip() for c in self.df.columns]
        
        logger.info("Data cleaning complete.")
        return self.df

    def get_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Splits the dataframe into X (features) and y (target).
        """
        target_col = 'Rented Bike Count'
        
        if target_col not in self.df.columns:
            potential = [c for c in self.df.columns if c.startswith('Rented Bike')]
            if potential:
                target_col = potential[0]
            else:
                raise ValueError(f"Target column '{target_col}' not found.")

        drop_cols = [target_col]
        if 'Date' in self.df.columns:
            drop_cols.append('Date')
            
        X = self.df.drop(columns=drop_cols)
        y = self.df[target_col]
        
        return X, y

if __name__ == "__main__":
    dl = DataLoader()
    dl.load_data()
    print("Columns after load:", dl.df.columns.tolist())
    dl.clean_and_feature_extract()
    print("Columns after clean:", dl.df.columns.tolist())