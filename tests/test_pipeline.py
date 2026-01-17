import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.config import DATA_FILE

class TestDataPipeline:
    """
    Engineering Standard: Automated tests to ensure data integrity
    before any expensive training begins.
    """
    
    def test_data_file_exists(self):
        assert os.path.exists(DATA_FILE), "Data file not found!"

    def test_data_loader_structure(self):
        """Test if data loads and has correct columns"""
        if not os.path.exists(DATA_FILE):
            pytest.skip("Data file missing, skipping integration test")
            
        dl = DataLoader()
        df = dl.load_data()
        assert not df.empty, "Dataframe is empty"
        assert 'Rented Bike Count' in df.columns, "Target column missing"

    def test_feature_generation(self):
        """Test if temporal features are created correctly"""
        if not os.path.exists(DATA_FILE):
            pytest.skip("Data file missing")
            
        dl = DataLoader()
        dl.load_data()
        df = dl.clean_and_feature_extract()
        
        assert 'Month' in df.columns
        assert 'IsWeekend' in df.columns
        assert df['Month'].dtype == int or np.int64