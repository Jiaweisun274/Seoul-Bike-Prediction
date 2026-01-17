import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Manages advanced feature transformations including OneHotEncoding
    and Polynomial features.
    """
    
    def __init__(self):
        # We will separate categorical and numerical columns processing
        self.preprocessor = None

    def create_pipeline(self, categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
        """
        Creates a Scikit-learn ColumnTransformer pipeline.
        
        Args:
            categorical_cols: List of column names for OHE.
            numerical_cols: List of column names for Scaling/Poly.
        """
        logger.info("Building Feature Engineering Pipeline...")

        # 1. Categorical Pipeline
        cat_pipeline = Pipeline([
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])

        # 2. Numerical Pipeline (Scaling is crucial for Linear Models, helpful for others)
        num_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

        # Combine them
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ])
        
        return self.preprocessor

    def add_polynomials(self, X_train: np.array, X_test: np.array, degree: int = 2) -> Tuple[np.array, np.array]:
        """
        Adds polynomial features to capture non-linear relationships.
        Note: This increases dimensionality significantly.
        """
        logger.info(f"Generating Polynomial Features (Degree={degree})...")
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        return X_train_poly, X_test_poly