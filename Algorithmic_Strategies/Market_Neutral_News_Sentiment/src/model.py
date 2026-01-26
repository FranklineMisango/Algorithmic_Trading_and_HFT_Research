import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from typing import Tuple

class SentimentModel:
    def __init__(self, alpha: float = 1.0, training_years: int = 10):
        self.alpha = alpha
        self.training_years = training_years
        self.model = Ridge(alpha=alpha)
        self.feature_cols = None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and target"""
        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns if c.startswith('topic_') or c == 'sentiment_zscore']
        X = df[self.feature_cols].values
        y = df['sector_relative_return'].values
        return X, y
    
    def train(self, train_df: pd.DataFrame):
        """Train linear regression model"""
        X, y = self.prepare_features(train_df)
        self.model.fit(X, y)
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict sector-relative returns"""
        X, _ = self.prepare_features(test_df)
        return self.model.predict(X)
    
    def walk_forward_validation(self, df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
        """Rolling 10-year training, annual retraining"""
        predictions = []
        
        for year in range(start_year, end_year + 1):
            train_start = year - self.training_years
            train_df = df[(df['year'] >= train_start) & (df['year'] < year)]
            test_df = df[df['year'] == year]
            
            if len(train_df) > 0 and len(test_df) > 0:
                self.train(train_df)
                test_df = test_df.copy()
                test_df['predicted_return'] = self.predict(test_df)
                predictions.append(test_df)
        
        return pd.concat(predictions, ignore_index=True)
