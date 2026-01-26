"""
Machine learning models for Foreign Market Lead-Lag ML Strategy.
Implements Lasso, Random Forest, Gradient Boosting, and Neural Networks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

logger = logging.getLogger(__name__)


class MLModels:
    """Machine learning model training and prediction."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_type = config['models']['primary_model']
        self.models = {}
        
    def create_model(self, model_type: Optional[str] = None):
        """Create ML model based on configuration."""
        if model_type is None:
            model_type = self.model_type
        
        if model_type == 'lasso':
            params = self.config['models']['lasso']
            return Lasso(**params)
        
        elif model_type == 'random_forest':
            params = self.config['models']['random_forest']
            return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        
        elif model_type == 'gradient_boosting':
            params = self.config['models']['gradient_boosting']
            return GradientBoostingRegressor(**params, random_state=42)
        
        elif model_type == 'neural_network':
            params = self.config['models']['neural_network']
            hidden_layers = tuple(params['hidden_layers'])
            return MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=params['activation'],
                max_iter=params['epochs'],
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series,
                                train_years: int = 5, test_years: int = 1) -> Dict:
        """
        Perform walk-forward out-of-sample validation.
        
        Args:
            X: Features
            y: Target
            train_years: Years of training data
            test_years: Years of test data
            
        Returns:
            Dictionary with predictions, actuals, and metrics
        """
        logger.info(f"Starting walk-forward validation (train={train_years}y, test={test_years}y)")
        
        # Convert years to trading days
        train_days = train_years * 252
        test_days = test_years * 252
        
        predictions = []
        actuals = []
        dates = []
        
        start_idx = train_days
        
        while start_idx + test_days <= len(X):
            # Split data
            train_start = start_idx - train_days
            train_end = start_idx
            test_start = start_idx
            test_end = start_idx + test_days
            
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Train model
            model = self.create_model()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Store results
            predictions.extend(y_pred)
            actuals.extend(y_test.values)
            dates.extend(y_test.index)
            
            # Move window forward
            start_idx += test_days
            
            logger.info(f"Completed fold: train {X_train.index[0]} to {X_train.index[-1]}, "
                       f"test {X_test.index[0]} to {X_test.index[-1]}")
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        r2_oos = r2_score(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        ic = np.corrcoef(predictions, actuals)[0, 1]
        
        results = {
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates,
            'r2_oos': r2_oos,
            'rmse': rmse,
            'ic': ic
        }
        
        logger.info(f"Walk-forward validation complete: R²_OOS={r2_oos:.4f}, IC={ic:.4f}")
        
        return results
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series, stock_ticker: str):
        """Train final model on all available data."""
        logger.info(f"Training final model for {stock_ticker}...")
        
        model = self.create_model()
        model.fit(X, y)
        
        # Store model
        self.models[stock_ticker] = model
        
        # Calculate in-sample metrics
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        logger.info(f"Model trained for {stock_ticker}: R²={r2:.4f}")
        
        return model
    
    def predict(self, X: pd.DataFrame, stock_ticker: str) -> np.ndarray:
        """Make predictions using trained model."""
        if stock_ticker not in self.models:
            raise ValueError(f"No model found for {stock_ticker}")
        
        return self.models[stock_ticker].predict(X)
    
    def save_model(self, stock_ticker: str, filepath: str):
        """Save trained model to disk."""
        if stock_ticker not in self.models:
            raise ValueError(f"No model found for {stock_ticker}")
        
        joblib.dump(self.models[stock_ticker], filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, stock_ticker: str, filepath: str):
        """Load trained model from disk."""
        self.models[stock_ticker] = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, stock_ticker: str) -> pd.Series:
        """Get feature importance for tree-based models or coefficients for Lasso."""
        if stock_ticker not in self.models:
            raise ValueError(f"No model found for {stock_ticker}")
        
        model = self.models[stock_ticker]
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            return pd.Series(model.feature_importances_)
        elif hasattr(model, 'coef_'):
            # Linear models
            return pd.Series(model.coef_)
        else:
            logger.warning(f"Model type does not support feature importance")
            return pd.Series()


class MultiStockPredictor:
    """Manages predictions for multiple stocks."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ml_models = MLModels(config)
        self.stock_models = {}
        
    def train_all_stocks(self, stock_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                        validate: bool = True) -> Dict[str, Dict]:
        """
        Train models for all stocks with optional validation.
        
        Args:
            stock_data: Dictionary mapping stock ticker to (X, y) tuple
            validate: Whether to perform walk-forward validation
            
        Returns:
            Dictionary of validation results per stock
        """
        logger.info(f"Training models for {len(stock_data)} stocks...")
        
        results = {}
        
        for stock, (X, y) in stock_data.items():
            try:
                if validate:
                    # Perform walk-forward validation
                    val_results = self.ml_models.walk_forward_validation(X, y)
                    results[stock] = val_results
                
                # Train final model
                model = self.ml_models.train_final_model(X, y, stock)
                self.stock_models[stock] = model
                
            except Exception as e:
                logger.error(f"Error training model for {stock}: {e}")
                continue
        
        logger.info(f"Successfully trained {len(self.stock_models)} models")
        
        return results
    
    def predict_all_stocks(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate predictions for all stocks.
        
        Args:
            features: Feature DataFrame (single row or multiple rows)
            
        Returns:
            Series of predictions indexed by stock ticker
        """
        predictions = {}
        
        for stock, model in self.stock_models.items():
            try:
                pred = model.predict(features)
                predictions[stock] = pred[0] if len(pred) == 1 else pred
            except Exception as e:
                logger.error(f"Error predicting for {stock}: {e}")
                continue
        
        return pd.Series(predictions)


if __name__ == "__main__":
    import yaml
    from feature_engineering import FeatureEngineering
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    sp500_returns = pd.read_csv('data/sp500_daily_returns.csv', index_col=0, parse_dates=True)
    foreign_returns = pd.read_csv('data/foreign_weekly_returns.csv', index_col=0, parse_dates=True)
    
    # Prepare features
    feature_eng = FeatureEngineering(config)
    test_stock = sp500_returns.columns[0]
    X, y = feature_eng.prepare_training_data(foreign_returns, sp500_returns, test_stock)
    
    # Train and validate model
    ml_models = MLModels(config)
    results = ml_models.walk_forward_validation(X, y)
    
    print(f"\nValidation Results for {test_stock}:")
    print(f"R²_OOS: {results['r2_oos']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"IC: {results['ic']:.4f}")
