"""
Model Training Module for Statistical Arbitrage Strategy

Implements rolling window training regime for ML models predicting short-term returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from loguru import logger


class ModelTrainer:
    """
    Manages ML model training with rolling window regime.
    
    Key features:
    - Multiple model types (linear, tree-based, gradient boosting)
    - Rolling window retraining
    - Feature scaling
    - Model persistence
    - Performance tracking
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        model_dir: str = './models',
        rolling_window_years: int = 10,
        retrain_frequency_days: int = 365
    ):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model ('ridge', 'random_forest', 'xgboost', 'lightgbm')
            model_dir: Directory for saving trained models
            rolling_window_years: Years of historical data for training
            retrain_frequency_days: How often to retrain the model
        """
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.rolling_window_years = rolling_window_years
        self.retrain_frequency_days = retrain_frequency_days
        
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = []
        
        logger.info(
            f"ModelTrainer initialized: {model_type}, "
            f"rolling window: {rolling_window_years} years, "
            f"retrain frequency: {retrain_frequency_days} days"
        )
    
    def _get_model(self, **kwargs) -> Any:
        """
        Initialize model based on type.
        
        Args:
            **kwargs: Model-specific hyperparameters
            
        Returns:
            Initialized model object
        """
        if self.model_type == 'ridge':
            return Ridge(alpha=kwargs.get('alpha', 1.0), random_state=42)
        
        elif self.model_type == 'lasso':
            return Lasso(alpha=kwargs.get('alpha', 0.1), random_state=42)
        
        elif self.model_type == 'elastic_net':
            return ElasticNet(
                alpha=kwargs.get('alpha', 0.1),
                l1_ratio=kwargs.get('l1_ratio', 0.5),
                random_state=42
            )
        
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 100),
                min_samples_leaf=kwargs.get('min_samples_leaf', 50),
                n_jobs=-1,
                random_state=42
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                subsample=kwargs.get('subsample', 0.8),
                random_state=42
            )
        
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **model_kwargs
    ) -> Dict[str, float]:
        """
        Train the model on provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **model_kwargs: Model-specific hyperparameters
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")
        
        # Initialize model
        self.model = self._get_model(**model_kwargs)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_scaled)
        train_metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred)
        }
        
        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            
            train_metrics.update({
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred)
            })
        
        logger.info(f"Training complete. Metrics: {train_metrics}")
        
        # Save to history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'metrics': train_metrics
        })
        
        return train_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using trained model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def rolling_window_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.DatetimeIndex,
        start_date: datetime,
        end_date: datetime,
        **model_kwargs
    ) -> List[Dict]:
        """
        Train model using rolling window approach.
        
        Args:
            X: Full feature matrix
            y: Full target vector
            dates: DatetimeIndex corresponding to X and y
            start_date: First date to generate predictions for
            end_date: Last date to generate predictions for
            **model_kwargs: Model hyperparameters
            
        Returns:
            List of training sessions with metadata
        """
        logger.info(
            f"Starting rolling window training from {start_date} to {end_date}"
        )
        
        training_sessions = []
        current_date = start_date
        
        while current_date <= end_date:
            # Define training window
            train_end = current_date
            train_start = train_end - timedelta(days=365 * self.rolling_window_years)
            
            # Select training data
            train_mask = (dates >= train_start) & (dates < train_end)
            X_train = X[train_mask]
            y_train = y[train_mask]
            
            if len(X_train) < 1000:  # Minimum samples threshold
                logger.warning(
                    f"Insufficient training data for {current_date} "
                    f"({len(X_train)} samples). Skipping."
                )
                current_date += timedelta(days=self.retrain_frequency_days)
                continue
            
            # Train model
            metrics = self.train(X_train, y_train, **model_kwargs)
            
            # Save model checkpoint
            model_filename = f"model_{current_date.strftime('%Y%m%d')}.pkl"
            self.save_model(model_filename)
            
            # Record session
            session = {
                'train_start': train_start.isoformat(),
                'train_end': train_end.isoformat(),
                'n_samples': len(X_train),
                'metrics': metrics,
                'model_file': model_filename
            }
            training_sessions.append(session)
            
            logger.info(
                f"Completed training session for {current_date.strftime('%Y-%m-%d')}"
            )
            
            # Move to next training date
            current_date += timedelta(days=self.retrain_frequency_days)
        
        logger.info(f"Rolling window training complete. {len(training_sessions)} sessions.")
        
        return training_sessions
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get importances based on model type
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            logger.warning(f"Model type {self.model_type} does not support feature importance")
            return pd.DataFrame()
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        **model_kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            X: Features
            y: Targets
            n_splits: Number of CV splits
            **model_kwargs: Model hyperparameters
            
        Returns:
            Dictionary of CV scores
        """
        logger.info(f"Performing {n_splits}-fold time series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            metrics = self.train(X_train, y_train, X_val, y_val, **model_kwargs)
            
            # Store validation metrics
            cv_scores['mse'].append(metrics['val_mse'])
            cv_scores['mae'].append(metrics['val_mae'])
            cv_scores['r2'].append(metrics['val_r2'])
            
            logger.info(
                f"Fold {fold}/{n_splits}: "
                f"MSE={metrics['val_mse']:.6f}, "
                f"MAE={metrics['val_mae']:.6f}, "
                f"R2={metrics['val_r2']:.6f}"
            )
        
        # Calculate average scores
        avg_scores = {
            metric: np.mean(scores) for metric, scores in cv_scores.items()
        }
        
        logger.info(f"CV complete. Average scores: {avg_scores}")
        
        return cv_scores
    
    def save_model(self, filename: str) -> None:
        """
        Save trained model and scaler to disk.
        
        Args:
            filename: Name of file to save to
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        filepath = self.model_dir / filename
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename: str) -> None:
        """
        Load trained model and scaler from disk.
        
        Args:
            filename: Name of file to load from
        """
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.training_history = model_data.get('training_history', [])
        
        logger.info(f"Model loaded from {filepath}")
    
    def save_training_history(self, filename: str = 'training_history.json') -> None:
        """
        Save training history to JSON file.
        
        Args:
            filename: Name of JSON file
        """
        filepath = self.model_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training history saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data_acquisition import DataAcquisitionEngine
    from feature_engineering import FeatureEngineer
    from datetime import datetime, timedelta
    
    # Get sample data
    engine = DataAcquisitionEngine()
    universe = engine.get_russell_3000_universe()[:20]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    
    df = engine.get_training_data(universe, start_date, end_date, apply_filters=False)
    
    # Calculate features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.calculate_all_features(df, target_horizons=[3])
    X, y = feature_engineer.prepare_ml_dataset(df_features)
    
    # Train model
    trainer = ModelTrainer(model_type='xgboost')
    
    # Split into train/test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train and evaluate
    metrics = trainer.train(X_train, y_train, X_test, y_test)
    print(f"\nTraining metrics:\n{json.dumps(metrics, indent=2)}")
    
    # Feature importance
    importance = trainer.get_feature_importance(X.columns.tolist())
    print(f"\nTop 10 features:\n{importance.head(10)}")
    
    # Save model
    trainer.save_model('example_model.pkl')
