"""
Model Training for Music Royalties Strategy
Trains regression model to predict fair price multipliers
Model 3: PredictedMultiplier = β₀ + β₁(StabilityRatio) + β₂(CatalogAge) + ε
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Tuple, List
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoyaltyPriceModel:
    """
    Model for predicting fair price multipliers for music royalty assets
    """
    
    def __init__(self, config: Dict):
        """
        Initialize model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = config['model']['features']
        self.target_name = config['model']['target']
        self.is_fitted = False
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              model_type: str = 'linear') -> None:
        """
        Train the price multiplier prediction model
        
        Args:
            X_train: Training features
            y_train: Training target (price_multiplier)
            model_type: Type of model ('linear', 'ridge', 'lasso')
        """
        logger.info(f"Training {model_type} regression model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Select model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=0.1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Log coefficients
        self._log_model_coefficients()
        
        logger.info("Model training complete")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict price multipliers
        
        Args:
            X: Features
            
        Returns:
            Predicted price multipliers
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X: Features
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'mean_prediction': predictions.mean(),
            'std_prediction': predictions.std()
        }
        
        # Residual analysis
        residuals = y - predictions
        metrics['mean_residual'] = residuals.mean()
        metrics['std_residual'] = residuals.std()
        
        return metrics
    
    def _log_model_coefficients(self) -> None:
        """
        Log model coefficients for interpretation
        """
        if hasattr(self.model, 'coef_'):
            logger.info("\n=== Model Coefficients ===")
            logger.info(f"Intercept: {self.model.intercept_:.4f}")
            for feature, coef in zip(self.feature_names, self.model.coef_):
                logger.info(f"{feature}: {coef:.4f}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on absolute coefficient values
        
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.model, 'coef_'):
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        })
        importance = importance.sort_values('abs_coefficient', ascending=False)
        
        return importance
    
    def calculate_mispricing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mispricing: predicted fair value - observed market multiplier
        
        Positive mispricing = undervalued (model predicts higher than market)
        Negative mispricing = overvalued (model predicts lower than market)
        
        Args:
            df: DataFrame with features and observed price_multiplier
            
        Returns:
            DataFrame with mispricing column
        """
        df = df.copy()
        
        X = df[self.feature_names]
        df['predicted_multiplier'] = self.predict(X)
        df['mispricing'] = df['predicted_multiplier'] - df['price_multiplier']
        df['mispricing_pct'] = (df['mispricing'] / df['price_multiplier']) * 100
        
        return df
    
    def save(self, filepath: str) -> None:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RoyaltyPriceModel':
        """
        Load trained model from disk
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.target_name = model_data['target_name']
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance


class ModelValidator:
    """
    Validates model performance and robustness
    """
    
    def __init__(self, config: Dict):
        """
        Initialize validator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_mse = config['model']['target_mse']
    
    def validate_performance(self, model: RoyaltyPriceModel,
                           X_val: pd.DataFrame, y_val: pd.Series) -> bool:
        """
        Validate that model meets performance targets
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            True if validation passes
        """
        metrics = model.evaluate(X_val, y_val)
        
        logger.info("\n=== Model Validation ===")
        logger.info(f"MSE: {metrics['mse']:.4f} (target: {self.target_mse})")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        
        # Check if meets target
        if metrics['mse'] <= self.target_mse:
            logger.info("✓ Model meets MSE target")
            return True
        else:
            logger.warning(f"✗ Model MSE ({metrics['mse']:.2f}) exceeds target ({self.target_mse})")
            return False
    
    def cross_validate(self, train_df: pd.DataFrame,
                      feature_names: List[str], target_name: str,
                      n_folds: int = 5) -> Dict:
        """
        Perform time-series cross-validation
        
        Args:
            train_df: Training data
            feature_names: List of feature columns
            target_name: Target column name
            n_folds: Number of folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {n_folds}-fold time-series cross-validation...")
        
        # Sort by date
        train_df = train_df.sort_values('transaction_date')
        
        fold_size = len(train_df) // n_folds
        
        cv_scores = []
        
        for i in range(1, n_folds + 1):
            # Use data up to fold i for training
            train_idx = i * fold_size
            X_train = train_df.iloc[:train_idx][feature_names]
            y_train = train_df.iloc[:train_idx][target_name]
            
            # Use next fold for validation
            if i < n_folds:
                val_start = train_idx
                val_end = (i + 1) * fold_size
                X_val = train_df.iloc[val_start:val_end][feature_names]
                y_val = train_df.iloc[val_start:val_end][target_name]
                
                # Train and evaluate
                model = RoyaltyPriceModel(self.config)
                model.train(X_train, y_train)
                metrics = model.evaluate(X_val, y_val)
                
                cv_scores.append(metrics['mse'])
                logger.info(f"Fold {i}: MSE = {metrics['mse']:.4f}")
        
        cv_results = {
            'mean_mse': np.mean(cv_scores),
            'std_mse': np.std(cv_scores),
            'fold_scores': cv_scores
        }
        
        logger.info(f"\nCV Results: Mean MSE = {cv_results['mean_mse']:.4f} ± {cv_results['std_mse']:.4f}")
        
        return cv_results
    
    def analyze_residuals(self, model: RoyaltyPriceModel,
                         X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Analyze model residuals for patterns
        
        Args:
            model: Trained model
            X: Features
            y: True targets
            
        Returns:
            DataFrame with residual analysis
        """
        predictions = model.predict(X)
        residuals = y - predictions
        
        residual_df = pd.DataFrame({
            'true': y,
            'predicted': predictions,
            'residual': residuals,
            'abs_residual': np.abs(residuals),
            'pct_error': (residuals / y) * 100
        })
        
        # Add features for analysis
        for col in X.columns:
            residual_df[col] = X[col].values
        
        logger.info(f"Mean absolute error: {residual_df['abs_residual'].mean():.4f}")
        logger.info(f"Mean percentage error: {residual_df['pct_error'].mean():.2f}%")
        
        return residual_df


def train_and_validate_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            config: Dict) -> Tuple[RoyaltyPriceModel, Dict]:
    """
    Convenience function to train and validate model
    
    Args:
        train_df: Training data with features
        val_df: Validation data with features
        config: Configuration dictionary
        
    Returns:
        Trained model and validation metrics
    """
    feature_names = config['model']['features']
    target_name = config['model']['target']
    
    # Prepare data
    X_train = train_df[feature_names]
    y_train = train_df[target_name]
    X_val = val_df[feature_names]
    y_val = val_df[target_name]
    
    # Train model
    model = RoyaltyPriceModel(config)
    model.train(X_train, y_train)
    
    # Validate
    validator = ModelValidator(config)
    val_metrics = model.evaluate(X_val, y_val)
    is_valid = validator.validate_performance(model, X_val, y_val)
    
    # Feature importance
    importance = model.get_feature_importance()
    logger.info("\n=== Feature Importance ===")
    logger.info(f"\n{importance}")
    
    return model, val_metrics


if __name__ == '__main__':
    import yaml
    from data_loader import load_and_prepare_data
    from feature_engineering import engineer_all_features
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and prepare data
    data_splits = load_and_prepare_data(config)
    
    # Engineer features
    train_df = engineer_all_features(data_splits['train'], config)
    val_df = engineer_all_features(data_splits['validation'], config)
    
    # Train and validate
    model, metrics = train_and_validate_model(train_df, val_df, config)
    
    print("\n=== Model Training Complete ===")
    print(f"Validation MSE: {metrics['mse']:.4f}")
    print(f"Validation R²: {metrics['r2']:.4f}")
