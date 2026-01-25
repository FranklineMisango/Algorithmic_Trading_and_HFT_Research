"""
ML Model Module for AI-Enhanced 60/40 Portfolio

This module implements the decision tree regression model for
predicting optimal portfolio allocations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os


class PortfolioMLModel:
    """Machine learning model for portfolio allocation prediction."""
    
    def __init__(self, config: Dict):
        """
        Initialize the ML model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}  # One model per asset
        self.feature_importance = {}
        self.training_history = {}
        
        # Model parameters from config
        model_params = config['model']['parameters']
        self.model_params = {
            'max_depth': model_params['max_depth'],
            'min_samples_split': model_params['min_samples_split'],
            'min_samples_leaf': model_params['min_samples_leaf'],
            'random_state': model_params['random_state']
        }
        
    def create_target_variables(self, 
                                returns: pd.DataFrame,
                                lookback: int = 1) -> pd.DataFrame:
        """
        Create target variables (future returns) for each asset.
        
        Args:
            returns: DataFrame with asset returns
            lookback: Number of periods to look forward
            
        Returns:
            DataFrame with future returns as targets
        """
        # Shift returns backward to create future returns
        targets = returns.shift(-lookback)
        targets = targets.dropna()
        
        return targets
    
    def prepare_train_test_data(self,
                               features: pd.DataFrame,
                               targets: pd.DataFrame,
                               test_size: float = 0.2) -> Tuple:
        """
        Prepare training and testing datasets.
        
        Args:
            features: DataFrame with features
            targets: DataFrame with target variables
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Align features and targets
        common_idx = features.index.intersection(targets.index)
        X = features.loc[common_idx]
        y = targets.loc[common_idx]
        
        # Time series split (no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   asset_name: str) -> DecisionTreeRegressor:
        """
        Train a decision tree model for a specific asset.
        
        Args:
            X_train: Training features
            y_train: Training targets
            asset_name: Name of the asset
            
        Returns:
            Trained model
        """
        print(f"Training model for {asset_name}...")
        
        # Create and train model
        model = DecisionTreeRegressor(**self.model_params)
        model.fit(X_train, y_train)
        
        # Store model
        self.models[asset_name] = model
        
        # Store feature importance
        self.feature_importance[asset_name] = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        # Training metrics
        train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        self.training_history[asset_name] = {
            'train_mse': train_mse,
            'train_r2': train_r2
        }
        
        print(f"  Training MSE: {train_mse:.6f}")
        print(f"  Training R²: {train_r2:.4f}")
        
        return model
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.DataFrame) -> Dict[str, DecisionTreeRegressor]:
        """
        Train models for all assets.
        
        Args:
            X_train: Training features
            y_train: Training targets (one column per asset)
            
        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*50)
        print("Training Models for All Assets")
        print("="*50)
        
        for asset in y_train.columns:
            self.train_model(X_train, y_train[asset], asset)
        
        print(f"\nTrained {len(self.models)} models successfully!")
        
        return self.models
    
    def evaluate_model(self,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      asset_name: str) -> Dict:
        """
        Evaluate model performance on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            asset_name: Name of the asset
            
        Returns:
            Dictionary of evaluation metrics
        """
        model = self.models[asset_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy
        actual_direction = np.sign(y_test)
        pred_direction = np.sign(y_pred)
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
        
        return metrics
    
    def evaluate_all_models(self,
                           X_test: pd.DataFrame,
                           y_test: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all models on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with evaluation metrics for all assets
        """
        print("\n" + "="*50)
        print("Evaluating Models on Test Set")
        print("="*50)
        
        results = {}
        
        for asset in y_test.columns:
            metrics = self.evaluate_model(X_test, y_test[asset], asset)
            results[asset] = metrics
            
            print(f"\n{asset}:")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        
        results_df = pd.DataFrame(results).T
        
        print("\n" + "="*50)
        print("Evaluation Complete!")
        print("="*50)
        
        return results_df
    
    def predict_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict returns for all assets.
        
        Args:
            X: Features for prediction
            
        Returns:
            DataFrame with predicted returns for each asset
        """
        predictions = {}
        
        for asset_name, model in self.models.items():
            predictions[asset_name] = model.predict(X)
        
        predictions_df = pd.DataFrame(predictions, index=X.index)
        
        return predictions_df
    
    def calculate_optimal_allocations(self,
                                     predicted_returns: pd.DataFrame,
                                     risk_aversion: float = 1.0) -> pd.DataFrame:
        """
        Calculate optimal portfolio allocations based on predicted returns.
        
        Uses a simple mean-variance optimization approach.
        
        Args:
            predicted_returns: DataFrame with predicted returns
            risk_aversion: Risk aversion parameter (higher = more conservative)
            
        Returns:
            DataFrame with optimal allocations
        """
        allocations = pd.DataFrame(index=predicted_returns.index, 
                                  columns=predicted_returns.columns)
        
        for idx in predicted_returns.index:
            returns = predicted_returns.loc[idx]
            
            # Simple allocation based on predicted returns
            # Positive returns get higher allocations
            positive_returns = returns.clip(lower=0)
            
            if positive_returns.sum() > 0:
                weights = positive_returns / positive_returns.sum()
            else:
                # If all negative, use equal weights
                weights = pd.Series(1.0 / len(returns), index=returns.index)
            
            # Apply constraints from config
            weights = self.apply_allocation_constraints(weights)
            
            allocations.loc[idx] = weights
        
        return allocations
    
    def apply_allocation_constraints(self, weights: pd.Series) -> pd.Series:
        """
        Apply allocation constraints based on config.
        
        Args:
            weights: Proposed asset weights
            
        Returns:
            Constrained weights
        """
        constrained = weights.copy()
        
        # Apply max allocation constraints for alternative assets
        for asset in self.config['assets']['alternative']:
            ticker = asset['ticker']
            max_alloc = asset.get('max_allocation', 1.0)
            
            if ticker in constrained.index:
                if constrained[ticker] > max_alloc:
                    constrained[ticker] = max_alloc
        
        # Normalize to sum to 1
        if constrained.sum() > 0:
            constrained = constrained / constrained.sum()
        
        return constrained
    
    def cross_validate(self, X: pd.DataFrame, y: pd.DataFrame, cv: int = 5) -> Dict:
        """
        Perform time series cross-validation.
        
        Args:
            X: Features
            y: Targets
            cv: Number of folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        tscv = TimeSeriesSplit(n_splits=cv)
        cv_scores = {}
        
        for asset in y.columns:
            model = DecisionTreeRegressor(**self.model_params)
            scores = cross_val_score(
                model, X, y[asset], 
                cv=tscv, 
                scoring='neg_mean_squared_error'
            )
            
            cv_scores[asset] = {
                'mean_mse': -scores.mean(),
                'std_mse': scores.std()
            }
        
        return cv_scores
    
    def save_models(self, output_dir: str = 'models'):
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for asset_name, model in self.models.items():
            filename = os.path.join(output_dir, f'{asset_name}_model.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"\nSaved {len(self.models)} models to {output_dir}/")
    
    def load_models(self, output_dir: str = 'models'):
        """
        Load trained models from disk.
        
        Args:
            output_dir: Directory containing saved models
        """
        for filename in os.listdir(output_dir):
            if filename.endswith('_model.pkl'):
                asset_name = filename.replace('_model.pkl', '')
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'rb') as f:
                    self.models[asset_name] = pickle.load(f)
        
        print(f"\nLoaded {len(self.models)} models from {output_dir}/")


if __name__ == "__main__":
    # Test the module
    import yaml
    from data_acquisition import DataAcquisition
    from feature_engineering import FeatureEngineer
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Fetch data
    print("Fetching data...")
    data_acq = DataAcquisition(config)
    prices, returns, indicators = data_acq.get_full_dataset()
    
    # Engineer features
    print("\nEngineering features...")
    feature_eng = FeatureEngineer(config)
    features = feature_eng.engineer_all_features(indicators)
    features_prepared = feature_eng.prepare_features_for_training(features)
    
    # Create ML model
    print("\nCreating ML model...")
    ml_model = PortfolioMLModel(config)
    
    # Create targets
    targets = ml_model.create_target_variables(returns, lookback=1)
    
    # Prepare train/test data
    X_train, X_test, y_train, y_test = ml_model.prepare_train_test_data(
        features_prepared, targets
    )
    
    # Train models
    ml_model.train_all_models(X_train, y_train)
    
    # Evaluate models
    results = ml_model.evaluate_all_models(X_test, y_test)
    
    print("\n" + "="*50)
    print("ML Model Testing Complete!")
    print("="*50)
