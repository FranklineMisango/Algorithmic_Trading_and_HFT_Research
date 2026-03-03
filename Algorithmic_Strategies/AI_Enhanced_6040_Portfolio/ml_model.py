"""
ML Model Module for AI-Enhanced 60/40 Portfolio

This module implements the decision tree regression model for
predicting optimal portfolio allocations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import pickle
import os

# Conditional imports for optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


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
        self.feature_selector = None
        self.best_params = {}  # Store best hyperparameters
        
        # Model parameters from config
        model_config = config['model']
        self.model_type = model_config['type']
        
        # Default parameters
        self.default_params = model_config.get('parameters', {})
        
        # Hyperparameter tuning
        self.hyperparameter_tuning = model_config.get('hyperparameter_tuning', {}).get('enabled', False)
        self.param_grid = model_config.get('hyperparameter_tuning', {}).get('param_grid', {})
        
        # Model parameters from config
        model_config = config['model']
        self.model_type = model_config['type']
        
        # Default parameters
        self.default_params = model_config.get('parameters', {})
        
        # Hyperparameter tuning
        self.hyperparameter_tuning = model_config.get('hyperparameter_tuning', {}).get('enabled', False)
        self.param_grid = model_config.get('hyperparameter_tuning', {}).get('param_grid', {})
        
        # Feature selection
        fs_config = model_config.get('feature_selection', {})
        self.fs_method = fs_config.get('method', 'SelectKBest')
        self.fs_k = fs_config.get('k', 25)
        self.fs_score_func = fs_config.get('score_func', 'f_regression')
    
    def create_model(self, model_type: str, params: Dict) -> object:
        """
        Create a model instance based on type.
        
        Args:
            model_type: Type of model to create
            params: Model parameters
            
        Returns:
            Model instance
        """
        if model_type == 'RandomForestRegressor':
            return RandomForestRegressor(**params)
        elif model_type == 'XGBoostRegressor':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            return xgb.XGBRegressor(**params)
        elif model_type == 'LightGBMRegressor':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            return lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def perform_hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best parameters
        """
        if not self.hyperparameter_tuning or not self.param_grid or len(X_train) < 5:
            return self.default_params
        
        print("  Performing hyperparameter tuning...")
        
        model = self.create_model(self.model_type, self.default_params)
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            model, 
            self.param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"    Best parameters: {grid_search.best_params_}")
        print(f"    Best CV score: {-grid_search.best_score_:.6f}")
        
        return grid_search.best_params_
        
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
                   asset_name: str) -> object:
        """
        Train a model for a specific asset with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            asset_name: Name of the asset
            
        Returns:
            Trained model
        """
        print(f"Training {self.model_type} for {asset_name}...")
        
        # Feature selection
        if self.feature_selector is None:
            if self.fs_score_func == 'f_regression':
                score_func = f_regression
            elif self.fs_score_func == 'mutual_info_regression':
                score_func = mutual_info_regression
            else:
                score_func = f_regression
            
            self.feature_selector = SelectKBest(score_func, k=min(self.fs_k, X_train.shape[1]))
            self.feature_selector.fit(X_train, y_train)
        
        # Apply feature selection
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_train)
            X_train_use = pd.DataFrame(X_selected, index=X_train.index)
            selected_features = X_train.columns[self.feature_selector.get_support()]
        else:
            X_train_use = X_train
            selected_features = X_train.columns
        
        # Hyperparameter tuning
        best_params = self.perform_hyperparameter_tuning(X_train_use, y_train)
        self.best_params[asset_name] = best_params
        
        # Adjust parameters for small datasets
        best_params = best_params.copy()
        min_split = best_params.get('min_samples_split', 2)
        if len(X_train_use) < min_split:
            best_params['min_samples_split'] = max(2, len(X_train_use) // 2)
        min_leaf = best_params.get('min_samples_leaf', 1)
        if len(X_train_use) < min_leaf:
            best_params['min_samples_leaf'] = max(1, len(X_train_use) // 2)
        
        # Create and train model
        model = self.create_model(self.model_type, best_params)
        model.fit(X_train_use, y_train)
        
        # Store model
        self.models[asset_name] = model
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[asset_name] = pd.Series(
                model.feature_importances_,
                index=selected_features
            ).sort_values(ascending=False)
        elif hasattr(model, 'coef_'):
            self.feature_importance[asset_name] = pd.Series(
                model.coef_,
                index=selected_features
            ).sort_values(ascending=False)
        
        # Training metrics
        train_pred = model.predict(X_train_use)
        train_mse = mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        self.training_history[asset_name] = {
            'train_mse': train_mse,
            'train_r2': train_r2,
            'best_params': best_params
        }
        
        print(f"  Training MSE: {train_mse:.6f}")
        print(f"  Training R²: {train_r2:.4f}")
        
        return model
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.DataFrame) -> Dict[str, RandomForestRegressor]:
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
        Evaluate model performance on test set with enhanced metrics.
        
        Args:
            X_test: Test features
            y_test: Test targets
            asset_name: Name of the asset
            
        Returns:
            Dictionary of evaluation metrics
        """
        model = self.models[asset_name]
        
        # Apply feature selection
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_test)
            X_test_use = pd.DataFrame(X_selected, index=X_test.index)
        else:
            X_test_use = X_test
        
        # Predictions
        y_pred = model.predict(X_test_use)
        
        # Basic metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy
        actual_direction = np.sign(y_test)
        pred_direction = np.sign(y_pred)
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        # Sharpe decomposition (simplified)
        risk_free_rate = self.config.get('portfolio', {}).get('risk_free_rate', 0.02) / 12  # Monthly
        excess_returns = y_test - risk_free_rate
        
        if excess_returns.std() > 0:
            sharpe_decomposition = {
                'return_contribution': excess_returns.mean(),
                'volatility_penalty': -excess_returns.std(),
                'sharpe_ratio': excess_returns.mean() / excess_returns.std()
            }
        else:
            sharpe_decomposition = {'sharpe_ratio': 0, 'return_contribution': 0, 'volatility_penalty': 0}
        
        # Statistical significance tests
        t_stat, p_value = stats.ttest_1samp(y_pred - y_test, 0)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'sharpe_decomposition': sharpe_decomposition,
            'prediction_bias': np.mean(y_pred - y_test),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
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
            # Apply feature selection if used
            if self.feature_selector is not None:
                X_selected = self.feature_selector.transform(X)
                X_use = pd.DataFrame(X_selected, index=X.index)
            else:
                X_use = X
            
            predictions[asset_name] = model.predict(X_use)
        
        predictions_df = pd.DataFrame(predictions, index=X.index)
        
        return predictions_df
    
    def calculate_optimal_allocations(self,
                                     predicted_returns: pd.DataFrame,
                                     historical_returns: pd.DataFrame = None,
                                     use_risk_parity: bool = True,
                                     regimes: pd.Series = None) -> pd.DataFrame:
        """
        Calculate optimal portfolio allocations with risk parity and regime adjustment.
        
        Args:
            predicted_returns: DataFrame with predicted returns
            historical_returns: Historical returns for volatility calculation
            use_risk_parity: Whether to use risk parity weighting
            regimes: Regime series for dynamic adjustments
            
        Returns:
            DataFrame with optimal allocations
        """
        allocations = pd.DataFrame(index=predicted_returns.index, 
                                  columns=predicted_returns.columns)
        
        # Calculate volatility
        if use_risk_parity and historical_returns is not None:
            vol_lookback = self.config.get('risk', {}).get('volatility_lookback', 12)
            rolling_vol = self.config.get('risk', {}).get('rolling_volatility', False)
            
            if rolling_vol:
                # Use rolling volatility - need to align indices properly
                vol = historical_returns.rolling(window=vol_lookback).std().shift(1)
                # Reindex to match predicted_returns index, using last available volatility
                vol = vol.reindex(predicted_returns.index, method='ffill')
            else:
                # Use fixed train volatility
                train_volatility = historical_returns.tail(vol_lookback).std()
                vol = pd.DataFrame(index=predicted_returns.index, columns=historical_returns.columns)
                for col in vol.columns:
                    vol[col] = train_volatility[col]
        else:
            vol = None
        
        for idx in predicted_returns.index:
            returns = predicted_returns.loc[idx]
            
            if use_risk_parity and historical_returns is not None and vol is not None:
                # Risk parity with rolling volatility
                current_vol = vol.loc[idx] if isinstance(vol, pd.DataFrame) else vol
                risk_contribution = 1 / (current_vol + 1e-6)
                
                # Apply return signal direction
                weights = risk_contribution * np.sign(returns) * np.abs(returns)
                
                # Handle all negative case
                if weights.sum() <= 0:
                    weights = risk_contribution / risk_contribution.sum()
                else:
                    weights = weights.clip(lower=0)
                    weights = weights / weights.sum()
            else:
                # Fallback to softmax
                scaled_returns = returns * 0.2
                exp_returns = np.exp(scaled_returns - scaled_returns.max())
                weights = exp_returns / exp_returns.sum()
            
            # Apply regime adjustments
            if regimes is not None and idx in regimes.index:
                regime = regimes.loc[idx]
                weights = self.apply_regime_adjustments(weights, regime)
            
            # Apply constraints
            if use_risk_parity and historical_returns is not None:
                # Calculate rolling correlation
                train_length = len(historical_returns) - len(predicted_returns)
                train_returns = historical_returns.iloc[:train_length]
                corr_matrix = train_returns.tail(12).corr()
                current_vol = vol.loc[idx] if isinstance(vol, pd.DataFrame) else vol
                weights = self.apply_allocation_constraints(weights, corr_matrix, current_vol)
            else:
                weights = self.apply_allocation_constraints(weights)
            
            allocations.loc[idx] = weights
        
        # Clean allocations
        allocations = allocations.fillna(0).replace([np.inf, -np.inf], 0)
        # Ensure weights sum to 1
        allocations = allocations.div(allocations.sum(axis=1).replace(0, 1), axis=0)
        # Ensure float dtype
        allocations = allocations.astype(float)
        
        return allocations
    
    def apply_allocation_constraints(self, weights: pd.Series, correlations: pd.DataFrame = None, 
                                    volatilities: pd.Series = None) -> pd.Series:
        """
        Apply allocation constraints with correlation penalty and dynamic vol-based caps.
        
        Args:
            weights: Proposed asset weights
            correlations: Correlation matrix (optional)
            volatilities: Asset volatilities (optional)
            
        Returns:
            Constrained weights
        """
        constrained = weights.copy()
        
        # Dynamic max allocation based on volatility
        if volatilities is not None:
            base_vol = volatilities.get('SPY', volatilities.median()) if 'SPY' in volatilities.index else volatilities.median()
            
            for asset in self.config['assets']['alternative']:
                ticker = asset['ticker']
                base_max = asset.get('max_allocation', 1.0)
                
                if ticker in constrained.index and ticker in volatilities.index:
                    vol_ratio = base_vol / (volatilities[ticker] + 1e-6)
                    dynamic_max = min(base_max, base_max * vol_ratio)
                    
                    if constrained[ticker] > dynamic_max:
                        constrained[ticker] = dynamic_max
        else:
            # Static constraints
            for asset in self.config['assets']['alternative']:
                ticker = asset['ticker']
                max_alloc = asset.get('max_allocation', 1.0)
                
                if ticker in constrained.index:
                    if constrained[ticker] > max_alloc:
                        constrained[ticker] = max_alloc
        
        # Correlation penalty: reduce allocation to highly correlated assets
        if correlations is not None and 'SPY' in constrained.index and 'TLT' in constrained.index:
            spy_tlt_corr = correlations.loc['SPY', 'TLT'] if 'SPY' in correlations.index else 0
            
            # If stock-bond correlation > 0.5, increase alternatives
            if spy_tlt_corr > 0.5:
                # Reduce SPY and TLT by 20%
                if 'SPY' in constrained.index:
                    constrained['SPY'] *= 0.8
                if 'TLT' in constrained.index:
                    constrained['TLT'] *= 0.8
                
                # Boost alternatives
                for alt in ['BTC-USD', 'GLD']:
                    if alt in constrained.index:
                        constrained[alt] *= 1.2
        
        # Normalize to sum to 1
        if constrained.sum() > 0:
            constrained = constrained / constrained.sum()
        
        return constrained
    
    def apply_regime_adjustments(self, weights: pd.Series, regime: int) -> pd.Series:
        """
        Apply regime-based adjustments to portfolio weights.
        
        Args:
            weights: Current portfolio weights
            regime: Current market regime (0=defensive, 1=neutral, 2=aggressive)
            
        Returns:
            Adjusted weights
        """
        adjusted = weights.copy()
        regime_config = self.config.get('risk', {}).get('regime_adjustments', {})
        
        if regime == 0:  # Defensive
            adjustments = regime_config.get('defensive', {})
        elif regime == 1:  # Neutral
            adjustments = regime_config.get('neutral', {})
        elif regime == 2:  # Aggressive
            adjustments = regime_config.get('aggressive', {})
        else:
            return adjusted
        
        # Apply equity limits
        if 'SPY' in adjusted.index and 'equity_max' in adjustments:
            if adjusted['SPY'] > adjustments['equity_max']:
                adjusted['SPY'] = adjustments['equity_max']
        
        # Apply bond minimums
        if 'TLT' in adjusted.index and 'bond_min' in adjustments:
            if adjusted['TLT'] < adjustments['bond_min']:
                adjusted['TLT'] = adjustments['bond_min']
        
        # Apply gold minimums
        if 'GLD' in adjusted.index and 'gold_min' in adjustments:
            if adjusted['GLD'] < adjustments['gold_min']:
                adjusted['GLD'] = adjustments['gold_min']
        
        # Renormalize
        if adjusted.sum() > 0:
            adjusted = adjusted / adjusted.sum()
        
        return adjusted
    
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
            model = RandomForestRegressor(**self.model_params)
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
