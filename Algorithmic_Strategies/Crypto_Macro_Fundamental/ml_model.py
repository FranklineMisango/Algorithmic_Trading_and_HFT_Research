"""
Machine Learning Model for Crypto Macro-Fundamental Strategy

XGBoost model with walk-forward validation and Bayesian hyperparameter optimization.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except ImportError:
    print("Warning: xgboost or scikit-learn not installed")

try:
    import optuna
except ImportError:
    print("Warning: optuna not installed")


class CryptoMLModel:
    """
    Machine learning model for predicting Bitcoin returns.
    Uses XGBoost with walk-forward validation.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.model_type = self.model_config['type']
        
        self.model = None
        self.feature_importance = None
    
    def create_walk_forward_splits(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> List[Tuple]:
        """
        Create walk-forward time series splits.
        
        Args:
            features: Feature DataFrame
            target: Target series
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        validation_config = self.model_config['validation']
        
        train_window = validation_config['train_window']
        test_window = validation_config['test_window']
        step_size = validation_config['step_size']
        
        splits = []
        
        start = 0
        while start + train_window + test_window <= len(features):
            train_end = start + train_window
            test_end = train_end + test_window
            
            train_indices = range(start, train_end)
            test_indices = range(train_end, test_end)
            
            splits.append((list(train_indices), list(test_indices)))
            
            start += step_size
        
        print(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        params: Dict = None
    ):
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            params: Model hyperparameters (optional)
        
        Returns:
            Trained model
        """
        if params is None:
            params = self.model_config['xgboost']['default_params']
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        eval_list = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_list.append((dval, 'val'))
        
        # Train
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=eval_list,
            verbose_eval=False
        )
        
        return model
    
    def predict(
        self,
        model,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            model: Trained model
            X: Features
        
        Returns:
            Predictions
        """
        if self.model_type == 'xgboost':
            dmatrix = xgb.DMatrix(X)
            predictions = model.predict(dmatrix)
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def calculate_sharpe_ratio(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray
    ) -> float:
        """
        Calculate Sharpe ratio of strategy based on predictions.
        
        Args:
            predictions: Predicted returns
            actual_returns: Actual returns
        
        Returns:
            Annualized Sharpe ratio
        """
        # Create positions based on predictions (long if positive, short if negative)
        positions = np.sign(predictions)
        
        # Calculate strategy returns
        strategy_returns = positions * actual_returns
        
        # Sharpe ratio
        mean_return = np.mean(strategy_returns) * 252  # Annualized
        std_return = np.std(strategy_returns) * np.sqrt(252)
        
        sharpe = mean_return / (std_return + 1e-8)
        
        return sharpe
    
    def hyperparameter_optimization(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50
    ) -> Dict:
        """
        Bayesian hyperparameter optimization with Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            n_trials: Number of trials
        
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            # Sample hyperparameters
            params = {
                'n_estimators': trial.suggest_categorical(
                    'n_estimators',
                    self.model_config['xgboost']['search_space']['n_estimators']
                ),
                'max_depth': trial.suggest_categorical(
                    'max_depth',
                    self.model_config['xgboost']['search_space']['max_depth']
                ),
                'learning_rate': trial.suggest_categorical(
                    'learning_rate',
                    self.model_config['xgboost']['search_space']['learning_rate']
                ),
                'subsample': trial.suggest_categorical(
                    'subsample',
                    self.model_config['xgboost']['search_space']['subsample']
                ),
                'colsample_bytree': trial.suggest_categorical(
                    'colsample_bytree',
                    self.model_config['xgboost']['search_space']['colsample_bytree']
                ),
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'random_state': 42
            }
            
            # Train model
            model = self.train_xgboost(X_train, y_train, X_val, y_val, params)
            
            # Predict
            predictions = self.predict(model, X_val)
            
            # Calculate Sharpe ratio (optimization objective)
            sharpe = self.calculate_sharpe_ratio(predictions, y_val.values)
            
            return sharpe
        
        # Create study
        study = optuna.create_study(direction='maximize', study_name='crypto_model')
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nBest Sharpe Ratio: {study.best_value:.4f}")
        print(f"Best hyperparameters: {study.best_params}")
        
        return study.best_params
    
    def walk_forward_validation(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        optimize_hyperparams: bool = False
    ) -> Dict:
        """
        Perform walk-forward validation.
        
        Args:
            features: Full feature DataFrame
            target: Full target series
            optimize_hyperparams: Whether to optimize hyperparameters
        
        Returns:
            Dict with validation results
        """
        # Align features and target
        aligned = features.join(target, rsuffix='_target').dropna()
        X = aligned[features.columns]
        y = aligned[target.name]
        
        # Create splits
        splits = self.create_walk_forward_splits(X, y)
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        fold_results = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"\nFold {i+1}/{len(splits)}")
            
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Optionally optimize hyperparameters (on first fold only)
            if optimize_hyperparams and i == 0:
                print("Optimizing hyperparameters...")
                
                # Use last 20% of train as validation
                val_size = int(len(X_train) * 0.2)
                X_train_opt = X_train.iloc[:-val_size]
                y_train_opt = y_train.iloc[:-val_size]
                X_val_opt = X_train.iloc[-val_size:]
                y_val_opt = y_train.iloc[-val_size:]
                
                best_params = self.hyperparameter_optimization(
                    X_train_opt, y_train_opt,
                    X_val_opt, y_val_opt,
                    n_trials=self.model_config['hyperopt']['n_trials']
                )
                
                # Update default params
                self.model_config['xgboost']['default_params'].update(best_params)
            
            # Train model
            model = self.train_xgboost(X_train, y_train)
            
            # Predict
            predictions = self.predict(model, X_test)
            
            # Store results
            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)
            all_dates.extend(y_test.index)
            
            # Calculate fold metrics
            fold_sharpe = self.calculate_sharpe_ratio(predictions, y_test.values)
            fold_mse = mean_squared_error(y_test.values, predictions)
            
            fold_results.append({
                'fold': i + 1,
                'sharpe': fold_sharpe,
                'mse': fold_mse,
                'test_period': f"{y_test.index[0]} to {y_test.index[-1]}"
            })
            
            print(f"  Sharpe: {fold_sharpe:.4f}, MSE: {fold_mse:.6f}")
        
        # Overall results
        overall_sharpe = self.calculate_sharpe_ratio(
            np.array(all_predictions),
            np.array(all_actuals)
        )
        overall_mse = mean_squared_error(all_actuals, all_predictions)
        
        results_df = pd.DataFrame({
            'date': all_dates,
            'actual': all_actuals,
            'predicted': all_predictions
        }).set_index('date')
        
        return {
            'results_df': results_df,
            'overall_sharpe': overall_sharpe,
            'overall_mse': overall_mse,
            'fold_results': fold_results
        }
    
    def get_feature_importance(self, model) -> pd.Series:
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained model
        
        Returns:
            Series with feature importances
        """
        if self.model_type == 'xgboost':
            importance_dict = model.get_score(importance_type='gain')
            importance = pd.Series(importance_dict).sort_values(ascending=False)
        else:
            importance = pd.Series(
                model.feature_importances_,
                index=model.feature_names_in_
            ).sort_values(ascending=False)
        
        return importance


# Test code
if __name__ == "__main__":
    from data_acquisition import DataAcquisition
    from feature_engineering import FeatureEngineer
    
    # Load data
    data_acq = DataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset()
    
    # Engineer features
    engineer = FeatureEngineer('config.yaml')
    features = engineer.engineer_all_features(dataset['prices'], dataset['events'])
    target = engineer.create_target_variable(dataset['prices'])
    
    # Train model
    model = CryptoMLModel('config.yaml')
    
    print("\nRunning walk-forward validation (without hyperparameter optimization)...")
    results = model.walk_forward_validation(features, target, optimize_hyperparams=False)
    
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Overall Sharpe Ratio: {results['overall_sharpe']:.4f}")
    print(f"Overall MSE: {results['overall_mse']:.6f}")
    
    print(f"\nFold-by-Fold Results:")
    for fold in results['fold_results']:
        print(f"  Fold {fold['fold']}: Sharpe={fold['sharpe']:.4f}, Period={fold['test_period']}")
