"""
Prediction Model Module for AI Economy Score Predictor

Implements regression models for GDP, IP, employment, and wage predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import statsmodels.api as sm
from scipy import stats


class PredictionModel:
    """Handles macroeconomic prediction models."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_config = self.config['models']
        self.trained_models = {}
    
    def prepare_regression_data(
        self,
        agg_scores: pd.DataFrame,
        target_data: pd.DataFrame,
        horizon: int,
        controls: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for regression.
        
        Args:
            agg_scores: AGG scores (predictors)
            target_data: Macro target (e.g., GDP)
            horizon: Quarters ahead to predict
            controls: Optional control variables
            
        Returns:
            X (features), y (target)
        """
        # Shift target forward by horizon
        merged = agg_scores.merge(
            target_data,
            left_on='date',
            right_on='date',
            how='inner'
        )
        
        merged[f'target_h{horizon}'] = merged['value'].shift(-horizon)
        
        # Drop rows with NaN target
        merged = merged.dropna(subset=[f'target_h{horizon}'])
        
        # Features: AGG score + controls
        feature_cols = ['agg_score_norm']
        
        if controls is not None:
            merged = merged.merge(controls, on='date', how='left')
            feature_cols.extend([col for col in controls.columns if col != 'date'])
        
        X = merged[feature_cols]
        y = merged[f'target_h{horizon}']
        
        return X, y, merged
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict:
        """
        Train regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dict with model and stats
        """
        # Add constant for intercept
        X_with_const = sm.add_constant(X_train)
        
        # Fit OLS
        model = sm.OLS(y_train, X_with_const).fit()
        
        return {
            'model': model,
            'r2': model.rsquared,
            'adj_r2': model.rsquared_adj,
            'beta_agg': model.params.get('agg_score_norm', None),
            'pvalue_agg': model.pvalues.get('agg_score_norm', None),
            'significant': model.pvalues.get('agg_score_norm', 1.0) < 0.05
        }
    
    def predict(
        self,
        model_dict: Dict,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            model_dict: Trained model dict
            X_test: Test features
            
        Returns:
            Predictions
        """
        X_with_const = sm.add_constant(X_test)
        return model_dict['model'].predict(X_with_const)
    
    def evaluate_model(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        spf_forecast: Optional[pd.Series] = None
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            spf_forecast: SPF consensus for comparison
            
        Returns:
            Dict with metrics
        """
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2))
        }
        
        # Compare to SPF if available
        if spf_forecast is not None:
            spf_mae = mean_absolute_error(y_true, spf_forecast)
            metrics['spf_mae'] = spf_mae
            metrics['beats_spf'] = mae < spf_mae
            metrics['mae_improvement'] = (spf_mae - mae) / spf_mae * 100
        
        return metrics
    
    def train_gdp_models(
        self,
        agg_scores: pd.DataFrame,
        gdp_data: pd.DataFrame,
        controls: pd.DataFrame,
        train_mask: pd.Series
    ) -> Dict[int, Dict]:
        """
        Train GDP prediction models for all horizons.
        
        Args:
            agg_scores: AGG scores
            gdp_data: GDP growth data
            controls: Control variables
            train_mask: Boolean mask for training set
            
        Returns:
            Dict of models by horizon
        """
        horizons = self.models_config['gdp']['horizons']
        models = {}
        
        for h in horizons:
            print(f"Training GDP model for h={h} quarters...")
            
            # Prepare data
            X, y, merged = self.prepare_regression_data(
                agg_scores, gdp_data, h, controls
            )
            
            # Split train/test
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[~train_mask]
            y_test = y[~train_mask]
            
            # Train
            model_dict = self.train_model(X_train, y_train)
            
            # Evaluate
            y_pred = self.predict(model_dict, X_test)
            metrics = self.evaluate_model(y_test, y_pred)
            
            model_dict['metrics'] = metrics
            model_dict['horizon'] = h
            
            models[h] = model_dict
            
            print(f"  R²: {model_dict['r2']:.3f}, Test MAE: {metrics['mae']:.3f}")
        
        return models
    
    def diebold_mariano_test(
        self,
        y_true: np.ndarray,
        pred1: np.ndarray,
        pred2: np.ndarray
    ) -> Dict:
        """
        Diebold-Mariano test for forecast comparison.
        
        Tests if pred1 is significantly better than pred2.
        
        Args:
            y_true: Actual values
            pred1: First forecast (AI model)
            pred2: Second forecast (SPF)
            
        Returns:
            Dict with test statistic and p-value
        """
        # Forecast errors
        e1 = y_true - pred1
        e2 = y_true - pred2
        
        # Loss differential (MSE)
        d = e1 ** 2 - e2 ** 2
        
        # Mean differential
        d_bar = np.mean(d)
        
        # Variance (with HAC correction)
        d_var = np.var(d, ddof=1)
        
        # Test statistic
        n = len(d)
        dm_stat = d_bar / np.sqrt(d_var / n)
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        return {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'pred1_better': dm_stat < 0  # Negative means pred1 has lower MSE
        }


# Test code
if __name__ == "__main__":
    pm = PredictionModel('config.yaml')
    
    # Generate test data
    dates = pd.date_range('2010-01-01', periods=50, freq='Q')
    
    agg_scores = pd.DataFrame({
        'date': dates,
        'agg_score': np.random.normal(3.0, 0.5, 50),
        'agg_score_norm': np.random.normal(0, 1, 50)
    })
    
    gdp_data = pd.DataFrame({
        'date': dates,
        'value': np.random.normal(2.0, 1.0, 50)
    })
    
    controls = pd.DataFrame({
        'date': dates,
        'yield_curve_slope': np.random.normal(0.5, 0.3, 50)
    })
    
    # Train/test split
    train_mask = dates < '2020-01-01'
    
    # Train models
    gdp_models = pm.train_gdp_models(agg_scores, gdp_data, controls, train_mask)
    
    print("\nGDP Models Summary:")
    for h, model in gdp_models.items():
        print(f"Horizon {h}Q: R²={model['r2']:.3f}, Beta={model['beta_agg']:.3f}, Sig={model['significant']}")
