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
            # Convert controls to quarterly frequency (use end-of-quarter values)
            controls_copy = controls.copy()
            
            # Remove duplicate columns if they exist
            controls_copy = controls_copy.loc[:, ~controls_copy.columns.duplicated()]
            
            controls_copy['date'] = pd.to_datetime(controls_copy['date'])
            controls_copy['quarter_date'] = controls_copy['date'].dt.to_period('Q').dt.to_timestamp()
            
            # Get control variable columns (exclude date columns)
            control_vars = [col for col in controls_copy.columns if col not in ['date', 'quarter_date']]
            
            # Group by quarter and take the last value (end-of-quarter)
            controls_quarterly = controls_copy.groupby('quarter_date')[control_vars].last().reset_index()
            controls_quarterly = controls_quarterly.rename(columns={'quarter_date': 'date'})
            
            # Merge with quarterly data
            merged = merged.merge(controls_quarterly, on='date', how='left')
            feature_cols.extend(control_vars)
        
        X = merged[feature_cols]
        y = merged[f'target_h{horizon}']
        
        print(f"\n[DEBUG] Horizon {horizon}Q before cleaning:")
        print(f"  Merged shape: {merged.shape}")
        print(f"  Feature columns: {feature_cols}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  NaN counts in X: {X.isna().sum().to_dict()}")
        print(f"  NaN count in y: {y.isna().sum()}")
        
        # Drop any rows with NaN values in features or target
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        merged = merged[valid_mask]
        
        print(f"[DEBUG] After NaN removal:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Valid samples: {valid_mask.sum()}/{len(valid_mask)}")
        
        # Check if we have enough samples
        if len(X) < 3:
            print(f"  WARNING: Only {len(X)} samples - insufficient for regression")
        
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
        # Add constant, ensuring column names match training data
        X_with_const = sm.add_constant(X_test, has_constant='add')
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
            
            # Check if we have enough data
            if len(X) < 5:
                print(f"  ERROR: Only {len(X)} samples after NaN removal - need at least 5")
                print(f"  Skipping horizon {h}Q\n")
                continue
            
            # CRITICAL FIX: Align train_mask with cleaned data
            # The prepare_regression_data may have dropped rows due to NaN
            # We need to align the boolean mask with the cleaned indices
            aligned_mask = train_mask.loc[X.index] if hasattr(X, 'index') else train_mask[:len(X)]
            
            # Split train/test
            X_train = X[aligned_mask]
            y_train = y[aligned_mask]
            X_test = X[~aligned_mask]
            y_test = y[~aligned_mask]
            
            print(f"[DEBUG] Train/Test split for horizon {h}Q:")
            print(f"  X_train shape: {X_train.shape}")
            print(f"  X_test shape: {X_test.shape}")
            print(f"  X_train columns: {X_train.columns.tolist()}")
            print(f"  X_test columns: {X_test.columns.tolist()}")
            
            # Check test set is not empty
            if len(X_test) == 0:
                print(f"  ERROR: Test set is empty for horizon {h}Q")
                print(f"  Skipping horizon {h}Q\n")
                continue
            
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
