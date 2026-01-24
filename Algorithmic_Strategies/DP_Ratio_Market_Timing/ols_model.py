"""
OLS Regression Model for Dividend-Price Ratio Strategy

This module implements:
1. Ordinary Least Squares (OLS) regression
2. In-sample model training (e.g., 1928-2002)
3. Out-of-sample validation (e.g., 2003-2017+)
4. Statistical significance testing
5. Performance metrics (R², RMSE, p-values)

Model Equation:
    Return_month_t = α + β * Δlog(D/P)_month_t-1 + ε
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from scipy import stats
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class DPRatioOLSModel:
    """
    OLS regression model for D/P ratio strategy.
    """
    
    def __init__(self, config: dict):
        """
        Initialize OLS model.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.model = None
        self.model_stats = None
        self.in_sample_results = {}
        self.out_sample_results = {}
        
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Split data into in-sample and out-of-sample periods.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features (delta_log_dp)
        y : pd.Series
            Target (next_month_return)
            
        Returns
        -------
        Tuple
            (X_in, X_out, y_in, y_out)
        """
        in_sample_end = pd.to_datetime(self.config['model']['in_sample_end'])
        out_sample_start = pd.to_datetime(self.config['model']['out_sample_start'])
        
        # Split based on dates
        X_in = X[X.index <= in_sample_end]
        y_in = y[y.index <= in_sample_end]
        
        X_out = X[X.index >= out_sample_start]
        y_out = y[y.index >= out_sample_start]
        
        print(f"In-sample period: {X_in.index[0].strftime('%Y-%m-%d')} to {X_in.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Observations: {len(X_in)}")
        
        print(f"Out-of-sample period: {X_out.index[0].strftime('%Y-%m-%d')} to {X_out.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Observations: {len(X_out)}")
        
        return X_in, X_out, y_in, y_out
    
    def train_ols_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train OLS regression model on in-sample data.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        """
        print("\n" + "="*60)
        print("TRAINING OLS MODEL (IN-SAMPLE)")
        print("="*60)
        
        # Use statsmodels for full statistical analysis
        X_with_const = sm.add_constant(X_train)
        self.model_stats = sm.OLS(y_train, X_with_const).fit()
        
        # Also fit scikit-learn model for easy predictions
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Extract results
        alpha = self.model_stats.params['const']
        beta = self.model_stats.params[X_train.columns[0]]
        p_value = self.model_stats.pvalues[X_train.columns[0]]
        r_squared = self.model_stats.rsquared
        adj_r_squared = self.model_stats.rsquared_adj
        
        print(f"\nModel Equation:")
        print(f"  Return_t = {alpha:.6f} + {beta:.6f} * Δlog(D/P)_t-1 + ε")
        
        print(f"\nCoefficients:")
        print(f"  α (Intercept): {alpha:.6f}")
        print(f"  β (Slope):     {beta:.6f}")
        
        print(f"\nStatistical Significance:")
        print(f"  p-value (β):   {p_value:.6f} {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''}")
        print(f"  Significant:   {'YES' if p_value < 0.05 else 'NO'}")
        
        print(f"\nModel Fit (In-Sample):")
        print(f"  R²:            {r_squared:.4f} ({r_squared*100:.2f}%)")
        print(f"  Adj. R²:       {adj_r_squared:.4f}")
        print(f"  F-statistic:   {self.model_stats.fvalue:.2f}")
        print(f"  Prob(F):       {self.model_stats.f_pvalue:.6f}")
        
        # Calculate RMSE and MAE
        y_pred_in = self.model.predict(X_train)
        rmse_in = np.sqrt(mean_squared_error(y_train, y_pred_in))
        mae_in = mean_absolute_error(y_train, y_pred_in)
        
        print(f"\nPrediction Error (In-Sample):")
        print(f"  RMSE:          {rmse_in:.4f} ({rmse_in*100:.2f}% monthly)")
        print(f"  MAE:           {mae_in:.4f} ({mae_in*100:.2f}% monthly)")
        
        # Store results
        self.in_sample_results = {
            'alpha': alpha,
            'beta': beta,
            'p_value': p_value,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'rmse': rmse_in,
            'mae': mae_in,
            'n_obs': len(X_train),
            'predictions': y_pred_in,
            'actuals': y_train.values
        }
        
        print("\n" + "="*60)
        
    def validate_out_of_sample(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Validate model on out-of-sample data (THE CRITICAL TEST).
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
        """
        print("\n" + "="*60)
        print("OUT-OF-SAMPLE VALIDATION (THE CRITICAL TEST)")
        print("="*60)
        
        # Generate predictions using the in-sample trained model
        y_pred_out = self.model.predict(X_test)
        
        # Calculate metrics
        rmse_out = np.sqrt(mean_squared_error(y_test, y_pred_out))
        mae_out = mean_absolute_error(y_test, y_pred_out)
        r2_out = r2_score(y_test, y_pred_out)
        
        # Compare to naive forecast (historical mean)
        y_mean = self.in_sample_results['actuals'].mean()
        naive_predictions = np.full(len(y_test), y_mean)
        rmse_naive = np.sqrt(mean_squared_error(y_test, naive_predictions))
        
        print(f"\nOut-of-Sample Performance:")
        print(f"  RMSE:          {rmse_out:.4f} ({rmse_out*100:.2f}% monthly)")
        print(f"  MAE:           {mae_out:.4f} ({mae_out*100:.2f}% monthly)")
        print(f"  R² (OOS):      {r2_out:.4f} ({r2_out*100:.2f}%)")
        
        print(f"\nComparison to Naive Forecast (Historical Mean):")
        print(f"  Naive RMSE:    {rmse_naive:.4f} ({rmse_naive*100:.2f}%)")
        print(f"  Improvement:   {(rmse_naive - rmse_out)/rmse_naive*100:.2f}%")
        print(f"  Better?:       {'YES' if rmse_out < rmse_naive else 'NO'}")
        
        # Directional accuracy
        actual_direction = (y_test > 0).astype(int)
        pred_direction = (y_pred_out > 0).astype(int)
        directional_accuracy = (actual_direction == pred_direction).mean()
        
        print(f"\nDirectional Accuracy:")
        print(f"  Correct sign:  {directional_accuracy:.2%}")
        print(f"  (>50% = skill)")
        
        # Store results
        self.out_sample_results = {
            'rmse': rmse_out,
            'mae': mae_out,
            'r2': r2_out,
            'rmse_naive': rmse_naive,
            'directional_accuracy': directional_accuracy,
            'n_obs': len(X_test),
            'predictions': y_pred_out,
            'actuals': y_test.values
        }
        
        print("\n" + "="*60)
        
    def print_model_summary(self):
        """
        Print comprehensive model summary.
        """
        print("\n" + "="*60)
        print("FULL OLS REGRESSION SUMMARY")
        print("="*60)
        print(self.model_stats.summary())
        print("="*60)
        
    def get_coefficients(self) -> Tuple[float, float]:
        """
        Get model coefficients.
        
        Returns
        -------
        Tuple[float, float]
            (alpha, beta)
        """
        return self.in_sample_results['alpha'], self.in_sample_results['beta']
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        np.ndarray
            Predicted returns
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_ols_model() first.")
        
        return self.model.predict(X)
    
    def diagnose_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Run diagnostic tests on the model.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        """
        print("\n" + "="*60)
        print("MODEL DIAGNOSTICS")
        print("="*60)
        
        # Get residuals
        y_pred = self.model.predict(X_train)
        residuals = y_train - y_pred
        
        # 1. Normality test (Jarque-Bera)
        from statsmodels.stats.stattools import jarque_bera
        jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
        
        print(f"\n1. Residual Normality (Jarque-Bera Test):")
        print(f"   Test statistic: {jb_stat:.4f}")
        print(f"   p-value: {jb_pvalue:.4f}")
        print(f"   Normal?: {'YES' if jb_pvalue > 0.05 else 'NO (non-normal)'}")
        print(f"   Skewness: {skew:.4f}")
        print(f"   Kurtosis: {kurtosis:.4f}")
        
        # 2. Heteroskedasticity test (Breusch-Pagan)
        from statsmodels.stats.diagnostic import het_breuschpagan
        X_with_const = sm.add_constant(X_train)
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_with_const)
        
        print(f"\n2. Heteroskedasticity (Breusch-Pagan Test):")
        print(f"   Test statistic: {bp_stat:.4f}")
        print(f"   p-value: {bp_pvalue:.4f}")
        print(f"   Homoskedastic?: {'YES' if bp_pvalue > 0.05 else 'NO (heteroskedastic)'}")
        
        # 3. Autocorrelation test (Durbin-Watson)
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(residuals)
        
        print(f"\n3. Autocorrelation (Durbin-Watson Test):")
        print(f"   Test statistic: {dw_stat:.4f}")
        print(f"   Interpretation: 2 = no autocorrelation, 0 = positive, 4 = negative")
        print(f"   Autocorrelated?: {'YES (positive)' if dw_stat < 1.5 else 'YES (negative)' if dw_stat > 2.5 else 'NO'}")
        
        # 4. Linearity check (residuals vs fitted)
        corr_resid_fitted = np.corrcoef(y_pred, residuals)[0, 1]
        print(f"\n4. Linearity Check:")
        print(f"   Corr(Fitted, Residuals): {corr_resid_fitted:.4f}")
        print(f"   Linear?: {'YES' if abs(corr_resid_fitted) < 0.1 else 'NO (pattern in residuals)'}")
        
        print("\n" + "="*60)
        
        return {
            'jarque_bera': jb_pvalue,
            'breusch_pagan': bp_pvalue,
            'durbin_watson': dw_stat,
            'residuals': residuals
        }
    
    def save_model(self, filepath: str = 'models/ols_model.pkl'):
        """
        Save trained model.
        
        Parameters
        ----------
        filepath : str
            Path to save model
        """
        import pickle
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_stats': self.model_stats,
            'in_sample_results': self.in_sample_results,
            'out_sample_results': self.out_sample_results,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {filepath}")


def main():
    """
    Test OLS model.
    """
    import yaml
    from data_acquisition import DividendPriceDataFetcher
    from feature_engineering import DPRatioFeatureEngineer
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get data
    fetcher = DividendPriceDataFetcher(config)
    data = fetcher.fetch_and_prepare_data()
    
    # Engineer features
    engineer = DPRatioFeatureEngineer(config)
    data = engineer.engineer_all_features(data)
    X, y = engineer.prepare_model_dataset(data)
    
    # Initialize model
    model = DPRatioOLSModel(config)
    
    # Split data
    X_in, X_out, y_in, y_out = model.split_data(X, y)
    
    # Train on in-sample
    model.train_ols_model(X_in, y_in)
    
    # Print full summary
    model.print_model_summary()
    
    # Diagnose model
    diagnostics = model.diagnose_model(X_in, y_in)
    
    # Validate out-of-sample
    model.validate_out_of_sample(X_out, y_out)
    
    # Save model
    model.save_model()
    
    # Get coefficients
    alpha, beta = model.get_coefficients()
    print(f"\nFinal Model Coefficients:")
    print(f"  α = {alpha:.6f}")
    print(f"  β = {beta:.6f}")


if __name__ == "__main__":
    main()
