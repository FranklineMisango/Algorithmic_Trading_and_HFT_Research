"""
Jump Detection Module - Phase 1
Implements baseline jump-diffusion model with 3-sigma threshold rule
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JumpDetector:
    """
    Detects discontinuous price jumps using baseline model + 3-sigma rule
    """
    
    def __init__(self, config: Dict):
        """
        Initialize jump detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.jump_config = config['jump_detection']
        self.lookback = self.jump_config['baseline_model']['lookback_days']
        self.threshold = self.jump_config['threshold']['sigma_multiplier']
        self.models = {}  # Store fitted models for each asset
        
    def detect_jumps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect jumps for all assets
        
        Args:
            df: DataFrame with returns data
            
        Returns:
            DataFrame with jump indicators and metrics
        """
        logger.info("Detecting jumps using 3-sigma threshold rule...")
        
        results = []
        
        for asset in df['asset'].unique():
            asset_data = df[df['asset'] == asset].copy()
            asset_results = self._detect_asset_jumps(asset_data, asset)
            results.append(asset_results)
        
        # Combine all results
        df_with_jumps = pd.concat(results, ignore_index=True)
        
        # Calculate summary statistics
        self._log_jump_statistics(df_with_jumps)
        
        return df_with_jumps
    
    def _detect_asset_jumps(self, asset_df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Detect jumps for a single asset
        
        Args:
            asset_df: DataFrame for single asset
            asset: Asset name
            
        Returns:
            DataFrame with jump indicators
        """
        # Extract features
        features_df = self._engineer_features(asset_df)
        
        # Fit baseline model
        model, residuals = self._fit_baseline_model(features_df, asset)
        
        # Apply 3-sigma threshold
        jump_indicators = self._apply_threshold(residuals)
        
        # Combine results
        result_df = asset_df.copy()
        result_df['residual'] = residuals
        result_df['is_jump'] = jump_indicators
        result_df['jump_size'] = np.where(jump_indicators, residuals, 0)
        result_df['jump_direction'] = np.where(jump_indicators, np.sign(residuals), 0)
        
        return result_df
    
    def _engineer_features(self, asset_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for baseline model
        
        Features:
        - recent_return: Average return over lookback window
        - recent_volatility: Std dev of returns over lookback window
        - log_volume: Log of trading volume
        
        Args:
            asset_df: DataFrame for single asset
            
        Returns:
            DataFrame with features
        """
        df = asset_df.copy()
        
        # Recent return (rolling mean)
        df['recent_return'] = df['returns'].rolling(
            window=self.lookback, 
            min_periods=1
        ).mean()
        
        # Recent volatility (rolling std)
        df['recent_volatility'] = df['returns'].rolling(
            window=self.lookback, 
            min_periods=1
        ).std()
        
        # Log volume (handle zeros)
        df['log_volume'] = np.log(df['volume'] + 1)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    
    def _fit_baseline_model(self, features_df: pd.DataFrame, asset: str) -> Tuple[LinearRegression, np.ndarray]:
        """
        Fit baseline jump-diffusion model
        
        Model: return_t = β₀ + β₁(recent_return) + β₂(recent_vol) + β₃(log_vol) + ε
        
        Args:
            features_df: DataFrame with features
            asset: Asset name
            
        Returns:
            Tuple of (fitted_model, residuals)
        """
        # Prepare features and target
        feature_cols = ['recent_return', 'recent_volatility', 'log_volume']
        X = features_df[feature_cols].values
        y = features_df['returns'].values
        
        # Remove NaN/inf values
        valid_idx = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate residuals
        y_pred = model.predict(X)
        residuals_valid = y - y_pred
        
        # Reconstruct full residuals array
        residuals = np.full(len(features_df), np.nan)
        residuals[valid_idx] = residuals_valid
        
        # Store model
        self.models[asset] = {
            'model': model,
            'feature_cols': feature_cols,
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
        
        # Log model info
        logger.info(f"  {asset}: R²={model.score(X, y):.4f}, "
                   f"β=[{', '.join([f'{c:.4f}' for c in model.coef_])}]")
        
        return model, residuals
    
    def _apply_threshold(self, residuals: np.ndarray) -> np.ndarray:
        """
        Apply 3-sigma threshold rule to identify jumps
        
        Args:
            residuals: Model residuals
            
        Returns:
            Boolean array indicating jumps
        """
        # Remove NaN values for threshold calculation
        valid_residuals = residuals[np.isfinite(residuals)]
        
        if len(valid_residuals) == 0:
            return np.zeros(len(residuals), dtype=bool)
        
        # Calculate threshold
        mean = np.mean(valid_residuals)
        std = np.std(valid_residuals)
        threshold = self.threshold * std
        
        # Identify jumps
        is_jump = np.abs(residuals - mean) > threshold
        
        return is_jump
    
    def calculate_jump_metrics(self, df_with_jumps: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate jump intensity and size metrics
        
        Args:
            df_with_jumps: DataFrame with jump indicators
            
        Returns:
            Dictionary of metrics per asset
        """
        metrics = {}
        
        for asset in df_with_jumps['asset'].unique():
            asset_data = df_with_jumps[df_with_jumps['asset'] == asset]
            
            # Jump intensity (frequency)
            jump_intensity = asset_data['is_jump'].mean()
            
            # Jump size (magnitude)
            jump_returns = asset_data[asset_data['is_jump']]['jump_size']
            if len(jump_returns) > 0:
                avg_jump_size = np.abs(jump_returns).mean()
                max_jump_size = np.abs(jump_returns).max()
            else:
                avg_jump_size = 0
                max_jump_size = 0
            
            # Jump direction bias
            if len(jump_returns) > 0:
                positive_jumps = (jump_returns > 0).sum()
                negative_jumps = (jump_returns < 0).sum()
                direction_bias = (positive_jumps - negative_jumps) / len(jump_returns)
            else:
                direction_bias = 0
            
            metrics[asset] = {
                'jump_intensity': jump_intensity,
                'avg_jump_size': avg_jump_size,
                'max_jump_size': max_jump_size,
                'direction_bias': direction_bias,
                'n_jumps': asset_data['is_jump'].sum(),
                'n_positive_jumps': (asset_data['jump_direction'] == 1).sum(),
                'n_negative_jumps': (asset_data['jump_direction'] == -1).sum()
            }
        
        return metrics
    
    def identify_cojumps(self, df_with_jumps: pd.DataFrame) -> pd.DataFrame:
        """
        Identify co-jumps (simultaneous jumps across assets)
        
        Args:
            df_with_jumps: DataFrame with jump indicators
            
        Returns:
            DataFrame with co-jump information by date
        """
        # Pivot to wide format
        jump_matrix = df_with_jumps.pivot(
            index='date', 
            columns='asset', 
            values='is_jump'
        )
        
        # Count co-jumps per date
        cojump_counts = jump_matrix.sum(axis=1)
        
        # Identify significant co-jump events (>= 30% of assets)
        n_assets = len(jump_matrix.columns)
        threshold = n_assets * self.jump_config['cojump_threshold']
        
        cojump_df = pd.DataFrame({
            'date': cojump_counts.index,
            'n_cojumps': cojump_counts.values,
            'cojump_ratio': cojump_counts.values / n_assets,
            'is_systemic': cojump_counts.values >= threshold
        })
        
        logger.info(f"Identified {cojump_df['is_systemic'].sum()} systemic co-jump events")
        
        return cojump_df
    
    def _log_jump_statistics(self, df_with_jumps: pd.DataFrame):
        """
        Log summary statistics for detected jumps
        
        Args:
            df_with_jumps: DataFrame with jump indicators
        """
        logger.info("\n=== Jump Detection Summary ===")
        
        metrics = self.calculate_jump_metrics(df_with_jumps)
        
        for asset, m in list(metrics.items())[:5]:  # Show first 5
            logger.info(
                f"  {asset}: "
                f"intensity={m['jump_intensity']*100:.2f}%, "
                f"avg_size={m['avg_jump_size']*100:.2f}%, "
                f"bias={m['direction_bias']:+.2f}"
            )
        
        # Overall statistics
        total_jumps = sum(m['n_jumps'] for m in metrics.values())
        total_obs = len(df_with_jumps)
        logger.info(f"\nTotal jumps: {total_jumps} / {total_obs} obs ({total_jumps/total_obs*100:.2f}%)")


def detect_and_analyze_jumps(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to detect jumps and calculate metrics
    
    Args:
        df: Input DataFrame with returns
        config: Configuration dictionary
        
    Returns:
        Tuple of (df_with_jumps, metrics_dict)
    """
    detector = JumpDetector(config)
    
    # Detect jumps
    df_with_jumps = detector.detect_jumps(df)
    
    # Calculate metrics
    metrics = detector.calculate_jump_metrics(df_with_jumps)
    
    # Identify co-jumps
    cojump_df = detector.identify_cojumps(df_with_jumps)
    
    return df_with_jumps, metrics, cojump_df


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from data_loader import load_and_prepare_data
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    data_splits = load_and_prepare_data(config)
    train_df = data_splits['train']
    
    # Detect jumps
    df_with_jumps, metrics, cojump_df = detect_and_analyze_jumps(train_df, config)
    
    print("\n=== Jump Detection Complete ===")
    print(f"Assets with jumps: {df_with_jumps['is_jump'].groupby(df_with_jumps['asset']).sum().to_dict()}")
    print(f"Systemic co-jump events: {cojump_df['is_systemic'].sum()}")
