"""
Feature Engineering for Crypto Macro-Fundamental Strategy

Implements the 4-factor model: External Macro, Risk Premium, Adoption, and Institutional signals.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Tuple, List


class FeatureEngineer:
    """Creates features for crypto price prediction model."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.features_config = self.config['features']
    
    def calculate_external_macro_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate External Macro Signal.
        
        Formula: Z-Score(Δ(US 2-Year Treasury Yield) + Δ(VIX Index))
        
        Args:
            data: DataFrame with 'DGS2' and 'VIX' columns
        
        Returns:
            Z-scored macro signal
        """
        macro_config = self.features_config['external_macro']
        lookback = macro_config['lookback_period']
        
        # Calculate changes
        treasury_change = data['DGS2'].diff()
        vix_change = data['VIX'].diff()
        
        # Combine
        macro_raw = treasury_change + vix_change
        
        # Z-Score normalization
        rolling_mean = macro_raw.rolling(window=lookback).mean()
        rolling_std = macro_raw.rolling(window=lookback).std()
        
        macro_signal = (macro_raw - rolling_mean) / (rolling_std + 1e-8)
        
        return macro_signal
    
    def calculate_crypto_risk_premium_signal(
        self,
        data: pd.DataFrame,
        window: int = 5
    ) -> pd.Series:
        """
        Calculate Crypto Risk Premium Signal.
        
        Formula: Growth(Stablecoin MCap) / Growth(Total Crypto MCap)
        
        Args:
            data: DataFrame with stablecoin and crypto market cap columns
            window: Growth calculation window (5 or 20 days)
        
        Returns:
            Risk premium signal
        """
        # Total stablecoin market cap
        stablecoin_mcap = data['Total_Stablecoin_MCap']
        
        # Total crypto market cap (excluding stablecoins for denominator)
        total_crypto_mcap = data['Total_Crypto_MCap']
        
        # Calculate growth rates
        stablecoin_growth = stablecoin_mcap.pct_change(window)
        crypto_growth = total_crypto_mcap.pct_change(window)
        
        # Risk premium ratio
        risk_premium = stablecoin_growth / (crypto_growth + 1e-8)
        
        return risk_premium
    
    def calculate_adoption_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Adoption/Growth Signal.
        
        Formula: Δ(Log(Total Crypto Market Cap ex-Stablecoins))
        
        Args:
            data: DataFrame with market cap data
        
        Returns:
            Adoption signal
        """
        window = self.features_config['adoption_growth']['window']
        
        # Total crypto market cap (already excludes stablecoins in our data)
        crypto_mcap = data['Total_Crypto_MCap']
        
        # Log transform
        log_mcap = np.log(crypto_mcap + 1)
        
        # Rolling growth rate
        adoption_signal = log_mcap.diff(window)
        
        return adoption_signal
    
    def calculate_institutional_signal(
        self,
        dates: pd.DatetimeIndex,
        events: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate Institutional Validation Signal.
        
        Binary flag set to 1 for 30 days following major institutional events.
        
        Args:
            dates: Date index for output series
            events: DataFrame with institutional events
        
        Returns:
            Binary signal series
        """
        duration = self.features_config['institutional_validation']['duration']
        
        # Initialize signal
        signal = pd.Series(0, index=dates, name='institutional_signal')
        
        # For each event, set flag for duration days
        for event_date, row in events.iterrows():
            impact = row['impact']
            
            # Define window
            start_date = event_date
            end_date = event_date + pd.Timedelta(days=duration)
            
            # Set flag (positive impact = +1, negative = -1)
            mask = (dates >= start_date) & (dates <= end_date)
            
            if impact == 'positive':
                signal[mask] = 1
            elif impact == 'negative':
                signal[mask] = -1
        
        return signal
    
    def create_interaction_terms(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction terms between signals.
        
        Args:
            features: DataFrame with base features
        
        Returns:
            DataFrame with added interaction terms
        """
        if not self.features_config['interactions']['enabled']:
            return features
        
        # Macro × Risk Premium (compounding stress)
        features['macro_risk_compound'] = (
            features['external_macro'] * features['crypto_risk_premium_5d']
        )
        
        # Macro × Adoption (conflicting forces)
        features['macro_adoption_conflict'] = (
            features['external_macro'] * features['adoption_signal']
        )
        
        return features
    
    def apply_lag(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Lag all features to avoid look-ahead bias.
        
        Args:
            features: DataFrame with features
        
        Returns:
            Lagged features
        """
        lag_days = self.features_config['lag_days']
        
        features_lagged = features.shift(lag_days)
        
        return features_lagged
    
    def winsorize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize features to handle outliers.
        
        Args:
            features: DataFrame with features
        
        Returns:
            Winsorized features
        """
        if not self.features_config['winsorize']['enabled']:
            return features
        
        lower_pct = self.features_config['winsorize']['lower_percentile']
        upper_pct = self.features_config['winsorize']['upper_percentile']
        
        features_winsorized = features.copy()
        
        for col in features.columns:
            lower = features[col].quantile(lower_pct / 100)
            upper = features[col].quantile(upper_pct / 100)
            
            features_winsorized[col] = features[col].clip(lower, upper)
        
        return features_winsorized
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            period: RSI period
        
        Returns:
            RSI values (0-100 scale, normalized to 0-1)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize to 0-1
        rsi_normalized = rsi / 100
        
        return rsi_normalized
    
    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        
        Returns:
            (macd_line, signal_line, histogram)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_momentum_features(self, prices: pd.Series, periods: list = [3, 5, 10]) -> pd.DataFrame:
        """
        Calculate momentum features over multiple periods.
        
        Args:
            prices: Price series
            periods: List of lookback periods
        
        Returns:
            DataFrame with momentum features
        """
        momentum_df = pd.DataFrame(index=prices.index)
        
        for period in periods:
            momentum_df[f'momentum_{period}d'] = prices.pct_change(period)
        
        return momentum_df
    
    def calculate_volatility_features(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate volatility features.
        
        Args:
            prices: Price series
        
        Returns:
            DataFrame with volatility features
        """
        returns = prices.pct_change()
        
        vol_df = pd.DataFrame(index=prices.index)
        vol_df['volatility_20d'] = returns.rolling(20).std()
        vol_df['volatility_5d'] = returns.rolling(5).std()
        
        # Volatility of volatility
        vol_df['vol_of_vol_20d'] = (returns.rolling(20).std()).rolling(20).std()
        
        return vol_df
    
    def engineer_all_features(
        self,
        data: pd.DataFrame,
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create complete feature set.
        
        Args:
            data: Raw data DataFrame
            events: Institutional events DataFrame
        
        Returns:
            Full feature DataFrame
        """
        features = pd.DataFrame(index=data.index)
        
        print("Engineering features...")
        
        # 1. External Macro Signal
        print("  - External macro signal...")
        features['external_macro'] = self.calculate_external_macro_signal(data)
        
        # 2. Crypto Risk Premium Signal (5-day and 20-day)
        print("  - Crypto risk premium signals...")
        features['crypto_risk_premium_5d'] = self.calculate_crypto_risk_premium_signal(
            data, window=5
        )
        features['crypto_risk_premium_20d'] = self.calculate_crypto_risk_premium_signal(
            data, window=20
        )
        
        # 3. Adoption Signal
        print("  - Adoption signal...")
        features['adoption_signal'] = self.calculate_adoption_signal(data)
        
        # 4. Institutional Signal
        print("  - Institutional validation signal...")
        features['institutional_signal'] = self.calculate_institutional_signal(
            data.index, events
        )
        
        # 5. Additional technical features
        print("  - Technical features...")
        
        # BTC returns (various windows)
        features['btc_return_1d'] = data['BTC-USD'].pct_change(1)
        features['btc_return_5d'] = data['BTC-USD'].pct_change(5)
        features['btc_return_20d'] = data['BTC-USD'].pct_change(20)
        
        # BTC volatility
        features['btc_volatility_20d'] = data['BTC-USD'].pct_change().rolling(20).std()
        
        # VIX level (not just change)
        features['vix_level'] = data['VIX']
        
        # Treasury yield level
        features['treasury_level'] = data['DGS2']
        
        # 6. Momentum features (NEW)
        if self.features_config.get('momentum', {}).get('enabled', False):
            print("  - Momentum indicators...")
            
            # RSI
            rsi_period = self.features_config['momentum'].get('rsi_period', 14)
            features[f'rsi_{rsi_period}'] = self.calculate_rsi(data['BTC-USD'], period=rsi_period)
            
            # MACD
            macd_fast = self.features_config['momentum'].get('macd_fast', 12)
            macd_slow = self.features_config['momentum'].get('macd_slow', 26)
            macd_signal = self.features_config['momentum'].get('macd_signal', 9)
            
            macd_line, signal_line, histogram = self.calculate_macd(
                data['BTC-USD'],
                fast=macd_fast,
                slow=macd_slow,
                signal=macd_signal
            )
            features['macd_line'] = macd_line
            features['macd_signal'] = signal_line
            features['macd_histogram'] = histogram
            
            # Momentum features
            momentum_periods = self.features_config['momentum'].get('momentum_periods', [3, 5, 10])
            momentum_feats = self.calculate_momentum_features(data['BTC-USD'], periods=momentum_periods)
            features = features.join(momentum_feats)
            
            # Volatility features
            vol_feats = self.calculate_volatility_features(data['BTC-USD'])
            features = features.join(vol_feats)
        
        # 7. Create interaction terms
        print("  - Interaction terms...")
        features = self.create_interaction_terms(features)
        
        # 7. Apply lag (critical to avoid look-ahead bias)
        print("  - Applying 1-day lag...")
        features = self.apply_lag(features)
        
        # 8. Winsorize
        print("  - Winsorizing outliers...")
        features = self.winsorize_features(features)
        
        # Drop NaN rows
        features = features.dropna()
        
        print(f"Final features shape: {features.shape}")
        
        return features
    
    def create_target_variable(self, data: pd.DataFrame) -> pd.Series:
        """
        Create target variable: next-day or 5-day Bitcoin log return.
        
        Uses config to determine horizon (1-day vs 5-day).
        
        Args:
            data: DataFrame with BTC-USD prices
        
        Returns:
            Log returns series
        """
        btc_prices = data['BTC-USD']
        target_config = self.config['model']['target']
        
        # Determine horizon
        if '5d' in target_config:
            horizon = 5
        else:
            horizon = 1
        
        # Log return = log(P_t+h / P_t)
        log_return = np.log(btc_prices / btc_prices.shift(horizon))
        
        # Shift negative to get future return
        target = log_return.shift(-horizon)
        
        return target.dropna()


# Test code
if __name__ == "__main__":
    from data_acquisition import DataAcquisition
    
    data_acq = DataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset()
    
    engineer = FeatureEngineer('config.yaml')
    
    # Create features
    features = engineer.engineer_all_features(
        dataset['prices'],
        dataset['events']
    )
    
    # Create target
    target = engineer.create_target_variable(dataset['prices'])
    
    print(f"\nFeatures:")
    print(features.head())
    
    print(f"\nTarget (next-day BTC log return):")
    print(target.head())
    
    print(f"\nFeature correlations with target:")
    aligned = features.join(target, rsuffix='_target').dropna()
    correlations = aligned.corr()['BTC-USD'].sort_values(ascending=False)
    print(correlations.head(10))
