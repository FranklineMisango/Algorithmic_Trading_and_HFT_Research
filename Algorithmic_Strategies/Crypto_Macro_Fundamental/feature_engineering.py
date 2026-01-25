"""
Feature Engineering for Crypto Macro-Fundamental Strategy

Implements the 4-factor model: External Macro, Risk Premium, Adoption, and Institutional signals.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Tuple


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
        
        # 6. Create interaction terms
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
        Create target variable: next-day Bitcoin log return.
        
        Args:
            data: DataFrame with BTC-USD prices
        
        Returns:
            Log returns series
        """
        btc_prices = data['BTC-USD']
        
        # Log return = log(P_t+1 / P_t)
        log_return = np.log(btc_prices / btc_prices.shift(1))
        
        # Shift negative to get next-day return
        target = log_return.shift(-1)
        
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
