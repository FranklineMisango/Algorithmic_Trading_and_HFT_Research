"""
Feature Engineering for Canadian Bond Day Count Arbitrage

Calculates day count conventions, accrued interest differences,
and generates trading signals based on the arbitrage opportunity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yaml
import logging
from pathlib import Path


class DayCountCalculator:
    """
    Calculate accrued interest using different day count conventions.
    
    The arbitrage opportunity arises from the mismatch between:
    1. Actual/Actual (used for quoted prices)
    2. Actual/365 (used for settlement)
    """
    
    @staticmethod
    def actual_actual(coupon_rate: float, 
                     days_accrued: int, 
                     coupon_period_length: int,
                     face_value: float = 100.0) -> float:
        """
        Calculate accrued interest using Actual/Actual convention.
        
        This is the convention used for QUOTED bond prices in Canada.
        
        Args:
            coupon_rate: Annual coupon rate (e.g., 0.03 for 3%)
            days_accrued: Days since last coupon payment
            coupon_period_length: Actual days in current coupon period
            face_value: Bond face value (default 100)
        
        Returns:
            Accrued interest amount
        """
        # Semi-annual coupon payment
        coupon_payment = (coupon_rate / 2) * face_value
        
        # Accrued = Coupon × (Days Accrued / Coupon Period Length)
        accrued_interest = coupon_payment * (days_accrued / coupon_period_length)
        
        return accrued_interest
    
    @staticmethod
    def actual_365(coupon_rate: float,
                  days_accrued: int,
                  face_value: float = 100.0) -> float:
        """
        Calculate accrued interest using Actual/365 convention.
        
        This is the convention used for SETTLEMENT cash flows in Canada.
        
        Args:
            coupon_rate: Annual coupon rate (e.g., 0.03 for 3%)
            days_accrued: Days since last coupon payment
            face_value: Bond face value (default 100)
        
        Returns:
            Accrued interest amount
        """
        # Semi-annual coupon payment
        coupon_payment = (coupon_rate / 2) * face_value
        
        # For settlement, assumes 182.5 days per half-year (365/2)
        # Accrued = Coupon × (Days Accrued / 365) × 2
        accrued_interest = coupon_payment * (days_accrued / 182.5)
        
        return accrued_interest
    
    @staticmethod
    def calculate_arbitrage_profit(coupon_rate: float,
                                  days_accrued: int,
                                  coupon_period_length: int,
                                  face_value: float = 100.0) -> float:
        """
        Calculate theoretical arbitrage profit from day count mismatch.
        
        Profit = Accrued Interest (Settlement) - Accrued Interest (Quoted)
        
        Args:
            coupon_rate: Annual coupon rate
            days_accrued: Days since last coupon
            coupon_period_length: Actual days in current period
            face_value: Bond face value
        
        Returns:
            Arbitrage profit per $100 face value
        """
        ai_actual_actual = DayCountCalculator.actual_actual(
            coupon_rate, days_accrued, coupon_period_length, face_value
        )
        
        ai_actual_365 = DayCountCalculator.actual_365(
            coupon_rate, days_accrued, face_value
        )
        
        # The arbitrage: you pay based on Actual/Actual but receive based on Actual/365
        arbitrage_profit = ai_actual_365 - ai_actual_actual
        
        return arbitrage_profit


class CanadianBondFeatureEngineering:
    """
    Feature engineering for Canadian Bond day count arbitrage strategy.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize feature engineering with configuration."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Signal parameters
        self.target_periods = self.config['signals']['target_coupon_periods']
        self.entry_window = self.config['signals']['entry_window_days']
        self.min_entry_days = self.config['signals']['min_entry_days']
        self.min_profit_bps = self.config['signals']['min_profit_bps']
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def calculate_coupon_period_length(self, 
                                      prev_coupon_date: datetime,
                                      next_coupon_date: datetime) -> int:
        """
        Calculate the number of calendar days in the coupon period.
        
        Args:
            prev_coupon_date: Previous coupon payment date
            next_coupon_date: Next coupon payment date
        
        Returns:
            Number of calendar days
        """
        delta = next_coupon_date - prev_coupon_date
        return delta.days
    
    def calculate_days_to_next_coupon(self,
                                     current_date: datetime,
                                     next_coupon_date: datetime) -> int:
        """
        Calculate days remaining until next coupon payment.
        
        Args:
            current_date: Current/evaluation date
            next_coupon_date: Next coupon payment date
        
        Returns:
            Days to next coupon
        """
        delta = next_coupon_date - current_date
        return delta.days
    
    def calculate_days_since_last_coupon(self,
                                        current_date: datetime,
                                        prev_coupon_date: datetime) -> int:
        """
        Calculate days elapsed since last coupon payment.
        
        Args:
            current_date: Current/evaluation date
            prev_coupon_date: Previous coupon payment date
        
        Returns:
            Days since last coupon
        """
        delta = current_date - prev_coupon_date
        return delta.days
    
    def engineer_features(self, bond_data: pd.DataFrame, 
                         as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Create all features needed for signal generation.
        
        Args:
            bond_data: DataFrame with bond details from data acquisition
            as_of_date: Evaluation date (default: today)
        
        Returns:
            DataFrame with engineered features
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        df = bond_data.copy()
        
        # Ensure datetime types
        df['NXT_CPN_DT'] = pd.to_datetime(df['NXT_CPN_DT'])
        df['PREV_CPN_DT'] = pd.to_datetime(df['PREV_CPN_DT'])
        
        # Calculate temporal features
        df['days_to_next_coupon'] = df['NXT_CPN_DT'].apply(
            lambda x: self.calculate_days_to_next_coupon(as_of_date, x)
        )
        
        df['days_since_last_coupon'] = df['PREV_CPN_DT'].apply(
            lambda x: self.calculate_days_since_last_coupon(as_of_date, x)
        )
        
        df['coupon_period_length'] = df.apply(
            lambda row: self.calculate_coupon_period_length(
                row['PREV_CPN_DT'], row['NXT_CPN_DT']
            ),
            axis=1
        )
        
        # Calculate day count differences
        df['accrued_interest_actual_actual'] = df.apply(
            lambda row: DayCountCalculator.actual_actual(
                row['CPN'] / 100,  # Convert from percentage
                row['days_since_last_coupon'],
                row['coupon_period_length']
            ),
            axis=1
        )
        
        df['accrued_interest_actual_365'] = df.apply(
            lambda row: DayCountCalculator.actual_365(
                row['CPN'] / 100,
                row['days_since_last_coupon']
            ),
            axis=1
        )
        
        df['arbitrage_profit_per_100'] = df.apply(
            lambda row: DayCountCalculator.calculate_arbitrage_profit(
                row['CPN'] / 100,
                row['days_since_last_coupon'],
                row['coupon_period_length']
            ),
            axis=1
        )
        
        # Convert to basis points for easier interpretation
        df['arbitrage_profit_bps'] = (df['arbitrage_profit_per_100'] / 
                                      df['PX_CLEAN']) * 10000
        
        # Flag target coupon periods (181 or 182 days)
        df['is_target_period'] = df['coupon_period_length'].isin(self.target_periods)
        
        # Flag bonds in entry window
        df['in_entry_window'] = (
            (df['days_to_next_coupon'] <= self.entry_window) &
            (df['days_to_next_coupon'] >= self.min_entry_days)
        )
        
        self.logger.info(f"Engineered features for {len(df)} bonds")
        
        return df
    
    def generate_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on engineered features.
        
        Args:
            features_df: DataFrame with engineered features
        
        Returns:
            DataFrame with signals and signal strength
        """
        df = features_df.copy()
        
        # Initialize signal column
        df['signal'] = 0  # 0 = no signal, 1 = buy signal
        df['signal_strength'] = 0.0
        
        # Generate buy signals
        buy_conditions = (
            df['is_target_period'] &  # Coupon period is 181 or 182 days
            df['in_entry_window'] &   # Within entry window
            (df['arbitrage_profit_bps'] >= self.min_profit_bps)  # Minimum profit threshold
        )
        
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[buy_conditions, 'signal_strength'] = df.loc[buy_conditions, 'arbitrage_profit_bps']
        
        # Add risk flags
        df['high_risk_182_day'] = (df['coupon_period_length'] == 182)  # Riskier edge case
        
        # Log signal generation
        num_signals = df['signal'].sum()
        self.logger.info(f"Generated {num_signals} buy signals")
        
        if num_signals > 0:
            avg_profit = df[df['signal'] == 1]['arbitrage_profit_bps'].mean()
            self.logger.info(f"Average expected profit: {avg_profit:.2f} bps")
        
        return df
    
    def calculate_pv01(self, clean_price: float, 
                      modified_duration: float,
                      face_value: float = 100.0) -> float:
        """
        Calculate PV01 (price value of a basis point) for duration hedging.
        
        Args:
            clean_price: Clean price of the bond
            modified_duration: Modified duration
            face_value: Face value
        
        Returns:
            PV01 value
        """
        # PV01 = (Clean Price / 100) × Modified Duration × 0.0001 × Face Value
        pv01 = (clean_price / 100) * modified_duration * 0.0001 * face_value
        return pv01
    
    def create_duration_hedge(self, 
                            target_bond: pd.Series,
                            hedge_universe: pd.DataFrame) -> Dict:
        """
        Create a duration-matched hedge for the arbitrage position.
        
        Args:
            target_bond: Series with target bond details
            hedge_universe: DataFrame with potential hedge bonds
        
        Returns:
            Dictionary with hedge composition
        """
        # Calculate target PV01
        target_pv01 = self.calculate_pv01(
            target_bond['PX_CLEAN'],
            target_bond['DUR_ADJ_MID']
        )
        
        # Remove the target bond from hedge universe
        hedge_universe = hedge_universe[
            hedge_universe['identifier'] != target_bond['identifier']
        ]
        
        # Calculate PV01 for all hedge candidates
        hedge_universe['pv01'] = hedge_universe.apply(
            lambda row: self.calculate_pv01(row['PX_CLEAN'], row['DUR_ADJ_MID']),
            axis=1
        )
        
        # Simple hedge: select bonds closest in duration
        # Sort by duration similarity
        hedge_universe['duration_diff'] = abs(
            hedge_universe['DUR_ADJ_MID'] - target_bond['DUR_ADJ_MID']
        )
        hedge_universe = hedge_universe.sort_values('duration_diff')
        
        # Select top N bonds for diversification
        min_hedge_bonds = self.config['hedging']['min_hedge_bonds']
        hedge_bonds = hedge_universe.head(min_hedge_bonds)
        
        # Calculate hedge weights (simple equal-weight for now)
        # In production, use optimization to match PV01 exactly
        hedge_weights = {}
        total_hedge_pv01 = 0
        
        for _, bond in hedge_bonds.iterrows():
            weight = 1 / min_hedge_bonds
            hedge_weights[bond['identifier']] = {
                'weight': weight,
                'pv01': bond['pv01'],
                'duration': bond['DUR_ADJ_MID'],
                'price': bond['PX_CLEAN']
            }
            total_hedge_pv01 += bond['pv01'] * weight
        
        # Scaling factor to match target PV01
        scale_factor = target_pv01 / total_hedge_pv01
        
        for bond_id in hedge_weights:
            hedge_weights[bond_id]['notional'] = (
                hedge_weights[bond_id]['weight'] * scale_factor * 100
            )
        
        hedge = {
            'target_pv01': target_pv01,
            'hedge_pv01': total_hedge_pv01 * scale_factor,
            'hedge_composition': hedge_weights,
            'tracking_error': abs(target_pv01 - total_hedge_pv01 * scale_factor)
        }
        
        self.logger.info(f"Created duration hedge with tracking error: {hedge['tracking_error']:.6f}")
        
        return hedge
    
    def validate_signal_quality(self, signals_df: pd.DataFrame) -> Dict:
        """
        Validate the quality and characteristics of generated signals.
        
        Args:
            signals_df: DataFrame with signals
        
        Returns:
            Dictionary with validation metrics
        """
        signals = signals_df[signals_df['signal'] == 1]
        
        if len(signals) == 0:
            return {
                'num_signals': 0,
                'avg_profit_bps': 0,
                'max_profit_bps': 0,
                'min_profit_bps': 0,
                'pct_high_risk': 0
            }
        
        validation = {
            'num_signals': len(signals),
            'avg_profit_bps': signals['arbitrage_profit_bps'].mean(),
            'median_profit_bps': signals['arbitrage_profit_bps'].median(),
            'max_profit_bps': signals['arbitrage_profit_bps'].max(),
            'min_profit_bps': signals['arbitrage_profit_bps'].min(),
            'pct_high_risk': (signals['high_risk_182_day'].sum() / len(signals)) * 100,
            'avg_days_to_coupon': signals['days_to_next_coupon'].mean(),
            'coupon_period_distribution': signals['coupon_period_length'].value_counts().to_dict()
        }
        
        return validation


if __name__ == "__main__":
    # Example usage
    from data_acquisition import CanadianBondDataAcquisition
    
    # Acquire bond data
    acquirer = CanadianBondDataAcquisition()
    bonds = acquirer.get_canadian_government_bonds()
    
    if len(bonds) > 0:
        # Fetch detailed data
        bond_details = acquirer.get_bond_details(bonds['identifier'].tolist())
        
        # Engineer features
        fe = CanadianBondFeatureEngineering()
        features = fe.engineer_features(bond_details)
        
        # Generate signals
        signals = fe.generate_signals(features)
        
        # Validate
        validation = fe.validate_signal_quality(signals)
        
        print("\n=== Signal Validation ===")
        for key, value in validation.items():
            print(f"{key}: {value}")
        
        # Display signals
        if validation['num_signals'] > 0:
            print("\n=== Active Signals ===")
            signal_cols = [
                'identifier', 'coupon_period_length', 'days_to_next_coupon',
                'arbitrage_profit_bps', 'signal_strength'
            ]
            print(signals[signals['signal'] == 1][signal_cols])
