"""
Simplified VOLTS Signal Generator

Based on paper's actual results:
- Only LONG positions (no shorting)
- Entry: When predictor volatility rises AND target price rises
- Exit: When predictor volatility falls OR target price falls
- More conservative to avoid overtrading
"""

import numpy as np
import pandas as pd
from typing import Dict
from signal_generator import SignalGenerator, Signal, TrendMethod


class SimpleVoltsSignalGenerator(SignalGenerator):
    """
    Simplified signal generator that only takes long positions.
    """
    
    def generate_signals_for_pair(
        self,
        predictor_volatility: pd.Series,
        target_price: pd.Series,
        target_ticker: str
    ) -> pd.DataFrame:
        """
        Generate LONG-ONLY signals.
        
        Strategy:
        - BUY: When predictor volatility trending UP AND target price trending UP
        - HOLD/EXIT: Otherwise (don't short)
        
        Parameters:
        -----------
        predictor_volatility : pd.Series
            Volatility time series of predictor stock
        target_price : pd.Series
            Price time series of target stock
        target_ticker : str
            Ticker symbol of target stock
            
        Returns:
        --------
        pd.DataFrame : DataFrame with dates, trends, and signals
        """
        # Detect trend in predictor's volatility
        vol_trend = self.detect_trend(predictor_volatility)
        
        # Detect trend in target's price
        price_trend = self.detect_trend(target_price)
        
        # Align the series
        common_idx = vol_trend.index.intersection(price_trend.index)
        vol_trend = vol_trend.loc[common_idx]
        price_trend = price_trend.loc[common_idx]
        
        # Generate LONG-ONLY signals
        signals = pd.Series(Signal.HOLD.value, index=common_idx)
        
        # BUY: ONLY when both volatility AND price are rising
        buy_condition = (vol_trend == 1) & (price_trend == 1)
        signals[buy_condition] = Signal.BUY.value
        
        # Everything else is HOLD (exit position or stay out)
        # No SELL signals (no shorting)
        
        result = pd.DataFrame({
            'date': common_idx,
            'predictor_volatility': predictor_volatility.loc[common_idx].values,
            'target_price': target_price.loc[common_idx].values,
            'vol_trend': vol_trend.values,
            'price_trend': price_trend.values,
            'signal': signals.values,
            'target': target_ticker
        })
        
        result.set_index('date', inplace=True)
        
        return result
    
    def generate_signals_for_all_pairs(
        self,
        volatility_df: pd.DataFrame,
        trading_pairs: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate LONG-ONLY signals for all trading pairs.
        """
        all_signals = {}
        
        for _, row in trading_pairs.iterrows():
            predictor = row['predictor']
            target = row['target']
            pair_name = f"{predictor}->{target}"
            
            if target not in price_data:
                print(f"Warning: No price data for {target}, skipping {pair_name}")
                continue
            
            target_price = price_data[target]['Close']
            
            signals = self.generate_signals_for_pair(
                volatility_df[predictor],
                target_price,
                target
            )
            
            all_signals[pair_name] = signals
        
        return all_signals
