import numpy as np
import pandas as pd
from typing import Dict, Optional

class RiskManager:
    def __init__(self, config: dict):
        self.max_country_weight = config['risk']['max_country_weight']
        self.min_credit_rating = config['risk']['min_credit_rating']
        self.fx_vol_trigger = config['risk']['fx_vol_hedge_trigger']
        self.hedge_ratio = config['risk']['hedge_ratio']
        self.vix_trigger = config['risk']['vix_reduction_trigger']
        self.max_correlation = config['risk']['max_correlation_alert']
        self.min_spread = config['risk']['min_spread_alert']
        
        self.rating_map = {'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4, 'A+': 5, 'A': 6, 'A-': 7,
                          'BBB+': 8, 'BBB': 9, 'BBB-': 10, 'BB+': 11, 'BB': 12, 'BB-': 13,
                          'B+': 14, 'B': 15, 'B-': 16, 'CCC+': 17, 'CCC': 18, 'CCC-': 19}
    
    def filter_by_credit_rating(self, countries: Dict[str, str]) -> Dict[str, str]:
        """Filter countries below minimum credit rating"""
        min_score = self.rating_map.get(self.min_credit_rating, 16)
        return {k: v for k, v in countries.items() 
                if self.rating_map.get(v, 20) <= min_score}
    
    def calculate_fx_volatility(self, fx_returns: pd.Series, window=21) -> float:
        """Calculate annualized FX volatility"""
        return fx_returns.rolling(window).std() * np.sqrt(252)
    
    def should_hedge_currency(self, fx_vol: float) -> bool:
        """Determine if currency hedging is needed"""
        return fx_vol > self.fx_vol_trigger
    
    def calculate_hedge_amount(self, position_size: float, should_hedge: bool) -> float:
        """Calculate hedge amount"""
        return position_size * self.hedge_ratio if should_hedge else 0.0
    
    def check_vix_trigger(self, vix_level: float) -> bool:
        """Check if VIX trigger for allocation reduction is hit"""
        return vix_level > self.vix_trigger
    
    def calculate_rolling_correlation(self, returns1: pd.Series, returns2: pd.Series, 
                                     window=90) -> pd.Series:
        """Calculate rolling correlation"""
        return returns1.rolling(window).corr(returns2)
    
    def check_correlation_alert(self, correlation: float) -> bool:
        """Alert if correlation exceeds threshold"""
        return correlation > self.max_correlation
    
    def check_spread_compression(self, avg_spread: float) -> bool:
        """Alert if spreads compress below threshold"""
        return avg_spread < self.min_spread
    
    def apply_stress_test(self, portfolio_value: float, scenario: str) -> Dict[str, float]:
        """Apply stress test scenarios"""
        scenarios = {
            'taper_tantrum': {'us_yield_shock': 1.0, 'usd_strength': 0.10},
            'covid_crash': {'equity_drawdown': -0.30, 'oil_crash': -0.50}
        }
        
        if scenario not in scenarios:
            return {'portfolio_value': portfolio_value}
        
        params = scenarios[scenario]
        
        if scenario == 'taper_tantrum':
            # Simplified: assume 10% loss on local currency, 5% on hard currency
            stressed_value = portfolio_value * 0.93
        elif scenario == 'covid_crash':
            # Simplified: assume 15% loss
            stressed_value = portfolio_value * 0.85
        else:
            stressed_value = portfolio_value
        
        return {
            'portfolio_value': stressed_value,
            'loss_pct': (stressed_value - portfolio_value) / portfolio_value,
            'scenario': scenario
        }
