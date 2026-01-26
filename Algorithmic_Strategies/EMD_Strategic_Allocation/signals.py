import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class EMDSignals:
    def __init__(self, ppp_window_years=10, ppp_threshold=-0.5, spread_percentile=60):
        self.ppp_window = ppp_window_years * 252
        self.ppp_threshold = ppp_threshold
        self.spread_percentile = spread_percentile
    
    def calculate_ppp_zscore(self, ppp_values: pd.Series) -> pd.Series:
        """Calculate PPP valuation Z-score with rolling window"""
        rolling_mean = ppp_values.rolling(window=self.ppp_window, min_periods=252).mean()
        rolling_std = ppp_values.rolling(window=self.ppp_window, min_periods=252).std()
        zscore = (ppp_values - rolling_mean) / rolling_std
        return zscore
    
    def get_undervalued_currencies(self, ppp_zscores: Dict[str, float]) -> List[str]:
        """Return list of undervalued currencies (Z < threshold)"""
        return [country for country, z in ppp_zscores.items() if z < self.ppp_threshold]
    
    def calculate_yield_spread(self, em_yield: float, dm_yield: float) -> float:
        """Calculate yield spread in basis points"""
        return (em_yield - dm_yield) * 10000
    
    def rank_by_spread(self, spreads: Dict[str, float]) -> Dict[str, float]:
        """Rank countries by spread percentile"""
        if not spreads:
            return {}
        spread_values = list(spreads.values())
        threshold = np.percentile(spread_values, self.spread_percentile)
        return {k: v for k, v in spreads.items() if v >= threshold}
    
    def apply_diversification_cap(self, weights: Dict[str, float], max_weight=0.05) -> Dict[str, float]:
        """Cap individual country weights"""
        total = sum(weights.values())
        if total == 0:
            return weights
        
        normalized = {k: v/total for k, v in weights.items()}
        capped = {k: min(v, max_weight) for k, v in normalized.items()}
        
        # Renormalize after capping
        capped_total = sum(capped.values())
        if capped_total > 0:
            return {k: v/capped_total for k, v in capped.items()}
        return capped
    
    def calculate_real_yield(self, nominal_yield: float, expected_inflation: float) -> float:
        """Calculate real yield"""
        return nominal_yield - expected_inflation
    
    def calculate_hedge_cost(self, domestic_rate: float, foreign_rate: float, time_years=0.25) -> float:
        """Calculate quarterly hedge cost"""
        return (domestic_rate - foreign_rate) * time_years
