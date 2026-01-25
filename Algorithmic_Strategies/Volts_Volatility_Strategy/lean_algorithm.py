# QuantConnect Lean - Volatility Strategy

from AlgorithmImports import *
import numpy as np
from collections import deque


class VolatilityStrategyAlgorithm(QCAlgorithm):
    """
    Volatility Trading Strategy (VIX-based)
    
    Trades volatility regime changes and VIX mean reversion
    """
    
    def Initialize(self):
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # VIX and volatility products
        self.vix = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        
        # Volatility ETPs
        self.vxx = self.AddEquity("VXX", Resolution.Daily).Symbol  # Short-term VIX futures
        self.svxy = self.AddEquity("SVXY", Resolution.Daily).Symbol  # Short VIX
        
        # Equity for regime detection
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # VIX tracking
        self.vix_lookback = 60
        self.vix_history = deque(maxlen=self.vix_lookback)
        
        # Thresholds
        self.high_vol_threshold = 25
        self.low_vol_threshold = 15
        
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.At(10, 0),
            self.Rebalance
        )
        
        self.Debug("Volatility Strategy Algorithm Initialized")
    
    def OnData(self, data):
        """Track VIX levels"""
        if self.vix in data and data[self.vix]:
            self.vix_history.append(data[self.vix].Value)
    
    def Rebalance(self):
        """Trade based on volatility regime"""
        if len(self.vix_history) < 20:
            return
        
        current_vix = self.vix_history[-1]
        vix_mean = np.mean(self.vix_history)
        vix_std = np.std(self.vix_history)
        
        # Z-score
        zscore = (current_vix - vix_mean) / vix_std if vix_std > 0 else 0
        
        # High volatility regime - short volatility (mean reversion)
        if current_vix > self.high_vol_threshold and zscore > 1.5:
            self.SetHoldings(self.svxy, 0.30)  # Short VIX
            self.Liquidate(self.vxx)
            self.Debug(f"High vol regime: VIX={current_vix:.1f}, shorting volatility")
        
        # Low volatility regime - long volatility (tail risk hedge)
        elif current_vix < self.low_vol_threshold:
            self.SetHoldings(self.vxx, 0.20)  # Long VIX
            self.Liquidate(self.svxy)
            self.Debug(f"Low vol regime: VIX={current_vix:.1f}, buying volatility")
        
        # Mean reversion to normal
        elif abs(zscore) < 0.5:
            self.Liquidate()
