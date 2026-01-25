# QuantConnect Lean - Leveraged Index Funds Strategy

from AlgorithmImports import *
import numpy as np
from collections import deque


class LeveragedIndexAlgorithm(QCAlgorithm):
    """
    Leveraged ETF Decay Strategy
    
    Exploits volatility decay in leveraged ETFs
    """
    
    def Initialize(self):
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Leveraged pairs
        self.spxl = self.AddEquity("SPXL", Resolution.Daily).Symbol  # 3x Bull S&P
        self.spxs = self.AddEquity("SPXS", Resolution.Daily).Symbol  # 3x Bear S&P
        
        # Underlying for volatility
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Volatility tracking
        self.vol_lookback = 20
        self.spy_history = deque(maxlen=self.vol_lookback)
        
        # High vol threshold for decay exploitation
        self.high_vol_threshold = 0.30  # 30% annualized
        
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 30),
            self.Rebalance
        )
        
        self.Debug("Leveraged Index Strategy Initialized")
    
    def OnData(self, data):
        """Track underlying volatility"""
        if self.spy in data and data[self.spy]:
            self.spy_history.append(data[self.spy].Close)
    
    def Rebalance(self):
        """Trade based on volatility regime"""
        if len(self.spy_history) < 10:
            return
        
        # Calculate realized volatility
        prices = np.array(self.spy_history)
        returns = np.diff(prices) / prices[:-1]
        vol = np.std(returns) * np.sqrt(252)
        
        # High volatility = short both leveraged ETFs (decay strategy)
        if vol > self.high_vol_threshold:
            self.SetHoldings(self.spxl, -0.25)
            self.SetHoldings(self.spxs, -0.25)
            self.Debug(f"High vol ({vol:.2%}): Short both leveraged ETFs")
        
        # Low volatility = liquidate
        else:
            self.Liquidate()
