# QuantConnect Lean - DP Ratio Market Timing

from AlgorithmImports import *
import numpy as np
from collections import deque


class DPRatioTimingAlgorithm(QCAlgorithm):
    """
    Dividend/Price Ratio Market Timing Strategy
    
    Uses D/P ratio as valuation signal for market timing
    """
    
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Market index
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Bonds for defensive allocation
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol
        
        # DP ratio tracking (simplified - would use actual dividend data)
        self.dp_history = deque(maxlen=252)
        
        # Thresholds (percentiles)
        self.cheap_threshold = 0.70  # 70th percentile
        self.expensive_threshold = 0.30  # 30th percentile
        
        self.Schedule.On(
            self.DateRules.MonthStart(self.spy),
            self.TimeRules.At(10, 0),
            self.Rebalance
        )
        
        self.Debug("DP Ratio Timing Algorithm Initialized")
    
    def OnData(self, data):
        """Track DP ratio proxy"""
        if self.spy in data and data[self.spy]:
            # Proxy: inverse of price momentum
            price = data[self.spy].Close
            self.dp_history.append(1.0 / price)
    
    def Rebalance(self):
        """Allocate based on DP ratio signal"""
        if len(self.dp_history) < 60:
            return
        
        current_dp = self.dp_history[-1]
        dp_array = np.array(self.dp_history)
        
        # Calculate percentile
        percentile = np.percentile(dp_array, 50)
        
        # High DP = cheap market = more equity
        if current_dp > percentile:
            self.SetHoldings(self.spy, 0.80)
            self.SetHoldings(self.tlt, 0.20)
            self.Debug("Market cheap: 80/20 stocks/bonds")
        
        # Low DP = expensive market = more bonds
        else:
            self.SetHoldings(self.spy, 0.40)
            self.SetHoldings(self.tlt, 0.60)
            self.Debug("Market expensive: 40/60 stocks/bonds")
