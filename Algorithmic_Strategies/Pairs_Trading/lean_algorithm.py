# QuantConnect Lean - Pairs Trading Strategy

from AlgorithmImports import *
import numpy as np
from collections import deque


class PairsTradingAlgorithm(QCAlgorithm):
    """
    Statistical Arbitrage Pairs Trading using Lean
    
    Identifies cointegrated stock pairs and trades mean reversion
    """
    
    def Initialize(self):
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Pair selection (in practice, test for cointegration)
        # Example: Coca-Cola vs Pepsi
        self.stock1 = self.AddEquity("KO", Resolution.Daily).Symbol
        self.stock2 = self.AddEquity("PEP", Resolution.Daily).Symbol
        
        # Spread tracking
        self.lookback = 60
        self.spread_history = deque(maxlen=self.lookback)
        
        # Trading parameters
        self.entry_zscore = 2.0
        self.exit_zscore = 0.5
        self.position = 0  # 0=flat, 1=long spread, -1=short spread
        
        # Hedge ratio
        self.hedge_ratio = 1.0
        
        self.Debug("Pairs Trading Algorithm Initialized")
    
    def OnData(self, data):
        """Calculate spread and trade"""
        if not (data.ContainsKey(self.stock1) and data.ContainsKey(self.stock2)):
            return
        
        price1 = data[self.stock1].Close
        price2 = data[self.stock2].Close
        
        # Calculate spread
        spread = price1 - self.hedge_ratio * price2
        self.spread_history.append(spread)
        
        if len(self.spread_history) < 20:
            return
        
        # Calculate z-score
        spread_array = np.array(self.spread_history)
        mean = np.mean(spread_array)
        std = np.std(spread_array)
        
        if std == 0:
            return
        
        zscore = (spread - mean) / std
        
        # Trading logic
        if self.position == 0:
            # Enter long spread (buy stock1, sell stock2)
            if zscore < -self.entry_zscore:
                self.SetHoldings(self.stock1, 0.5)
                self.SetHoldings(self.stock2, -0.5)
                self.position = 1
                self.Debug(f"Long spread at z={zscore:.2f}")
            
            # Enter short spread (sell stock1, buy stock2)
            elif zscore > self.entry_zscore:
                self.SetHoldings(self.stock1, -0.5)
                self.SetHoldings(self.stock2, 0.5)
                self.position = -1
                self.Debug(f"Short spread at z={zscore:.2f}")
        
        elif self.position == 1:
            # Exit long spread
            if zscore > -self.exit_zscore:
                self.Liquidate()
                self.position = 0
                self.Debug(f"Exit long at z={zscore:.2f}")
        
        elif self.position == -1:
            # Exit short spread
            if zscore < self.exit_zscore:
                self.Liquidate()
                self.position = 0
                self.Debug(f"Exit short at z={zscore:.2f}")
