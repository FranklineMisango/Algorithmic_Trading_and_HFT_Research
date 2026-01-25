# QuantConnect Lean - Credit Carry Strategy

from AlgorithmImports import *
import numpy as np
from collections import deque


class CreditCarryAlgorithm(QCAlgorithm):
    """
    Credit Carry Strategy using Lean
    
    Strategy: Sell protection on low-spread credits, buy on high spreads
    Signals: Z-score of credit spreads
    """
    
    def Initialize(self):
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Credit ETFs
        self.credits = {
            'LQD': self.AddEquity("LQD", Resolution.Daily),  # Investment Grade
            'HYG': self.AddEquity("HYG", Resolution.Daily),  # High Yield
            'EMB': self.AddEquity("EMB", Resolution.Daily),  # Emerging Markets
        }
        
        self.zscore_window = 252
        self.entry_threshold = 1.0
        self.exit_threshold = 0.5
        
        self.spread_history = {k: deque(maxlen=self.zscore_window) for k in self.credits.keys()}
        
        self.Schedule.On(
            self.DateRules.EveryDay("LQD"),
            self.TimeRules.AfterMarketOpen("LQD", 30),
            self.Rebalance
        )
    
    def OnData(self, data):
        """Track spread proxy (inverse of price)"""
        for symbol_str, symbol_obj in self.credits.items():
            if data.ContainsKey(symbol_obj.Symbol):
                price = data[symbol_obj.Symbol].Close
                # Spread proxy: normalize by history
                self.spread_history[symbol_str].append(1.0 / price if price > 0 else 0)
    
    def Rebalance(self):
        """Rebalance based on spread z-scores"""
        signals = {}
        
        for symbol_str, spreads in self.spread_history.items():
            if len(spreads) < 60:
                continue
            
            arr = np.array(spreads)
            mean, std = np.mean(arr), np.std(arr)
            
            if std > 0:
                zscore = (arr[-1] - mean) / std
                
                # High spread = sell protection (long)
                if zscore > self.entry_threshold:
                    signals[symbol_str] = -1
                elif zscore < -self.entry_threshold:
                    signals[symbol_str] = 1
                else:
                    signals[symbol_str] = 0
        
        # Execute
        for symbol_str, signal in signals.items():
            symbol = self.credits[symbol_str].Symbol
            if signal != 0:
                self.SetHoldings(symbol, signal * 0.33)
            else:
                self.Liquidate(symbol)
