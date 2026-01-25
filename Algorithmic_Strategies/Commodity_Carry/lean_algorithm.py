# QuantConnect Lean - Commodity Carry Strategy

from AlgorithmImports import *
import numpy as np
from collections import deque


class CommodityCarryAlgorithm(QCAlgorithm):
    """
    Commodity Carry Strategy using Lean
    
    Strategy: Long backwardated commodities, short contango
    Signals: Z-score of convenience yield (futures curve slope)
    """
    
    def Initialize(self):
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Commodity ETFs
        self.commodities = {
            'USO': self.AddEquity("USO", Resolution.Daily),   # Oil
            'GLD': self.AddEquity("GLD", Resolution.Daily),   # Gold
            'SLV': self.AddEquity("SLV", Resolution.Daily),   # Silver
            'DBA': self.AddEquity("DBA", Resolution.Daily),   # Agriculture
        }
        
        self.zscore_window = 252
        self.entry_threshold = 1.0
        self.exit_threshold = 0.5
        
        self.momentum_history = {k: deque(maxlen=self.zscore_window) for k in self.commodities.keys()}
        
        self.Schedule.On(
            self.DateRules.EveryDay("USO"),
            self.TimeRules.AfterMarketOpen("USO", 30),
            self.Rebalance
        )
    
    def OnData(self, data):
        """Track momentum as convenience yield proxy"""
        for symbol_str, symbol_obj in self.commodities.items():
            if data.ContainsKey(symbol_obj.Symbol):
                # Get 3-month momentum
                history = self.History(symbol_obj.Symbol, 63, Resolution.Daily)
                if len(history) > 10:
                    momentum = (history['close'].iloc[-1] / history['close'].iloc[0]) - 1
                    self.momentum_history[symbol_str].append(momentum)
    
    def Rebalance(self):
        """Rebalance based on convenience yield signals"""
        signals = {}
        
        for symbol_str, momentum in self.momentum_history.items():
            if len(momentum) < 60:
                continue
            
            arr = np.array(momentum)
            mean, std = np.mean(arr), np.std(arr)
            
            if std > 0:
                zscore = (arr[-1] - mean) / std
                
                # Positive momentum = backwardation
                if zscore > self.entry_threshold:
                    signals[symbol_str] = 1
                elif zscore < -self.entry_threshold:
                    signals[symbol_str] = -1
                else:
                    signals[symbol_str] = 0
        
        # Execute
        for symbol_str, signal in signals.items():
            symbol = self.commodities[symbol_str].Symbol
            if signal != 0:
                self.SetHoldings(symbol, signal * 0.25)
            else:
                self.Liquidate(symbol)
