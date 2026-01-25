# QuantConnect Lean - Rates Carry Strategy

from AlgorithmImports import *
import numpy as np
from collections import deque


class RatesCarryAlgorithm(QCAlgorithm):
    """
    Rates Carry Strategy using Lean
    
    Strategy: Long bonds with positive roll-down yield
    Signals: Z-score of roll-down return
    """
    
    def Initialize(self):
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Bond ETFs as proxies
        self.bonds = {
            'TLT': self.AddEquity("TLT", Resolution.Daily),  # 20+ Year Treasury
            'IEF': self.AddEquity("IEF", Resolution.Daily),  # 7-10 Year Treasury
            'SHY': self.AddEquity("SHY", Resolution.Daily),  # 1-3 Year Treasury
        }
        
        self.zscore_window = 252
        self.entry_threshold = 1.0
        self.exit_threshold = 0.5
        
        self.positions = {}
        self.rolldown_history = {k: deque(maxlen=self.zscore_window) for k in self.bonds.keys()}
        
        self.Schedule.On(
            self.DateRules.EveryDay("TLT"),
            self.TimeRules.AfterMarketOpen("TLT", 30),
            self.Rebalance
        )
    
    def OnData(self, data):
        """Calculate roll-down yield approximation"""
        for symbol_str, symbol_obj in self.bonds.items():
            if data.ContainsKey(symbol_obj.Symbol):
                # Roll-down proxy: recent yield change
                history = self.History(symbol_obj.Symbol, 63, Resolution.Daily)
                if len(history) > 20:
                    returns = history['close'].pct_change().dropna()
                    rolldown = returns.mean() * 252  # Annualized
                    self.rolldown_history[symbol_str].append(rolldown)
    
    def Rebalance(self):
        """Rebalance based on roll-down z-scores"""
        signals = {}
        
        for symbol_str, rolldown in self.rolldown_history.items():
            if len(rolldown) < 60:
                continue
            
            arr = np.array(rolldown)
            mean, std = np.mean(arr), np.std(arr)
            
            if std > 0:
                zscore = (arr[-1] - mean) / std
                
                if zscore > self.entry_threshold:
                    signals[symbol_str] = 1
                elif zscore < -self.entry_threshold:
                    signals[symbol_str] = -1
                else:
                    signals[symbol_str] = 0
        
        # Execute
        for symbol_str, signal in signals.items():
            symbol = self.bonds[symbol_str].Symbol
            if signal != 0:
                self.SetHoldings(symbol, signal * 0.33)
            else:
                self.Liquidate(symbol)
