# QuantConnect Lean Engine - FX Carry Strategy
# Event-driven backtesting implementation

from AlgorithmImports import *
import numpy as np
from collections import deque


class FXCarryAlgorithm(QCAlgorithm):
    """
    FX Carry Strategy using QuantConnect Lean
    
    Strategy: Go long high-yield currencies, short low-yield currencies
    Signals: Z-score of interest rate differential
    Rebalancing: Weekly
    """
    
    def Initialize(self):
        """Initialize algorithm parameters and data"""
        self.SetStartDate(2005, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Strategy parameters
        self.zscore_window = 252
        self.entry_threshold = 1.0
        self.exit_threshold = 0.5
        self.vol_lookback = 63
        self.target_vol = 0.10
        self.rebalance_days = 7
        
        # Currency pairs
        self.pairs = {
            'AUDJPY': (self.AddForex("AUDJPY", Resolution.Daily, Market.Oanda), None, deque(maxlen=self.zscore_window)),
            'NZDJPY': (self.AddForex("NZDJPY", Resolution.Daily, Market.Oanda), None, deque(maxlen=self.zscore_window)),
            'EURUSD': (self.AddForex("EURUSD", Resolution.Daily, Market.Oanda), None, deque(maxlen=self.zscore_window)),
            'GBPUSD': (self.AddForex("GBPUSD", Resolution.Daily, Market.Oanda), None, deque(maxlen=self.zscore_window)),
            'USDJPY': (self.AddForex("USDJPY", Resolution.Daily, Market.Oanda), None, deque(maxlen=self.zscore_window)),
        }
        
        # Interest rates (simplified - would need actual rate feeds)
        self.interest_rates = {
            'AUD': 0.025,
            'NZD': 0.020,
            'EUR': 0.000,
            'GBP': 0.005,
            'USD': 0.015,
            'JPY': -0.001,
        }
        
        # Position tracking
        self.positions = {}
        self.last_rebalance = self.Time
        
        # Performance tracking
        self.returns_buffer = deque(maxlen=self.vol_lookback)
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Friday),
            self.TimeRules.At(16, 0),
            self.Rebalance
        )
        
        self.Debug("FX Carry Algorithm Initialized")
    
    def OnData(self, data):
        """Event handler for new data"""
        # Update carry calculations
        for symbol_str, (symbol, position, carry_history) in self.pairs.items():
            if data.ContainsKey(symbol.Symbol):
                price = data[symbol.Symbol].Close
                
                # Calculate carry (interest rate differential)
                base_curr = symbol_str[:3]
                quote_curr = symbol_str[3:]
                carry = self.interest_rates.get(base_curr, 0) - self.interest_rates.get(quote_curr, 0)
                
                carry_history.append(carry)
                self.pairs[symbol_str] = (symbol, position, carry_history)
    
    def Rebalance(self):
        """Weekly rebalancing logic"""
        if (self.Time - self.last_rebalance).days < self.rebalance_days:
            return
        
        self.Debug(f"Rebalancing on {self.Time}")
        
        # Calculate z-scores for each pair
        signals = {}
        
        for symbol_str, (symbol, position, carry_history) in self.pairs.items():
            if len(carry_history) < 60:
                continue
            
            carry_array = np.array(carry_history)
            mean = np.mean(carry_array)
            std = np.std(carry_array)
            
            if std > 0:
                current_carry = carry_array[-1]
                zscore = (current_carry - mean) / std
                
                # Generate signal
                if position is None or position == 0:
                    if zscore > self.entry_threshold:
                        signals[symbol] = 1  # Long
                    elif zscore < -self.entry_threshold:
                        signals[symbol] = -1  # Short
                    else:
                        signals[symbol] = 0
                elif position == 1:
                    if zscore < self.exit_threshold:
                        signals[symbol] = 0
                    else:
                        signals[symbol] = 1
                elif position == -1:
                    if zscore > -self.exit_threshold:
                        signals[symbol] = 0
                    else:
                        signals[symbol] = -1
                
                self.pairs[symbol_str] = (symbol, signals.get(symbol, 0), carry_history)
        
        # Calculate inverse volatility weights
        weights = self.CalculateWeights(signals)
        
        # Execute trades
        for symbol, weight in weights.items():
            if weight != 0:
                self.SetHoldings(symbol.Symbol, weight)
            else:
                self.Liquidate(symbol.Symbol)
        
        self.last_rebalance = self.Time
    
    def CalculateWeights(self, signals):
        """Calculate position weights using inverse volatility"""
        if not signals:
            return {}
        
        # Simplified: equal weight for now
        # In practice, calculate volatility for each pair
        long_count = sum(1 for s in signals.values() if s == 1)
        short_count = sum(1 for s in signals.values() if s == -1)
        
        weights = {}
        
        for symbol, signal in signals.items():
            if signal == 1 and long_count > 0:
                weights[symbol] = 1.0 / long_count
            elif signal == -1 and short_count > 0:
                weights[symbol] = -1.0 / short_count
            else:
                weights[symbol] = 0
        
        return weights
    
    def OnEndOfAlgorithm(self):
        """Called when algorithm ends"""
        self.Debug(f"Algorithm ended. Final portfolio value: ${self.Portfolio.TotalPortfolioValue}")
