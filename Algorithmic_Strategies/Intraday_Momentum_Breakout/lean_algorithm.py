# QuantConnect Lean - Intraday Momentum Breakout

from AlgorithmImports import *
from collections import deque
import numpy as np


class IntradayMomentumAlgorithm(QCAlgorithm):
    """
    Intraday Momentum Breakout Strategy
    
    Trades breakouts from opening range with volume confirmation
    """
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Universe of liquid stocks
        self.universe_settings.resolution = Resolution.Minute
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Opening range parameters
        self.opening_range_minutes = 30
        self.breakout_threshold = 0.005  # 0.5% above/below range
        
        # Track opening ranges
        self.opening_ranges = {}  # symbol -> (high, low, volume)
        self.positions_today = set()
        
        # Schedule events
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 1),
            self.RecordMarketOpen
        )
        
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.BeforeMarketClose("SPY", 5),
            self.CloseAllPositions
        )
        
        self.Debug("Intraday Momentum Algorithm Initialized")
    
    def CoarseSelectionFunction(self, coarse):
        """Select top liquid stocks"""
        filtered = [x for x in coarse if x.Price > 10 and x.DollarVolume > 50000000]
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:100]]
    
    def RecordMarketOpen(self):
        """Reset daily tracking"""
        self.opening_ranges.clear()
        self.positions_today.clear()
    
    def OnData(self, data):
        """Track opening range and execute breakouts"""
        current_time = self.Time.time()
        market_open = time(9, 30)
        opening_end = time(10, 0)
        
        # Record opening range
        if market_open <= current_time <= opening_end:
            for symbol in data.Keys:
                if symbol not in self.opening_ranges:
                    self.opening_ranges[symbol] = {
                        'high': data[symbol].High,
                        'low': data[symbol].Low,
                        'volume': data[symbol].Volume
                    }
                else:
                    self.opening_ranges[symbol]['high'] = max(
                        self.opening_ranges[symbol]['high'],
                        data[symbol].High
                    )
                    self.opening_ranges[symbol]['low'] = min(
                        self.opening_ranges[symbol]['low'],
                        data[symbol].Low
                    )
                    self.opening_ranges[symbol]['volume'] += data[symbol].Volume
        
        # Trade breakouts after opening range
        elif current_time > opening_end:
            for symbol in data.Keys:
                if symbol not in self.opening_ranges:
                    continue
                
                if symbol in self.positions_today:
                    continue
                
                opening = self.opening_ranges[symbol]
                current_price = data[symbol].Close
                
                # Breakout above
                if current_price > opening['high'] * (1 + self.breakout_threshold):
                    # Volume confirmation
                    if data[symbol].Volume > opening['volume'] * 0.5:
                        self.SetHoldings(symbol, 0.05)  # Small position
                        self.positions_today.add(symbol)
                        self.Debug(f"Long breakout: {symbol} at {current_price}")
                
                # Breakdown below
                elif current_price < opening['low'] * (1 - self.breakout_threshold):
                    if data[symbol].Volume > opening['volume'] * 0.5:
                        self.SetHoldings(symbol, -0.05)
                        self.positions_today.add(symbol)
                        self.Debug(f"Short breakdown: {symbol} at {current_price}")
    
    def CloseAllPositions(self):
        """Close all positions before market close"""
        self.Liquidate()
        self.Debug(f"Closed all positions. Trades today: {len(self.positions_today)}")
