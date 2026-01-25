# QuantConnect Lean - Copy Congress Strategy

from AlgorithmImports import *
from collections import defaultdict, deque
import numpy as np


class CopyCongressAlgorithm(QCAlgorithm):
    """
    Copy Congress Trading Strategy using Lean
    
    Replicates Congressional stock trades with 45-day delay
    Uses inverse volatility weighting and weekly rebalancing
    """
    
    def Initialize(self):
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Strategy parameters
        self.lookback_days = 45  # Congressional disclosure delay
        self.rebalance_frequency = 7  # Weekly
        self.max_position_size = 0.10  # 10% max per stock
        self.vol_lookback = 63  # 3 months for volatility
        
        # Track congressional trades (in practice, would use custom data)
        # For demo, we'll simulate by tracking top holdings
        self.congress_signals = defaultdict(lambda: {'buy': 0, 'sell': 0, 'net': 0})
        self.active_stocks = set()
        
        # Add universe of liquid stocks
        self.universe_settings.resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Volatility tracking
        self.volatilities = {}
        self.price_history = defaultdict(lambda: deque(maxlen=self.vol_lookback))
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Friday),
            self.TimeRules.At(15, 30),
            self.Rebalance
        )
        
        self.last_rebalance = self.Time
        
        self.Debug("Copy Congress Algorithm Initialized")
    
    def CoarseSelectionFunction(self, coarse):
        """Select liquid stocks for universe"""
        # Filter for liquid stocks with sufficient price and volume
        filtered = [x for x in coarse if x.Price > 5 and x.DollarVolume > 10000000]
        
        # Sort by dollar volume and take top 500
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:500]]
    
    def OnData(self, data):
        """Track price history for volatility calculations"""
        for symbol in data.Keys:
            if data[symbol] is not None and data[symbol].Close > 0:
                self.price_history[symbol].append(data[symbol].Close)
    
    def OnSecuritiesChanged(self, changes):
        """Handle universe changes"""
        for security in changes.AddedSecurities:
            # Initialize tracking for new securities
            self.volatilities[security.Symbol] = 0.20  # Default volatility
        
        for security in changes.RemovedSecurities:
            # Clean up removed securities
            if security.Symbol in self.volatilities:
                del self.volatilities[security.Symbol]
            if security.Symbol in self.price_history:
                del self.price_history[security.Symbol]
    
    def SimulateCongressionalSignals(self):
        """
        Simulate Congressional trading signals
        
        In production, this would:
        1. Fetch Congressional trade disclosures via API
        2. Parse transactions with 45-day delay
        3. Aggregate buy/sell flows
        4. Apply committee weighting
        """
        # For demo: use momentum as proxy for "congressional interest"
        signals = {}
        
        for symbol in self.ActiveSecurities.Keys:
            if len(self.price_history[symbol]) < 20:
                continue
            
            prices = np.array(self.price_history[symbol])
            
            # Momentum signal (proxy for congressional buying)
            if len(prices) >= self.lookback_days:
                momentum = (prices[-1] / prices[-self.lookback_days]) - 1
                
                # Positive momentum = "buy" signal
                # In real implementation, this comes from actual congressional filings
                if momentum > 0.05:  # 5% threshold
                    signals[symbol] = 1
                elif momentum < -0.05:
                    signals[symbol] = -1
        
        return signals
    
    def CalculateVolatilities(self):
        """Calculate realized volatility for each stock"""
        for symbol, prices in self.price_history.items():
            if len(prices) >= 20:
                prices_array = np.array(prices)
                returns = np.diff(prices_array) / prices_array[:-1]
                vol = np.std(returns) * np.sqrt(252)
                self.volatilities[symbol] = vol if vol > 0 else 0.20
    
    def Rebalance(self):
        """Weekly rebalancing based on congressional signals"""
        if (self.Time - self.last_rebalance).days < self.rebalance_frequency:
            return
        
        self.Debug(f"Rebalancing on {self.Time}")
        
        # Update volatilities
        self.CalculateVolatilities()
        
        # Get congressional signals
        signals = self.SimulateCongressionalSignals()
        
        if not signals:
            return
        
        # Calculate inverse volatility weights
        weights = {}
        
        long_stocks = {s: sig for s, sig in signals.items() if sig > 0}
        short_stocks = {s: sig for s, sig in signals.items() if sig < 0}
        
        # Long positions
        if long_stocks:
            inv_vol_long = {s: 1/self.volatilities.get(s, 0.20) for s in long_stocks.keys()}
            total_inv_vol_long = sum(inv_vol_long.values())
            
            for symbol in long_stocks.keys():
                weight = (inv_vol_long[symbol] / total_inv_vol_long)
                weight = min(weight, self.max_position_size)  # Position limit
                weights[symbol] = weight
        
        # Short positions (if enabled)
        if short_stocks:
            inv_vol_short = {s: 1/self.volatilities.get(s, 0.20) for s in short_stocks.keys()}
            total_inv_vol_short = sum(inv_vol_short.values())
            
            for symbol in short_stocks.keys():
                weight = -(inv_vol_short[symbol] / total_inv_vol_short)
                weight = max(weight, -self.max_position_size)
                weights[symbol] = weight
        
        # Renormalize to ensure total doesn't exceed 100%
        total_abs_weight = sum(abs(w) for w in weights.values())
        if total_abs_weight > 1.0:
            weights = {s: w/total_abs_weight for s, w in weights.items()}
        
        # Execute trades
        current_symbols = set(self.Portfolio.Keys)
        target_symbols = set(weights.keys())
        
        # Liquidate positions not in target
        for symbol in current_symbols - target_symbols:
            if self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)
        
        # Set target positions
        for symbol, weight in weights.items():
            if weight != 0:
                self.SetHoldings(symbol, weight)
        
        self.last_rebalance = self.Time
    
    def OnEndOfAlgorithm(self):
        """Summary statistics"""
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Debug(f"Total Return: {(self.Portfolio.TotalPortfolioValue/1000000 - 1)*100:.2f}%")
