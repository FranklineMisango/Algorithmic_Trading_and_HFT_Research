from AlgorithmImports import *
import numpy as np
from collections import defaultdict

class NewsSentimentMarketNeutral(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2013, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1000000)
        
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        self.sentiment_scores = {}
        self.long_pct = 0.2
        self.short_pct = 0.2
        self.net_exposure = 0.0
        self.rebalance_time = time(15, 45)
        
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(self.rebalance_time),
            self.Rebalance
        )
    
    def CoarseSelectionFunction(self, coarse):
        """Select liquid Russell 3000 stocks"""
        filtered = [x for x in coarse if x.HasFundamentalData and x.DollarVolume > 1e8]
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:3000]]
    
    def OnData(self, data):
        """Process news sentiment data"""
        # In production, integrate with news sentiment API
        # For now, use mock sentiment based on price momentum
        for symbol in self.ActiveSecurities.Keys:
            if symbol in data and data[symbol]:
                history = self.History(symbol, 5, Resolution.Daily)
                if not history.empty:
                    returns = history['close'].pct_change().dropna()
                    sentiment = np.log((1 + returns.mean() + 1e-9) / (1 - returns.mean() + 1e-9))
                    self.sentiment_scores[symbol] = sentiment
    
    def Rebalance(self):
        """Daily portfolio rebalancing"""
        if not self.sentiment_scores:
            return
        
        sorted_by_sentiment = sorted(self.sentiment_scores.items(), key=lambda x: x[1], reverse=True)
        n_stocks = len(sorted_by_sentiment)
        
        n_long = int(n_stocks * self.long_pct)
        n_short = int(n_stocks * self.short_pct)
        
        long_symbols = [x[0] for x in sorted_by_sentiment[:n_long]]
        short_symbols = [x[0] for x in sorted_by_sentiment[-n_short:]]
        
        # Filter shorts to S&P 1500 (simplified: use high market cap)
        short_symbols = [s for s in short_symbols if self.Securities[s].Fundamentals.MarketCap > 5e9]
        
        # Calculate weights
        long_weight = (1.0 + self.net_exposure) / len(long_symbols) if long_symbols else 0
        short_weight = -(1.0 - self.net_exposure) / len(short_symbols) if short_symbols else 0
        
        # Liquidate positions not in new portfolio
        for symbol in self.Portfolio.Keys:
            if symbol not in long_symbols and symbol not in short_symbols:
                self.Liquidate(symbol)
        
        # Enter long positions
        for symbol in long_symbols:
            self.SetHoldings(symbol, long_weight)
        
        # Enter short positions
        for symbol in short_symbols:
            self.SetHoldings(symbol, short_weight)
        
        self.sentiment_scores.clear()
    
    def OnEndOfAlgorithm(self):
        """Calculate final metrics"""
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
