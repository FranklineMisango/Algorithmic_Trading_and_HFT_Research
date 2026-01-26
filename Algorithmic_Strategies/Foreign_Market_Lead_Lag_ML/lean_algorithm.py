"""
LEAN Algorithm for Foreign Market Lead-Lag ML Strategy.
QuantConnect implementation of cross-asset international equity momentum.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from datetime import timedelta


class ForeignMarketLeadLagAlgorithm(QCAlgorithm):
    """
    Foreign Market Lead-Lag ML Strategy for QuantConnect LEAN.
    
    Predicts S&P 500 stock returns using lagged weekly returns from 
    47 foreign equity markets via Lasso regression.
    """
    
    def Initialize(self):
        """Initialize algorithm parameters and data."""
        
        # Set dates and capital
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Strategy parameters
        self.long_percentile = 95  # Top 5%
        self.short_percentile = 5  # Bottom 5%
        self.rebalance_frequency = timedelta(days=1)
        self.last_rebalance = self.Time
        
        # Model parameters
        self.lags = [1, 2, 3, 4]  # weeks
        self.lookback_weeks = 260  # ~5 years for training
        
        # Foreign market ETFs (47 markets)
        self.foreign_markets = [
            "EWJ", "EWG", "EWU", "EWC", "EWA", "EWH", "EWS", "EWW", "EWZ", "EZA",
            "EWI", "EWP", "EWQ", "EWL", "EWN", "EWD", "EWK", "EWO", "EPOL", "EWT",
            "EWY", "EIDO", "EIRL", "EIS", "THD", "ENZL", "ECH", "EPU", "ARGT", "TUR",
            "RSX", "EWM", "EPHE", "EGPT", "QAT", "KSA", "UAE", "GREK", "FXI", "INDA",
            "VNM", "EDEN", "NORW"
        ]
        
        # Add foreign market ETFs
        for ticker in self.foreign_markets:
            self.AddEquity(ticker, Resolution.Daily)
        
        # Add S&P 500 universe
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Storage
        self.foreign_weekly_returns = {}
        self.models = {}
        self.predictions = {}
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )
        
        self.Debug("Foreign Market Lead-Lag ML Algorithm Initialized")
    
    def CoarseSelectionFunction(self, coarse):
        """Select S&P 500 stocks with sufficient liquidity."""
        
        # Filter for liquid stocks
        filtered = [x for x in coarse if x.HasFundamentalData 
                   and x.DollarVolume > 10000000]
        
        # Sort by dollar volume and take top 500
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        
        return [x.Symbol for x in sorted_by_volume[:500]]
    
    def OnData(self, data):
        """Process incoming data."""
        
        # Update foreign market weekly returns
        if self.Time.weekday() == 4:  # Friday
            self.UpdateForeignReturns(data)
    
    def UpdateForeignReturns(self, data):
        """Update weekly returns for foreign markets."""
        
        for ticker in self.foreign_markets:
            symbol = self.Symbol(ticker)
            
            if symbol in data and data[symbol] is not None:
                # Get weekly return
                history = self.History(symbol, 7, Resolution.Daily)
                
                if not history.empty and len(history) >= 2:
                    weekly_return = (history['close'][-1] / history['close'][0]) - 1
                    
                    if ticker not in self.foreign_weekly_returns:
                        self.foreign_weekly_returns[ticker] = []
                    
                    self.foreign_weekly_returns[ticker].append({
                        'date': self.Time,
                        'return': weekly_return
                    })
    
    def CreateLaggedFeatures(self, ticker_returns):
        """Create lagged features from foreign market returns."""
        
        features = []
        
        for ticker, returns_list in ticker_returns.items():
            if len(returns_list) < max(self.lags):
                continue
            
            for lag in self.lags:
                if len(returns_list) >= lag:
                    feature_value = returns_list[-lag]['return']
                    features.append(feature_value)
        
        return np.array(features) if features else None
    
    def TrainModel(self, symbol, features_history, returns_history):
        """Train Lasso model for a single stock."""
        
        if len(features_history) < 100:  # Minimum samples
            return None
        
        X = np.array(features_history)
        y = np.array(returns_history)
        
        # Train Lasso model
        model = Lasso(alpha=0.01, max_iter=10000)
        
        try:
            model.fit(X, y)
            return model
        except:
            return None
    
    def Rebalance(self):
        """Rebalance portfolio based on predictions."""
        
        # Check if enough time has passed
        if self.Time - self.last_rebalance < self.rebalance_frequency:
            return
        
        # Check if we have enough foreign market data
        min_weeks = max(self.lags) + 10
        if not all(len(returns) >= min_weeks 
                  for returns in self.foreign_weekly_returns.values()):
            return
        
        # Create current features
        current_features = self.CreateLaggedFeatures(self.foreign_weekly_returns)
        
        if current_features is None or len(current_features) == 0:
            return
        
        # Generate predictions for all stocks
        predictions = {}
        
        for symbol in self.ActiveSecurities.Keys:
            if symbol.Value in self.foreign_markets:
                continue
            
            # Get historical returns for this stock
            history = self.History(symbol, 252, Resolution.Daily)
            
            if history.empty or len(history) < 100:
                continue
            
            returns = history['close'].pct_change().dropna()
            
            # Train model if not exists or retrain periodically
            if symbol not in self.models or self.Time.day == 1:
                # Create training data
                features_history = []
                returns_history = []
                
                # This is simplified - in production, align features with returns properly
                for i in range(len(returns) - 1):
                    if i < len(current_features):
                        features_history.append(current_features)
                        returns_history.append(returns.iloc[i + 1])
                
                if len(features_history) > 100:
                    model = self.TrainModel(symbol, features_history, returns_history)
                    if model is not None:
                        self.models[symbol] = model
            
            # Make prediction
            if symbol in self.models:
                try:
                    prediction = self.models[symbol].predict([current_features])[0]
                    predictions[symbol] = prediction
                except:
                    continue
        
        if len(predictions) == 0:
            return
        
        # Rank predictions
        predictions_series = pd.Series(predictions)
        ranks = predictions_series.rank(pct=True) * 100
        
        # Select long and short positions
        long_symbols = ranks[ranks >= self.long_percentile].index.tolist()
        short_symbols = ranks[ranks <= self.short_percentile].index.tolist()
        
        # Calculate position sizes
        long_weight = 1.0 / len(long_symbols) if len(long_symbols) > 0 else 0
        short_weight = -1.0 / len(short_symbols) if len(short_symbols) > 0 else 0
        
        # Liquidate positions not in new portfolio
        for symbol in self.Portfolio.Keys:
            if (symbol not in long_symbols and symbol not in short_symbols 
                and self.Portfolio[symbol].Invested):
                self.Liquidate(symbol)
        
        # Enter long positions
        for symbol in long_symbols:
            self.SetHoldings(symbol, long_weight)
        
        # Enter short positions
        for symbol in short_symbols:
            self.SetHoldings(symbol, short_weight)
        
        self.last_rebalance = self.Time
        
        self.Debug(f"Rebalanced: {len(long_symbols)} long, {len(short_symbols)} short")
    
    def OnEndOfAlgorithm(self):
        """Log final statistics."""
        self.Debug(f"Algorithm finished. Final portfolio value: ${self.Portfolio.TotalPortfolioValue}")
