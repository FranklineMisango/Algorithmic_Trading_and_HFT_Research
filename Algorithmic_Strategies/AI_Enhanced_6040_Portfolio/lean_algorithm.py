# QuantConnect Lean - AI-Enhanced 60/40 Portfolio

from AlgorithmImports import *
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from collections import deque


class AIEnhanced6040Algorithm(QCAlgorithm):
    """
    AI-Enhanced 60/40 Portfolio using Lean
    
    Uses decision tree ML to dynamically allocate across:
    - Stocks (SPY)
    - Bonds (TLT)
    - Gold (GLD)
    - Bitcoin (BTC)
    
    Features: VIX, Yield Spread, Interest Rates
    """
    
    def Initialize(self):
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Assets
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol
        
        # Crypto (if available)
        try:
            self.btc = self.AddCrypto("BTCUSD", Resolution.Daily).Symbol
        except:
            self.btc = None
        
        # Indicators
        self.vix = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        
        # Feature history
        self.lookback = 252  # 1 year for training
        self.vix_history = deque(maxlen=self.lookback)
        self.yield_spread_history = deque(maxlen=self.lookback)
        self.rate_history = deque(maxlen=self.lookback)
        
        # ML model
        self.models = {
            'SPY': DecisionTreeRegressor(max_depth=5, random_state=42),
            'TLT': DecisionTreeRegressor(max_depth=5, random_state=42),
            'GLD': DecisionTreeRegressor(max_depth=5, random_state=42),
        }
        if self.btc:
            self.models['BTC'] = DecisionTreeRegressor(max_depth=5, random_state=42)
        
        self.trained = False
        self.retrain_frequency = 30  # Retrain monthly
        self.last_retrain = self.Time
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.MonthStart(self.spy),
            self.TimeRules.At(10, 0),
            self.Rebalance
        )
        
        self.Debug("AI-Enhanced 60/40 Algorithm Initialized")
    
    def OnData(self, data):
        """Collect features for ML"""
        # VIX
        if self.vix in data and data[self.vix]:
            self.vix_history.append(data[self.vix].Value)
        
        # Yield spread proxy: TLT vs SPY yield
        if self.tlt in data and self.spy in data:
            if data[self.tlt] and data[self.spy]:
                # Simplified: use price ratio as proxy
                spread = data[self.tlt].Close / data[self.spy].Close
                self.yield_spread_history.append(spread)
        
        # Interest rate proxy: inverse of bond price
        if self.tlt in data and data[self.tlt]:
            rate_proxy = 1 / data[self.tlt].Close * 100
            self.rate_history.append(rate_proxy)
    
    def GetCurrentFeatures(self):
        """Get current feature values"""
        if len(self.vix_history) < 10:
            return None
        
        features = [
            self.vix_history[-1] if self.vix_history else 20.0,
            self.yield_spread_history[-1] if self.yield_spread_history else 0.5,
            self.rate_history[-1] if self.rate_history else 2.0,
        ]
        
        return np.array(features).reshape(1, -1)
    
    def TrainModels(self):
        """Train/retrain ML models"""
        if len(self.vix_history) < 100:
            return False
        
        self.Debug(f"Training models on {self.Time}")
        
        # Prepare training data
        n_samples = min(len(self.vix_history), 252)
        
        X = []
        for i in range(n_samples):
            if (i < len(self.vix_history) and 
                i < len(self.yield_spread_history) and 
                i < len(self.rate_history)):
                X.append([
                    self.vix_history[i],
                    self.yield_spread_history[i],
                    self.rate_history[i]
                ])
        
        X = np.array(X)
        
        # Get historical returns for each asset
        for asset_name, symbol in [('SPY', self.spy), ('TLT', self.tlt), ('GLD', self.gld)]:
            history = self.History(symbol, n_samples + 20, Resolution.Daily)
            
            if len(history) > 20:
                returns = history['close'].pct_change().dropna()
                
                # Align with features
                if len(returns) >= len(X):
                    y = returns.values[-len(X):]
                    
                    # Train model
                    self.models[asset_name].fit(X, y)
        
        if self.btc:
            history = self.History(self.btc, n_samples + 20, Resolution.Daily)
            if len(history) > 20:
                returns = history['close'].pct_change().dropna()
                if len(returns) >= len(X):
                    y = returns.values[-len(X):]
                    self.models['BTC'].fit(X, y)
        
        self.trained = True
        self.last_retrain = self.Time
        return True
    
    def Rebalance(self):
        """Monthly rebalancing with ML predictions"""
        # Retrain if needed
        if not self.trained or (self.Time - self.last_retrain).days >= self.retrain_frequency:
            if not self.TrainModels():
                return
        
        # Get current features
        features = self.GetCurrentFeatures()
        if features is None:
            return
        
        # Predict returns for each asset
        predictions = {}
        for asset_name, model in self.models.items():
            pred = model.predict(features)[0]
            predictions[asset_name] = pred
        
        self.Debug(f"Predictions: {predictions}")
        
        # Convert to allocations (softmax-like)
        # Positive predictions only
        positive_preds = {k: max(v, 0) for k, v in predictions.items()}
        total = sum(positive_preds.values())
        
        if total == 0:
            # Default 60/40
            allocations = {'SPY': 0.60, 'TLT': 0.40, 'GLD': 0.0}
        else:
            allocations = {k: v/total for k, v in positive_preds.items()}
        
        # Set holdings
        self.SetHoldings(self.spy, allocations.get('SPY', 0))
        self.SetHoldings(self.tlt, allocations.get('TLT', 0))
        self.SetHoldings(self.gld, allocations.get('GLD', 0))
        if self.btc and 'BTC' in allocations:
            self.SetHoldings(self.btc, allocations.get('BTC', 0))
