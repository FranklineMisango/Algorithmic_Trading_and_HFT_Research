# QuantConnect Lean - Statistical Arbitrage ML

from AlgorithmImports import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import deque, defaultdict


class StatArbMLAlgorithm(QCAlgorithm):
    """
    Statistical Arbitrage with Machine Learning
    
    Uses ML to predict short-term mean reversion opportunities
    """
    
    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Universe
        self.universe_settings.resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # ML model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.trained = False
        
        # Feature tracking
        self.feature_lookback = 20
        self.price_history = defaultdict(lambda: deque(maxlen=self.feature_lookback))
        self.volume_history = defaultdict(lambda: deque(maxlen=self.feature_lookback))
        
        # Rebalance schedule
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(10, 0),
            self.Rebalance
        )
        
        self.Debug("Statistical Arbitrage ML Algorithm Initialized")
    
    def CoarseSelectionFunction(self, coarse):
        """Select universe"""
        filtered = [x for x in coarse if x.Price > 5 and x.DollarVolume > 10000000]
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:200]]
    
    def OnData(self, data):
        """Collect features"""
        for symbol in data.Keys:
            if data[symbol]:
                self.price_history[symbol].append(data[symbol].Close)
                self.volume_history[symbol].append(data[symbol].Volume)
    
    def CalculateFeatures(self, symbol):
        """Calculate ML features for a symbol"""
        if len(self.price_history[symbol]) < 10:
            return None
        
        prices = np.array(self.price_history[symbol])
        volumes = np.array(self.volume_history[symbol])
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        
        # Features
        features = [
            returns[-1] if len(returns) > 0 else 0,  # Last return
            returns[-5:].mean() if len(returns) >= 5 else 0,  # 5-day avg return
            np.std(returns[-10:]) if len(returns) >= 10 else 0,  # 10-day vol
            (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0,  # 20-day momentum
            volumes[-1] / volumes.mean() if volumes.mean() > 0 else 1,  # Volume ratio
        ]
        
        return features
    
    def TrainModel(self):
        """Train ML model on historical data"""
        X = []
        y = []
        
        for symbol in self.price_history.keys():
            if len(self.price_history[symbol]) < self.feature_lookback:
                continue
            
            prices = np.array(self.price_history[symbol])
            
            for i in range(10, len(prices) - 1):
                # Features at time t
                window_prices = prices[:i+1]
                returns = np.diff(window_prices) / window_prices[:-1]
                
                feat = [
                    returns[-1],
                    returns[-5:].mean(),
                    np.std(returns[-10:]),
                    (window_prices[-1] / window_prices[-20] - 1) if len(window_prices) >= 20 else 0,
                    1.0,
                ]
                
                # Target: next day return direction
                next_return = (prices[i+1] / prices[i]) - 1
                label = 1 if next_return > 0 else 0
                
                X.append(feat)
                y.append(label)
        
        if len(X) > 100:
            self.model.fit(np.array(X), np.array(y))
            self.trained = True
            self.Debug(f"Model trained on {len(X)} samples")
    
    def Rebalance(self):
        """Rebalance based on ML predictions"""
        if not self.trained:
            self.TrainModel()
            if not self.trained:
                return
        
        predictions = {}
        
        for symbol in self.ActiveSecurities.Keys:
            features = self.CalculateFeatures(symbol)
            if features is None:
                continue
            
            # Predict
            prob = self.model.predict_proba([features])[0]
            predictions[symbol] = prob[1] - 0.5  # Score: -0.5 to +0.5
        
        # Select top/bottom predictions
        sorted_symbols = sorted(predictions.items(), key=lambda x: x[1])
        
        n_positions = 20
        shorts = sorted_symbols[:n_positions//2]
        longs = sorted_symbols[-n_positions//2:]
        
        # Execute
        for symbol, score in longs:
            self.SetHoldings(symbol, 1.0 / n_positions)
        
        for symbol, score in shorts:
            self.SetHoldings(symbol, -1.0 / n_positions)
        
        # Liquidate others
        for symbol in self.Portfolio.Keys:
            if symbol not in [s for s, _ in longs + shorts]:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
