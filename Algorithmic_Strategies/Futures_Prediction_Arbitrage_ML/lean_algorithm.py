# QuantConnect Lean - Futures Prediction Arbitrage ML

from AlgorithmImports import *
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from collections import deque, defaultdict


class FuturesPredictionAlgorithm(QCAlgorithm):
    """
    Futures Prediction with ML Arbitrage
    
    Predicts futures mispricing using ML and trades basis
    """
    
    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Futures contracts
        self.es = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Daily)
        self.es.SetFilter(0, 90)  # Front month
        
        self.nq = self.AddFuture(Futures.Indices.NASDAQ100EMini, Resolution.Daily)
        self.nq.SetFilter(0, 90)
        
        # Spot proxies
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.qqq = self.AddEquity("QQQ", Resolution.Daily).Symbol
        
        # ML model for basis prediction
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        self.trained = False
        
        # Feature tracking
        self.basis_history = deque(maxlen=252)
        self.volume_history = deque(maxlen=60)
        
        self.Debug("Futures Prediction Algorithm Initialized")
    
    def OnData(self, data):
        """Track futures basis"""
        # Get front month futures
        for chain in data.FutureChains:
            contracts = sorted(chain.Value, key=lambda x: x.Expiry)
            
            if len(contracts) > 0:
                front = contracts[0]
                
                # Calculate basis vs spot
                if chain.Key.Canonical.Value == "ES":
                    if self.spy in data and data[self.spy]:
                        spot = data[self.spy].Close
                        basis = (front.LastPrice - spot) / spot
                        self.basis_history.append(basis)
                
                self.volume_history.append(front.Volume)
    
    def OnSecuritiesChanged(self, changes):
        """Handle futures contract rollovers"""
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
