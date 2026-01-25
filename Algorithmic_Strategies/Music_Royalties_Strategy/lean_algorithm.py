# QuantConnect Lean - Music Royalties Strategy

from AlgorithmImports import *
import numpy as np


class MusicRoyaltiesAlgorithm(QCAlgorithm):
    """
    Music Royalty Streaming Strategy
    
    Invests in music royalty companies and streaming platforms
    """
    
    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Music royalty companies
        self.symbols = []
        
        # Round Hill Music Royalty Fund
        try:
            self.rhmrf = self.AddEquity("RHM", Resolution.Daily).Symbol
            self.symbols.append(self.rhmrf)
        except:
            pass
        
        # Streaming platforms
        self.spot = self.AddEquity("SPOT", Resolution.Daily).Symbol  # Spotify
        self.symbols.append(self.spot)
        
        # Warner Music Group
        self.wmg = self.AddEquity("WMG", Resolution.Daily).Symbol
        self.symbols.append(self.wmg)
        
        # Equal weight rebalancing
        self.Schedule.On(
            self.DateRules.MonthStart(self.spot),
            self.TimeRules.At(10, 0),
            self.Rebalance
        )
        
        self.Debug("Music Royalties Algorithm Initialized")
    
    def Rebalance(self):
        """Equal weight portfolio"""
        if not self.symbols:
            return
        
        weight = 1.0 / len(self.symbols)
        
        for symbol in self.symbols:
            self.SetHoldings(symbol, weight)
