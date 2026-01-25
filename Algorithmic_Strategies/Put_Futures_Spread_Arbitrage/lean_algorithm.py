# QuantConnect Lean - Put Futures Spread Arbitrage

from AlgorithmImports import *
import numpy as np


class PutFuturesSpreadAlgorithm(QCAlgorithm):
    """
    Put-Call Parity Arbitrage with Futures
    
    Exploits violations of put-call parity in options
    """
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Underlying
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Options
        option = self.AddOption("SPY", Resolution.Daily)
        option.SetFilter(-3, 3, 30, 60)  # ATM, 1-2 months
        
        # Futures
        self.es = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Daily)
        self.es.SetFilter(0, 90)
        
        self.arbitrage_threshold = 0.02  # 2% mispricing threshold
        
        self.Debug("Put-Futures Spread Algorithm Initialized")
    
    def OnData(self, data):
        """Detect put-call parity violations"""
        # Get option chains
        for chain in data.OptionChains:
            # Find ATM options
            underlying_price = self.Securities[self.spy].Price
            
            calls = [x for x in chain.Value if x.Right == OptionRight.Call]
            puts = [x for x in chain.Value if x.Right == OptionRight.Put]
            
            if not calls or not puts:
                continue
            
            # Find matched pairs (same strike and expiry)
            for call in calls:
                matching_puts = [p for p in puts 
                               if p.Strike == call.Strike 
                               and p.Expiry == call.Expiry]
                
                if not matching_puts:
                    continue
                
                put = matching_puts[0]
                
                # Put-call parity: C - P = S - K*e^(-rT)
                strike = call.Strike
                time_to_expiry = (call.Expiry - self.Time).days / 365.0
                
                # Simplified (assume r=0 for demo)
                parity_lhs = call.BidPrice - put.AskPrice
                parity_rhs = underlying_price - strike
                
                mispricing = abs(parity_lhs - parity_rhs)
                
                # Trade if mispricing exceeds threshold
                if mispricing > self.arbitrage_threshold * underlying_price:
                    if parity_lhs > parity_rhs:
                        # Sell call, buy put, buy stock
                        self.Sell(call.Symbol, 1)
                        self.Buy(put.Symbol, 1)
                        self.SetHoldings(self.spy, 0.05)
                        self.Debug(f"Arbitrage: Sell {call.Symbol}")
                    else:
                        # Buy call, sell put, short stock
                        self.Buy(call.Symbol, 1)
                        self.Sell(put.Symbol, 1)
                        self.SetHoldings(self.spy, -0.05)
                        self.Debug(f"Arbitrage: Buy {call.Symbol}")
