from AlgorithmImports import *
import numpy as np
from collections import defaultdict

class EMDStrategicAllocation(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2003, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1000000)
        
        # Strategy parameters
        self.emd_allocation = 0.10
        self.local_hard_split = [0.6, 0.4]
        self.max_country_weight = 0.05
        self.ppp_threshold = -0.5
        self.rebalance_days = 63  # Quarterly
        
        # EMD ETFs (proxies for local and hard currency)
        self.local_currency_etf = self.AddEquity("EMLC", Resolution.Daily).Symbol  # VanEck EM Local Currency
        self.hard_currency_etf = self.AddEquity("EMB", Resolution.Daily).Symbol    # iShares EM USD Bonds
        
        # Benchmark: 60/40 portfolio
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.agg = self.AddEquity("AGG", Resolution.Daily).Symbol
        
        # Risk indicators
        self.vix = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        
        # Tracking
        self.last_rebalance = self.Time
        self.ppp_history = defaultdict(list)
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.AfterMarketOpen(self.spy, 30),
            self.Rebalance
        )
        
        self.SetWarmUp(252)  # 1 year warmup
    
    def Rebalance(self):
        if self.IsWarmingUp:
            return
        
        days_since_rebalance = (self.Time - self.last_rebalance).days
        if days_since_rebalance < self.rebalance_days:
            return
        
        # Check VIX trigger
        vix_level = self.Securities[self.vix].Price
        emd_allocation = self.emd_allocation
        
        if vix_level > 35:
            emd_allocation *= 0.5  # Reduce allocation by 50%
            self.Debug(f"VIX trigger: {vix_level:.2f}, reducing EMD allocation to {emd_allocation:.2%}")
        
        # Calculate target weights
        local_weight = emd_allocation * self.local_hard_split[0]
        hard_weight = emd_allocation * self.local_hard_split[1]
        
        # Benchmark allocation (remaining capital)
        benchmark_allocation = 1 - emd_allocation
        spy_weight = benchmark_allocation * 0.6
        agg_weight = benchmark_allocation * 0.4
        
        # Set target weights
        targets = {
            self.local_currency_etf: local_weight,
            self.hard_currency_etf: hard_weight,
            self.spy: spy_weight,
            self.agg: agg_weight
        }
        
        # Execute trades with transaction costs
        for symbol, target_weight in targets.items():
            if self.Securities[symbol].Price > 0:
                self.SetHoldings(symbol, target_weight)
        
        self.last_rebalance = self.Time
        
        # Log portfolio state
        self.Debug(f"Rebalanced: EMD={emd_allocation:.2%}, SPY={spy_weight:.2%}, AGG={agg_weight:.2%}")
    
    def OnData(self, data):
        # Monitor correlation (simplified)
        if len(self.ppp_history['emd']) > 90 and len(self.ppp_history['spy']) > 90:
            emd_returns = np.diff(self.ppp_history['emd'][-90:])
            spy_returns = np.diff(self.ppp_history['spy'][-90:])
            
            if len(emd_returns) > 0 and len(spy_returns) > 0:
                correlation = np.corrcoef(emd_returns, spy_returns)[0, 1]
                
                if correlation > 0.6:
                    self.Debug(f"High correlation alert: {correlation:.3f}")
        
        # Track returns for correlation monitoring
        if self.Securities[self.local_currency_etf].Price > 0:
            self.ppp_history['emd'].append(self.Securities[self.local_currency_etf].Price)
        if self.Securities[self.spy].Price > 0:
            self.ppp_history['spy'].append(self.Securities[self.spy].Price)
        
        # Keep only recent history
        if len(self.ppp_history['emd']) > 252:
            self.ppp_history['emd'] = self.ppp_history['emd'][-252:]
        if len(self.ppp_history['spy']) > 252:
            self.ppp_history['spy'] = self.ppp_history['spy'][-252:]
    
    def OnEndOfAlgorithm(self):
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
