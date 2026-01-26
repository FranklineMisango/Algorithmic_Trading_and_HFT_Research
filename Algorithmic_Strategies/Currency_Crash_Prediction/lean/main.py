# region imports
from AlgorithmImports import *
# endregion

class CurrencyCrashPredictionAlgorithm(QCAlgorithm):
    """
    Short-Term Currency Crash Prediction Model
    
    Strategy: Short currencies entering R-Zone (aggressive rate hike + weak currency)
    Expected crash probability: ~43% vs 7.8% baseline within 6 months
    """
    
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1000000)
        
        # Currency pairs (vs USD)
        self.currencies = {
            'EURUSD': {'type': 'advanced', 'symbol': None},
            'GBPUSD': {'type': 'advanced', 'symbol': None},
            'AUDUSD': {'type': 'advanced', 'symbol': None},
            'NZDUSD': {'type': 'advanced', 'symbol': None},
            'USDJPY': {'type': 'advanced', 'symbol': None, 'inverse': True},
            'USDCAD': {'type': 'advanced', 'symbol': None, 'inverse': True},
            'USDCHF': {'type': 'advanced', 'symbol': None, 'inverse': True},
        }
        
        # Add Forex symbols
        for pair, info in self.currencies.items():
            symbol = self.AddForex(pair, Resolution.Daily, Market.Oanda).Symbol
            info['symbol'] = symbol
        
        # Parameters
        self.lookback_months = 6
        self.rate_threshold_pct = 80  # Top 20%
        self.fx_threshold_pct = 33    # Bottom 33%
        self.crash_threshold_pct = 4  # Bottom 4%
        self.position_duration_months = 6
        self.max_exposure = 0.10
        
        # Data storage
        self.fx_history = {}
        self.rate_history = {}  # Simulated - LEAN doesn't have direct rate data
        self.thresholds = {}
        self.positions = {}
        self.r_zone_active = {}
        
        # Monthly rebalance
        self.Schedule.On(
            self.DateRules.MonthEnd(),
            self.TimeRules.At(16, 0),
            self.Rebalance
        )
        
        # Warm up period
        self.SetWarmUp(timedelta(days=365))
        
        self.Debug("Currency Crash Prediction Algorithm Initialized")
    
    def OnData(self, data):
        """Store daily FX data"""
        if self.IsWarmingUp:
            return
        
        for pair, info in self.currencies.items():
            symbol = info['symbol']
            if data.ContainsKey(symbol):
                if pair not in self.fx_history:
                    self.fx_history[pair] = []
                
                price = data[symbol].Close
                self.fx_history[pair].append({
                    'date': self.Time,
                    'price': price
                })
    
    def Rebalance(self):
        """Monthly rebalancing - generate signals and manage positions"""
        if self.IsWarmingUp:
            return
        
        self.Debug(f"Rebalancing on {self.Time}")
        
        # Calculate features and signals
        signals = self.GenerateSignals()
        
        # Close expired positions
        self.CloseExpiredPositions()
        
        # Open new positions for R-Zone entries
        self.OpenNewPositions(signals)
    
    def GenerateSignals(self):
        """Generate R-Zone signals for all currencies"""
        signals = {}
        
        for pair, info in self.currencies.items():
            # Get monthly prices (last 6 months)
            if pair not in self.fx_history or len(self.fx_history[pair]) < 180:
                continue
            
            prices = pd.DataFrame(self.fx_history[pair][-180:])
            prices['date'] = pd.to_datetime(prices['date'])
            prices.set_index('date', inplace=True)
            monthly_prices = prices.resample('M').last()
            
            if len(monthly_prices) < self.lookback_months + 1:
                continue
            
            # Calculate ΔFX (log change over 6 months)
            log_prices = np.log(monthly_prices['price'])
            delta_fx = log_prices.iloc[-1] - log_prices.iloc[-self.lookback_months-1]
            
            # Simulate Δi (interest rate change)
            # In production, fetch from external API or custom data
            delta_i = self.SimulateRateChange(pair)
            
            # Calculate thresholds from historical data
            if pair not in self.thresholds:
                self.thresholds[pair] = self.CalculateThresholds(pair, monthly_prices)
            
            # Check R-Zone conditions
            condition_a = delta_i >= self.thresholds[pair]['rate']
            condition_b = delta_fx <= self.thresholds[pair]['fx']
            
            r_zone = condition_a and condition_b
            
            signals[pair] = {
                'r_zone': r_zone,
                'delta_i': delta_i,
                'delta_fx': delta_fx,
                'current_price': monthly_prices['price'].iloc[-1]
            }
            
            if r_zone:
                self.Debug(f"{pair} entered R-Zone: Δi={delta_i:.4f}, ΔFX={delta_fx:.4f}")
        
        return signals
    
    def SimulateRateChange(self, pair):
        """Simulate interest rate change (placeholder)"""
        # In production: fetch from FRED, Bloomberg, or custom data feed
        # For demo: use random walk with mean reversion
        if pair not in self.rate_history:
            self.rate_history[pair] = 2.0
        
        shock = np.random.normal(0, 0.5)
        self.rate_history[pair] += shock
        
        return self.rate_history[pair]
    
    def CalculateThresholds(self, pair, monthly_prices):
        """Calculate historical thresholds for R-Zone"""
        # Use all available history
        log_prices = np.log(monthly_prices['price'])
        delta_fx_series = log_prices.diff(self.lookback_months).dropna()
        
        # Simulate rate changes for threshold calculation
        delta_i_series = pd.Series([
            np.random.normal(0.5, 1.0) for _ in range(len(delta_fx_series))
        ])
        
        # Monthly returns for crash threshold
        monthly_returns = monthly_prices['price'].pct_change().dropna()
        
        thresholds = {
            'rate': delta_i_series.quantile(self.rate_threshold_pct / 100),
            'fx': delta_fx_series.quantile(self.fx_threshold_pct / 100),
            'crash': monthly_returns.quantile(self.crash_threshold_pct / 100)
        }
        
        return thresholds
    
    def OpenNewPositions(self, signals):
        """Open short positions for new R-Zone entries"""
        for pair, signal in signals.items():
            if not signal['r_zone']:
                continue
            
            # Skip if already have position
            if pair in self.positions and self.positions[pair]['active']:
                continue
            
            symbol = self.currencies[pair]['symbol']
            
            # Position sizing: equal risk allocation
            position_size = self.Portfolio.TotalPortfolioValue * self.max_exposure
            quantity = position_size / signal['current_price']
            
            # Short the currency (expect depreciation/crash)
            is_inverse = self.currencies[pair].get('inverse', False)
            
            if is_inverse:
                # For pairs like USDJPY, go long to short foreign currency
                self.SetHoldings(symbol, self.max_exposure)
            else:
                # For pairs like EURUSD, go short
                self.SetHoldings(symbol, -self.max_exposure)
            
            # Track position
            self.positions[pair] = {
                'active': True,
                'entry_date': self.Time,
                'entry_price': signal['current_price'],
                'expiry_date': self.Time + timedelta(days=30 * self.position_duration_months)
            }
            
            self.Debug(f"Opened short position in {pair} at {signal['current_price']}")
    
    def CloseExpiredPositions(self):
        """Close positions after 6 months or if crash occurred"""
        for pair, position in list(self.positions.items()):
            if not position['active']:
                continue
            
            # Check expiry
            if self.Time >= position['expiry_date']:
                symbol = self.currencies[pair]['symbol']
                self.Liquidate(symbol)
                position['active'] = False
                self.Debug(f"Closed expired position in {pair}")
    
    def OnEndOfAlgorithm(self):
        """Log final statistics"""
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Debug(f"Total Return: {(self.Portfolio.TotalPortfolioValue / 1000000 - 1) * 100:.2f}%")
