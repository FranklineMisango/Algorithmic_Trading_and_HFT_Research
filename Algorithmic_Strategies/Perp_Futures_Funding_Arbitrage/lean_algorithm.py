from AlgorithmImports import *
import numpy as np

class PerpetualFuturesFundingArbitrage(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        # Strategy parameters
        self.delta = 0.0005  # 5 bps clamp
        self.spot_fee = 0.00009
        self.perp_fee = 0.0
        self.funding_interval_hours = 8
        self.stop_loss_bps = 0.001
        self.max_vol_pct = 0.02
        
        # Add crypto assets
        self.btc = self.AddCrypto("BTCUSD", Resolution.Hour)
        self.eth = self.AddCrypto("ETHUSD", Resolution.Hour)
        
        # Position tracking
        self.positions = {}
        
        # Risk-free and borrow rates (simplified)
        self.risk_free_rate = 0.05
        self.borrow_rate = 0.08
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.EveryDay(self.btc.Symbol),
            self.TimeRules.Every(TimeSpan.FromHours(1)),
            self.Rebalance
        )
    
    def CalculatePremiumIndex(self, perp_price, spot_price):
        """Calculate I = (PerpPrice / SpotPrice) - 1"""
        if spot_price == 0:
            return 0
        return (perp_price / spot_price) - 1
    
    def CalculateBounds(self):
        """Calculate dynamic no-arbitrage bounds"""
        dt = self.funding_interval_hours / (365 * 24)
        tc = self.spot_fee + self.perp_fee
        
        upper_bound = self.delta + tc + (self.borrow_rate - self.risk_free_rate) * dt
        lower_bound = -self.delta + tc + (self.risk_free_rate - self.borrow_rate) * dt
        
        return upper_bound, lower_bound
    
    def Rebalance(self):
        """Main trading logic"""
        for symbol in [self.btc.Symbol, self.eth.Symbol]:
            # Get current prices
            if not self.Securities[symbol].HasData:
                continue
            
            spot_price = self.Securities[symbol].Price
            
            # In real implementation, fetch perpetual price from futures
            # For demo, simulate with slight premium
            perp_price = spot_price * 1.0002
            
            # Calculate premium index
            premium_idx = self.CalculatePremiumIndex(perp_price, spot_price)
            
            # Calculate bounds
            upper_bound, lower_bound = self.CalculateBounds()
            
            # Check if position exists
            position_key = str(symbol)
            has_position = position_key in self.positions
            
            if not has_position:
                # Entry logic
                if premium_idx > upper_bound:
                    # Short perp / Long spot
                    self.SetHoldings(symbol, 0.25)  # Long spot
                    self.positions[position_key] = {
                        'type': 'short_perp_long_spot',
                        'entry_index': premium_idx,
                        'entry_bound': upper_bound
                    }
                    self.Debug(f"Enter SHORT PERP/LONG SPOT {symbol} at premium {premium_idx:.4f}")
                
                elif premium_idx < lower_bound:
                    # Long perp / Short spot
                    self.SetHoldings(symbol, -0.25)  # Short spot
                    self.positions[position_key] = {
                        'type': 'long_perp_short_spot',
                        'entry_index': premium_idx,
                        'entry_bound': lower_bound
                    }
                    self.Debug(f"Enter LONG PERP/SHORT SPOT {symbol} at premium {premium_idx:.4f}")
            
            else:
                # Exit logic
                position = self.positions[position_key]
                exit_signal = False
                
                # Exit if premium returns within bounds
                if position['type'] == 'short_perp_long_spot' and premium_idx <= upper_bound:
                    exit_signal = True
                elif position['type'] == 'long_perp_short_spot' and premium_idx >= lower_bound:
                    exit_signal = True
                
                # Stop loss
                if position['type'] == 'short_perp_long_spot':
                    if premium_idx > position['entry_bound'] + self.stop_loss_bps:
                        exit_signal = True
                        self.Debug(f"Stop loss triggered for {symbol}")
                elif position['type'] == 'long_perp_short_spot':
                    if premium_idx < position['entry_bound'] - self.stop_loss_bps:
                        exit_signal = True
                        self.Debug(f"Stop loss triggered for {symbol}")
                
                if exit_signal:
                    self.Liquidate(symbol)
                    del self.positions[position_key]
                    self.Debug(f"Exit {symbol} at premium {premium_idx:.4f}")
    
    def OnEndOfAlgorithm(self):
        """Log final statistics"""
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue}")
