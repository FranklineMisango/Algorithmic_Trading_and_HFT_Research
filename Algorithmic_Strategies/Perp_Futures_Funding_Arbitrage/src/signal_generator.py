import numpy as np
import pandas as pd
from loguru import logger

class SignalGenerator:
    def __init__(self, config):
        self.delta = config['parameters']['delta']
        self.spot_fee = config['parameters']['spot_fee']
        self.perp_fee = config['parameters']['perp_fee']
        self.funding_interval = config['parameters']['funding_interval']
        
    def calculate_premium_index(self, perp_price, spot_price):
        """Calculate premium/discount index: I = (PerpPrice / SpotPrice) - 1"""
        return (perp_price / spot_price) - 1
    
    def calculate_bounds(self, risk_free_rate, borrow_rate):
        """Calculate dynamic no-arbitrage bounds"""
        # Time fraction for funding interval (8 hours = 1/1095 years)
        dt = self.funding_interval / (365 * 24)
        
        # Transaction costs
        tc = self.spot_fee + self.perp_fee
        
        # Upper bound: Short perp / Long spot
        # I > δ + (cs + cp + (rc - rf)Δt)
        upper_bound = self.delta + tc + (borrow_rate - risk_free_rate) * dt
        
        # Lower bound: Long perp / Short spot
        # I < -δ + (cs + cp + (rf - rc)Δt)
        lower_bound = -self.delta + tc + (risk_free_rate - borrow_rate) * dt
        
        return upper_bound, lower_bound
    
    def generate_signals(self, df):
        """Generate trading signals based on bound breaches"""
        logger.info("Generating signals")
        
        signals = pd.DataFrame(index=df.index)
        
        # Calculate premium index
        signals['premium_index'] = self.calculate_premium_index(
            df['perp_price'], 
            df['spot_price']
        )
        
        # Calculate bounds for each timestamp
        bounds = df.apply(
            lambda row: self.calculate_bounds(row['risk_free_rate'], row['borrow_rate']),
            axis=1
        )
        signals['upper_bound'] = [b[0] for b in bounds]
        signals['lower_bound'] = [b[1] for b in bounds]
        
        # Generate signals
        # 1 = Short perp / Long spot (perp overpriced)
        # -1 = Long perp / Short spot (perp underpriced)
        # 0 = No signal
        signals['signal'] = 0
        signals.loc[signals['premium_index'] > signals['upper_bound'], 'signal'] = 1
        signals.loc[signals['premium_index'] < signals['lower_bound'], 'signal'] = -1
        
        # Signal active flag
        signals['signal_active'] = (signals['signal'] != 0).astype(int)
        
        # Calculate distance from bounds (for risk management)
        signals['distance_from_bound'] = np.where(
            signals['signal'] == 1,
            signals['premium_index'] - signals['upper_bound'],
            np.where(
                signals['signal'] == -1,
                signals['lower_bound'] - signals['premium_index'],
                0
            )
        )
        
        logger.info(f"Signals generated: {signals['signal_active'].sum()} active periods")
        return signals
    
    def validate_bounds(self, df, signals):
        """Validate that price ratio stays within bounds (>95% of time)"""
        within_bounds = (
            (signals['premium_index'] >= signals['lower_bound']) & 
            (signals['premium_index'] <= signals['upper_bound'])
        )
        
        pct_within = within_bounds.sum() / len(signals) * 100
        logger.info(f"Price ratio within bounds: {pct_within:.2f}% of time")
        
        if pct_within < 95:
            logger.warning("Model validation failed: <95% within bounds")
        
        return pct_within
