"""
Options Strategy Module for Holiday Effect

Implements put-selling overlay strategy around holiday events.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yaml


class OptionsStrategy:
    """
    Sell OTM puts on AMZN before holiday events.
    
    Strategy: Sell puts expiring shortly after event, capturing premium
    from anticipated positive price movement.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.options_config = self.config['options_strategy']
        self.portfolio_config = self.config['portfolio']
        
    def calculate_strike_price(self, 
                                 current_price: float,
                                 offset_pct: float = None) -> float:
        """
        Calculate OTM put strike price.
        
        Args:
            current_price: Current stock price
            offset_pct: Percentage below current (default from config)
            
        Returns:
            Strike price
        """
        if offset_pct is None:
            offset_pct = self.options_config['strike_offset_pct']
        
        strike = current_price * (1 - offset_pct)
        
        # Round to nearest $5 (standard options strikes)
        strike = round(strike / 5) * 5
        
        return strike
    
    def estimate_option_premium(self,
                                strike: float,
                                spot: float,
                                volatility: float = 0.30,
                                dte: int = 10,
                                risk_free_rate: float = 0.03) -> float:
        """
        Estimate put option premium using simplified Black-Scholes.
        
        NOTE: In production, use real options data from broker.
        
        Args:
            strike: Strike price
            spot: Current stock price
            volatility: Implied volatility (annualized)
            dte: Days to expiration
            risk_free_rate: Risk-free rate
            
        Returns:
            Estimated premium per share
        """
        from scipy.stats import norm
        
        # Time to expiration in years
        T = dte / 365.0
        
        # Black-Scholes for put
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        put_price = strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return max(put_price, 0.01)  # Minimum $0.01
    
    def size_options_position(self,
                              portfolio_value: float,
                              spot_price: float,
                              max_allocation: float = None) -> int:
        """
        Calculate number of put contracts to sell.
        
        Args:
            portfolio_value: Current portfolio value
            spot_price: Current stock price
            max_allocation: Max % of portfolio (default from config)
            
        Returns:
            Number of contracts (each = 100 shares)
        """
        if max_allocation is None:
            max_allocation = self.options_config['max_allocation_pct']
        
        # Max dollar amount to allocate
        max_dollars = portfolio_value * max_allocation
        
        # Each contract represents 100 shares
        contract_value = spot_price * 100
        
        # Number of contracts
        num_contracts = int(max_dollars / contract_value)
        
        return max(num_contracts, 1)  # At least 1 contract
    
    def simulate_options_trades(self,
                                 windows: pd.DataFrame,
                                 prices: pd.DataFrame,
                                 initial_capital: float) -> pd.DataFrame:
        """
        Simulate put-selling strategy across event windows.
        
        Args:
            windows: Event windows DataFrame
            prices: AMZN price data
            initial_capital: Starting capital
            
        Returns:
            DataFrame with simulated trades
        """
        trades = []
        portfolio_value = initial_capital
        
        for _, window in windows.iterrows():
            entry_date = window['entry_date']
            exit_date = window['exit_date']
            event_date = window['event_date']
            
            # Check if dates in price data
            if entry_date not in prices.index or exit_date not in prices.index:
                continue
            
            # Entry price
            entry_price = prices.loc[entry_date, 'Adj Close']
            
            # Exit price
            exit_price = prices.loc[exit_date, 'Adj Close']
            
            # Calculate strike
            strike = self.calculate_strike_price(entry_price)
            
            # Size position
            num_contracts = self.size_options_position(portfolio_value, entry_price)
            
            # Estimate premium (simplified - would use real data in production)
            holding_days = (exit_date - entry_date).days
            premium_per_share = self.estimate_option_premium(
                strike, entry_price, dte=holding_days
            )
            
            # Total premium collected
            total_premium = premium_per_share * num_contracts * 100
            
            # Determine outcome
            if exit_price > strike:
                # Put expires OTM - keep full premium
                pnl = total_premium
                outcome = 'win'
            else:
                # Put assigned - forced to buy at strike (loss)
                loss_per_share = strike - exit_price
                assignment_loss = loss_per_share * num_contracts * 100
                pnl = total_premium - assignment_loss
                outcome = 'loss'
            
            # Update portfolio
            portfolio_value += pnl
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'event_type': window['event_type'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'strike': strike,
                'num_contracts': num_contracts,
                'premium_collected': total_premium,
                'pnl': pnl,
                'outcome': outcome,
                'portfolio_value': portfolio_value
            })
        
        return pd.DataFrame(trades)
    
    def calculate_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics for options strategy.
        
        Args:
            trades_df: Trades DataFrame
            
        Returns:
            Dictionary of metrics
        """
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['outcome'] == 'win'])
        
        total_premium = trades_df['premium_collected'].sum()
        total_pnl = trades_df['pnl'].sum()
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_premium = total_premium / total_trades if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_premium_collected': total_premium,
            'total_pnl': total_pnl,
            'avg_premium_per_trade': avg_premium,
            'avg_pnl_per_trade': avg_pnl,
            'final_portfolio_value': trades_df['portfolio_value'].iloc[-1] if len(trades_df) > 0 else 0
        }
        
        return metrics


if __name__ == "__main__":
    # Test options strategy
    from data_acquisition import DataAcquisition
    from signal_generator import SignalGenerator
    
    # Fetch data
    data_acq = DataAcquisition()
    dataset = data_acq.fetch_full_dataset()
    
    # Generate event windows
    signal_gen = SignalGenerator()
    _, windows = signal_gen.generate_signal_series(dataset['amzn_prices'].index)
    
    # Simulate options trades
    options_strat = OptionsStrategy()
    
    # Use only recent years (2012+) per research
    recent_windows = windows[windows['year'] >= 2012]
    
    trades = options_strat.simulate_options_trades(
        recent_windows,
        dataset['amzn_prices'],
        initial_capital=1000000
    )
    
    print("=== Options Strategy Simulation ===")
    print(trades[['entry_date', 'event_type', 'strike', 'premium_collected', 'pnl', 'outcome']])
    
    # Calculate metrics
    metrics = options_strat.calculate_metrics(trades)
    
    print("\n=== Performance Metrics ===")
    for key, value in metrics.items():
        if 'rate' in key or 'win' in key:
            print(f"{key}: {value*100:.1f}%")
        elif 'value' in key or 'pnl' in key or 'premium' in key:
            print(f"{key}: ${value:,.2f}")
        else:
            print(f"{key}: {value}")
