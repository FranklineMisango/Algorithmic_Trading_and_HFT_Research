"""
Backtester for AI Economy Score Predictor Strategy
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict


class Backtester:
    """Backtest trading strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.costs = self.config['execution']['costs']
    
    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        initial_capital: float = 10000000
    ) -> Dict:
        """
        Backtest strategy with transaction costs.
        
        Args:
            signals: DataFrame with date and signal columns
            prices: DataFrame with asset prices
            initial_capital: Starting capital
            
        Returns:
            Dict with portfolio, trades, metrics
        """
        # Merge signals and prices
        merged = signals.merge(prices, on='date', how='inner')
        merged = merged.sort_values('date')
        
        # Initialize tracking
        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        
        for idx, row in merged.iterrows():
            date = row['date']
            signal = row['signal']
            price = row['close']
            
            # Current position value
            position_value = shares * price
            portfolio_value = cash + position_value
            
            # Rebalance if signal changed
            if signal != 0 and shares == 0:
                # Enter position
                trade_value = cash * 0.99  # Reserve for costs
                costs = trade_value * (self.costs['etf_commission_bps'] + self.costs['etf_slippage_bps']) / 10000
                shares = (trade_value - costs) / price
                cash -= (shares * price + costs)
                
                trades.append({
                    'date': date,
                    'action': 'buy' if signal > 0 else 'short',
                    'price': price,
                    'shares': shares,
                    'value': shares * price,
                    'costs': costs
                })
            
            elif signal == 0 and shares != 0:
                # Exit position
                proceeds = shares * price
                costs = proceeds * (self.costs['etf_commission_bps'] + self.costs['etf_slippage_bps']) / 10000
                cash += (proceeds - costs)
                
                trades.append({
                    'date': date,
                    'action': 'sell',
                    'price': price,
                    'shares': shares,
                    'value': shares * price,
                    'costs': costs
                })
                
                shares = 0
            
            portfolio_values.append({
                'date': date,
                'portfolio_value': cash + shares * price,
                'cash': cash,
                'position_value': shares * price,
                'signal': signal
            })
        
        portfolio_df = pd.DataFrame(portfolio_values)
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] - initial_capital) / initial_capital
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(4) if len(returns) > 0 else 0  # Quarterly data
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        metrics = {
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd * 100,
            'num_trades': len(trades_df),
            'total_costs': trades_df['costs'].sum() if len(trades_df) > 0 else 0
        }
        
        return {
            'portfolio': portfolio_df,
            'trades': trades_df,
            'metrics': metrics
        }


if __name__ == "__main__":
    bt = Backtester('config.yaml')
    
    # Test data
    dates = pd.date_range('2020-01-01', periods=16, freq='Q')
    signals = pd.DataFrame({
        'date': dates,
        'signal': [1, 1, 1, 0, -1, -1, 0, 1, 1, 0, 0, 1, -1, 0, 1, 0]
    })
    prices = pd.DataFrame({
        'date': dates,
        'close': np.cumsum(np.random.normal(0.5, 2, 16)) + 100
    })
    
    results = bt.run_backtest(signals, prices)
    print(f"Metrics: {results['metrics']}")
