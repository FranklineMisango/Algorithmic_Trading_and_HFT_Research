"""
Rates Carry Strategy - Backtesting Module
"""

import pandas as pd
import numpy as np
import yaml


class RatesBacktester:
    """Backtest rates carry strategy"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tc_bps = self.config['backtesting']['transaction_cost_bps']
    
    def apply_transaction_costs(self, returns: pd.Series, weights: pd.DataFrame) -> pd.Series:
        """Apply transaction costs"""
        print("\n" + "="*60)
        print("BACKTESTING")
        print("="*60)
        
        turnover = weights.diff().abs().sum(axis=1)
        tc_returns = -turnover * self.tc_bps / 10000
        net_returns = returns + tc_returns
        
        print(f"Average daily turnover: {turnover.mean():.2%}")
        print(f"Gross Sharpe: {returns.mean() / returns.std() * np.sqrt(252):.2f}")
        print(f"Net Sharpe: {net_returns.mean() / net_returns.std() * np.sqrt(252):.2f}")
        
        return net_returns
    
    def calculate_metrics(self, returns: pd.Series):
        """Calculate performance metrics"""
        cumulative = (1 + returns).cumprod()
        
        metrics = {
            'total_return': cumulative.iloc[-1] - 1,
            'ann_return': returns.mean() * 252,
            'ann_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (cumulative / cumulative.cummax() - 1).min(),
            'win_rate': (returns > 0).sum() / len(returns),
        }
        
        print("\nPERFORMANCE METRICS")
        print("-" * 60)
        for key, value in metrics.items():
            print(f"{key:.<40} {value:>10.2%}" if abs(value) < 10 else f"{key:.<40} {value:>10.2f}")
        
        return metrics
    
    def run_backtest(self, returns: pd.Series, weights: pd.DataFrame):
        """Run backtest"""
        net_returns = self.apply_transaction_costs(returns, weights)
        metrics = self.calculate_metrics(net_returns)
        
        print("\n✓ Backtesting complete")
        return {'metrics': metrics, 'returns': net_returns}
