"""
Commodity Carry Strategy - Backtesting Module
"""

import pandas as pd
import numpy as np
import yaml


class CommodityBacktester:
    """Backtest commodity carry strategy"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tc_bps = self.config['backtesting']['transaction_cost_bps']
    
    def run_backtest(self, returns: pd.Series, weights: pd.DataFrame):
        """Run backtest"""
        turnover = weights.diff().abs().sum(axis=1)
        tc_returns = -turnover * self.tc_bps / 10000
        net_returns = returns + tc_returns
        
        cumulative = (1 + net_returns).cumprod()
        
        metrics = {
            'total_return': cumulative.iloc[-1] - 1,
            'sharpe_ratio': net_returns.mean() / net_returns.std() * np.sqrt(252),
            'max_drawdown': (cumulative / cumulative.cummax() - 1).min(),
        }
        
        print("\nCOMMODITY CARRY BACKTEST")
        print("-" * 60)
        for key, value in metrics.items():
            print(f"{key}: {value:.2%}" if abs(value) < 10 else f"{key}: {value:.2f}")
        
        return {'metrics': metrics, 'returns': net_returns}
