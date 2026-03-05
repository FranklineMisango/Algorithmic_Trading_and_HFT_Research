"""
Portfolio construction for multi-asset geopolitical strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict
import yaml


class PortfolioConstructor:
    """Construct multi-asset portfolio based on geopolitical signals."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.allocation = self.config['portfolio']['allocation']
        self.max_position = self.config['portfolio']['max_position_size']
    
    def construct_portfolio(self, signals: pd.DataFrame, market_data: Dict) -> pd.DataFrame:
        """Construct portfolio weights based on signals."""
        
        # Initialize weights dataframe
        weights = pd.DataFrame(index=signals.index)
        
        # Get position multiplier (0-1 based on risk level)
        multiplier = signals['position_multiplier']
        
        # Long positions
        weights['long_energy'] = multiplier * self.allocation['long_energy']
        weights['long_defense'] = multiplier * self.allocation['long_defense']
        weights['long_treasuries'] = multiplier * self.allocation['long_treasuries']
        weights['long_fx_exporters'] = multiplier * self.allocation['long_fx_exporters']
        
        # Short positions (negative weights)
        weights['short_transport'] = -multiplier * self.allocation['short_transport']
        weights['short_asia_equity'] = -multiplier * self.allocation['short_asia_equity']
        weights['short_em_bonds'] = -multiplier * self.allocation['short_em_bonds']
        weights['short_fx_importers'] = -multiplier * self.allocation['short_fx_importers']
        
        # Cash (remainder)
        weights['cash'] = 1 - weights.abs().sum(axis=1)
        
        return weights
    
    def calculate_portfolio_returns(self, weights: pd.DataFrame, 
                                   returns: Dict[str, pd.Series]) -> pd.Series:
        """Calculate portfolio returns given weights and asset returns."""
        
        portfolio_returns = pd.Series(0.0, index=weights.index)
        
        # Map weights to actual returns
        weight_return_map = {
            'long_energy': 'energy',
            'long_defense': 'defense',
            'long_treasuries': 'treasuries',
            'long_fx_exporters': 'fx_exporters',
            'short_transport': 'transport',
            'short_asia_equity': 'asia_equity',
            'short_em_bonds': 'em_bonds',
            'short_fx_importers': 'fx_importers'
        }
        
        for weight_col, return_key in weight_return_map.items():
            if return_key in returns:
                portfolio_returns += weights[weight_col] * returns[return_key]
        
        return portfolio_returns
    
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate portfolio performance metrics."""
        
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        metrics = {
            'total_return': total_return * 100,
            'annualized_return': ann_return * 100,
            'annualized_volatility': ann_vol * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'num_trades': len(returns)
        }
        
        return metrics


if __name__ == "__main__":
    # Test portfolio constructor
    from data_acquisition import DataAcquisition
    from signal_generator import SignalGenerator
    
    print("Testing Portfolio Constructor...")
    
    acq = DataAcquisition()
    data = acq.fetch_all_data()
    
    generator = SignalGenerator()
    signals = generator.generate_master_signal(data)
    
    constructor = PortfolioConstructor()
    weights = constructor.construct_portfolio(signals, data['market'])
    
    print("\nPortfolio Weights Summary:")
    print(weights.describe())
    
    print("\nAverage Allocation:")
    print(weights.mean().round(3))
