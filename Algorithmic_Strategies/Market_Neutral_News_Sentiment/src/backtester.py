import pandas as pd
import numpy as np
from typing import Dict
from .portfolio import PortfolioConstructor

class Backtester:
    def __init__(self, portfolio_constructor: PortfolioConstructor):
        self.portfolio_constructor = portfolio_constructor
        self.results = None
    
    def run(self, predictions_df: pd.DataFrame, returns_df: pd.DataFrame, 
            sp1500_constituents: Dict[str, list]) -> pd.DataFrame:
        """Run daily rebalancing backtest"""
        daily_returns = []
        
        for date in predictions_df['date'].unique():
            day_predictions = predictions_df[predictions_df['date'] == date]
            sp1500_list = sp1500_constituents.get(date, [])
            
            portfolio = self.portfolio_constructor.construct_daily_portfolio(
                day_predictions, sp1500_list
            )
            
            next_day_returns = returns_df[returns_df['date'] == date]
            portfolio = portfolio.merge(next_day_returns[['ticker', 'return']], on='ticker', how='left')
            
            daily_return = self.portfolio_constructor.calculate_portfolio_return(portfolio)
            daily_returns.append({'date': date, 'portfolio_return': daily_return})
        
        self.results = pd.DataFrame(daily_returns)
        return self.results
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        returns = self.results['portfolio_return']
        
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': cumulative.iloc[-1] - 1
        }
