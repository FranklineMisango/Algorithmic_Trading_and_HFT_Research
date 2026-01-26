import pandas as pd
import numpy as np

class PortfolioConstructor:
    def __init__(self, long_pct: float = 0.2, short_pct: float = 0.2, 
                 net_exposure: float = 0.0, sp1500_only_shorts: bool = True):
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.net_exposure = net_exposure
        self.sp1500_only_shorts = sp1500_only_shorts
    
    def construct_daily_portfolio(self, predictions_df: pd.DataFrame, 
                                   sp1500_list: list = None) -> pd.DataFrame:
        """Build equal-weighted long/short portfolio"""
        predictions_df = predictions_df.sort_values('predicted_return', ascending=False)
        n_stocks = len(predictions_df)
        
        # Long positions: top 20%
        n_long = int(n_stocks * self.long_pct)
        long_stocks = predictions_df.head(n_long).copy()
        long_stocks['position'] = 'long'
        long_stocks['weight'] = (1.0 + self.net_exposure) / n_long
        
        # Short positions: bottom 20%, S&P 1500 only
        short_candidates = predictions_df.tail(int(n_stocks * self.short_pct))
        if self.sp1500_only_shorts and sp1500_list:
            short_candidates = short_candidates[short_candidates['ticker'].isin(sp1500_list)]
        
        n_short = len(short_candidates)
        short_stocks = short_candidates.copy()
        short_stocks['position'] = 'short'
        short_stocks['weight'] = -(1.0 - self.net_exposure) / n_short if n_short > 0 else 0
        
        return pd.concat([long_stocks, short_stocks], ignore_index=True)
    
    def calculate_portfolio_return(self, portfolio_df: pd.DataFrame, 
                                    transaction_cost_bps: float = 12.5) -> float:
        """Calculate daily portfolio return with transaction costs"""
        gross_return = (portfolio_df['weight'] * portfolio_df['return']).sum()
        n_positions = len(portfolio_df)
        transaction_cost = (transaction_cost_bps / 10000) * n_positions
        return gross_return - transaction_cost
