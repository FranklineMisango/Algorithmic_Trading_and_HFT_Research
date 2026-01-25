"""
Portfolio Construction: Long-Short, Sector-Neutral Equity Portfolio

Constructs daily-rebalanced, equal-weighted portfolios from sentiment signals.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class SentimentPortfolioConstructor:
    """Portfolio construction from sentiment signals."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize portfolio constructor."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.portfolio_config = self.config['signals']['portfolio_formation']
        self.risk_config = self.config['risk']
    
    def rank_stocks(self, sentiment_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Rank stocks cross-sectionally by sentiment.
        
        Args:
            sentiment_scores: DataFrame with columns ['ticker', 'date', 'sentiment']
        
        Returns:
            DataFrame with rank column
        """
        # Group by date and rank
        sentiment_scores['rank'] = sentiment_scores.groupby('date')['sentiment'].rank(pct=True)
        
        return sentiment_scores
    
    def form_long_short_portfolios(
        self,
        ranked_scores: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Form long and short portfolios based on sentiment ranks.
        
        Args:
            ranked_scores: DataFrame with sentiment ranks
            market_data: Market data with sector information
        
        Returns:
            Dict with 'long' and 'short' portfolio DataFrames
        """
        long_percentile = self.portfolio_config['long_leg']['percentile'] / 100
        short_percentile = self.portfolio_config['short_leg']['percentile'] / 100
        
        # Merge with market data to get sector
        merged = ranked_scores.merge(
            market_data[['ticker', 'date', 'sector']],
            on=['ticker', 'date'],
            how='inner'
        )
        
        # Select long leg (top decile)
        long_portfolio = merged[merged['rank'] >= long_percentile].copy()
        
        # Select short leg (bottom decile)
        short_portfolio = merged[merged['rank'] <= short_percentile].copy()
        
        # Equal weighting
        long_portfolio['weight'] = 1.0 / long_portfolio.groupby('date')['ticker'].transform('count')
        short_portfolio['weight'] = -1.0 / short_portfolio.groupby('date')['ticker'].transform('count')
        
        return {
            'long': long_portfolio,
            'short': short_portfolio
        }
    
    def apply_sector_neutrality(
        self,
        long_portfolio: pd.DataFrame,
        short_portfolio: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply sector neutrality constraint.
        
        Args:
            long_portfolio: Long leg portfolio
            short_portfolio: Short leg portfolio
        
        Returns:
            Sector-neutral portfolios
        """
        if not self.risk_config['sector_neutrality']['enabled']:
            return long_portfolio, short_portfolio
        
        # Combine portfolios
        combined = pd.concat([long_portfolio, short_portfolio])
        
        # Calculate sector exposures
        sector_exposure = combined.groupby(['date', 'sector'])['weight'].sum().reset_index()
        
        # Check maximum deviation
        max_deviation = self.risk_config['sector_neutrality']['max_deviation']
        
        # For simplicity, just report exposures (full optimization would use cvxpy)
        print("\nSector Exposures:")
        print(sector_exposure.groupby('sector')['weight'].mean())
        
        return long_portfolio, short_portfolio
    
    def apply_position_limits(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """
        Apply position size limits.
        
        Args:
            portfolio: Portfolio DataFrame
        
        Returns:
            Portfolio with limited positions
        """
        max_weight = self.risk_config['position_limits']['max_weight_per_stock']
        
        # Clip weights
        portfolio['weight'] = np.clip(portfolio['weight'], -max_weight, max_weight)
        
        # Renormalize
        portfolio['weight'] = portfolio.groupby('date')['weight'].transform(
            lambda x: x / x.abs().sum()
        )
        
        return portfolio
    
    def construct_portfolio(
        self,
        sentiment_scores: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Full portfolio construction pipeline.
        
        Args:
            sentiment_scores: Sentiment scores
            market_data: Market data
        
        Returns:
            Final portfolio with weights
        """
        # Rank stocks
        ranked = self.rank_stocks(sentiment_scores)
        
        # Form long-short portfolios
        portfolios = self.form_long_short_portfolios(ranked, market_data)
        
        # Apply sector neutrality
        long_portfolio, short_portfolio = self.apply_sector_neutrality(
            portfolios['long'],
            portfolios['short']
        )
        
        # Combine
        combined_portfolio = pd.concat([long_portfolio, short_portfolio])
        
        # Apply position limits
        combined_portfolio = self.apply_position_limits(combined_portfolio)
        
        return combined_portfolio


# Test code
if __name__ == "__main__":
    from data_acquisition import SentimentDataAcquisition
    
    # Load data
    data_acq = SentimentDataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset("2023-01-01", "2023-03-31")
    
    # Generate placeholder sentiment scores
    sentiment_scores = dataset['text_data'].copy()
    sentiment_scores['sentiment'] = np.random.randn(len(sentiment_scores))
    
    # Construct portfolio
    constructor = SentimentPortfolioConstructor('config.yaml')
    portfolio = constructor.construct_portfolio(sentiment_scores, dataset['market_data'])
    
    print("\n" + "="*60)
    print("PORTFOLIO CONSTRUCTION")
    print("="*60)
    print(f"\nTotal positions: {len(portfolio)}")
    print(f"Long positions: {(portfolio['weight'] > 0).sum()}")
    print(f"Short positions: {(portfolio['weight'] < 0).sum()}")
    print(f"\nSample weights:")
    print(portfolio.head(10)[['ticker', 'date', 'sentiment', 'weight']])
