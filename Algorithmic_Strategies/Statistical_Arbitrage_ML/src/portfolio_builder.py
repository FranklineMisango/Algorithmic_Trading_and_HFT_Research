"""
Portfolio Construction Module for Statistical Arbitrage Strategy

Builds market-neutral long/short portfolios based on ML predictions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger


class PortfolioBuilder:
    """
    Constructs market-neutral long/short portfolios.
    
    Key features:
    - Rank stocks by predicted returns
    - Select top/bottom stocks for long/short positions
    - Apply position sizing and risk limits
    - Exclude high-risk stocks
    """
    
    def __init__(
        self,
        n_long: int = 20,
        n_short: int = 20,
        max_position_size: float = 0.04,  # 4% max per position
        total_capital: float = 1_000_000,
        n_portfolios: int = 3  # Staggered portfolios
    ):
        """
        Initialize portfolio builder.
        
        Args:
            n_long: Number of long positions
            n_short: Number of short positions
            max_position_size: Maximum position size as fraction of capital
            total_capital: Total capital to allocate
            n_portfolios: Number of staggered portfolios
        """
        self.n_long = n_long
        self.n_short = n_short
        self.max_position_size = max_position_size
        self.total_capital = total_capital
        self.n_portfolios = n_portfolios
        
        # Capital per portfolio
        self.capital_per_portfolio = total_capital / n_portfolios
        
        logger.info(
            f"PortfolioBuilder initialized: "
            f"{n_long} long / {n_short} short positions, "
            f"max position size: {max_position_size*100:.1f}%, "
            f"{n_portfolios} staggered portfolios"
        )
    
    def rank_stocks(
        self,
        predictions: pd.Series,
        date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Rank stocks by predicted returns.
        
        Args:
            predictions: Series with ticker index and predicted returns
            date: Date for ranking
            
        Returns:
            DataFrame with tickers, predictions, and ranks
        """
        ranking = pd.DataFrame({
            'ticker': predictions.index,
            'predicted_return': predictions.values,
            'date': date
        })
        
        # Sort by predicted return (descending)
        ranking = ranking.sort_values('predicted_return', ascending=False)
        ranking['rank'] = range(1, len(ranking) + 1)
        
        return ranking
    
    def select_long_short(
        self,
        ranking: pd.DataFrame,
        exclude_tickers: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Select stocks for long and short positions.
        
        Args:
            ranking: DataFrame with stock rankings
            exclude_tickers: Tickers to exclude from selection
            
        Returns:
            Tuple of (long_tickers, short_tickers)
        """
        if exclude_tickers is None:
            exclude_tickers = []
        
        # Filter out excluded tickers
        valid_ranking = ranking[~ranking['ticker'].isin(exclude_tickers)]
        
        if len(valid_ranking) < (self.n_long + self.n_short):
            logger.warning(
                f"Not enough valid stocks ({len(valid_ranking)}) "
                f"for {self.n_long + self.n_short} positions"
            )
        
        # Select top N for long
        long_tickers = valid_ranking.head(self.n_long)['ticker'].tolist()
        
        # Select bottom N for short
        short_tickers = valid_ranking.tail(self.n_short)['ticker'].tolist()
        
        logger.info(
            f"Selected {len(long_tickers)} long and {len(short_tickers)} short positions"
        )
        
        return long_tickers, short_tickers
    
    def calculate_position_sizes(
        self,
        tickers: List[str],
        prices: pd.Series,
        side: str = 'long',
        weighting: str = 'equal'
    ) -> pd.DataFrame:
        """
        Calculate position sizes for selected stocks.
        
        Args:
            tickers: List of tickers to position
            prices: Series with current prices (ticker index)
            side: 'long' or 'short'
            weighting: 'equal' or 'predicted' (future enhancement)
            
        Returns:
            DataFrame with position details
        """
        n_positions = len(tickers)
        
        if n_positions == 0:
            return pd.DataFrame()
        
        if weighting == 'equal':
            # Equal weight allocation
            capital_per_position = self.capital_per_portfolio / (self.n_long + self.n_short)
            
            positions = []
            for ticker in tickers:
                if ticker not in prices.index:
                    logger.warning(f"Price not available for {ticker}, skipping")
                    continue
                
                price = prices[ticker]
                
                # Handle case where price might be a Series (duplicate index)
                if isinstance(price, pd.Series):
                    price = price.iloc[0]  # Take first value
                
                # Ensure price is a scalar
                price = float(price)
                
                # Calculate shares based on capital allocation
                shares = int(capital_per_position / price)
                actual_capital = shares * price
                
                # Check position size limit
                position_size_pct = actual_capital / self.total_capital
                if position_size_pct > self.max_position_size:
                    # Reduce shares to meet limit
                    max_shares = int((self.max_position_size * self.total_capital) / price)
                    shares = max_shares
                    actual_capital = shares * price
                    position_size_pct = actual_capital / self.total_capital
                    logger.debug(
                        f"Position size limited for {ticker}: {position_size_pct*100:.2f}%"
                    )
                
                positions.append({
                    'ticker': ticker,
                    'side': side,
                    'shares': shares if side == 'long' else -shares,
                    'price': price,
                    'capital': actual_capital,
                    'position_size_pct': position_size_pct
                })
            
            return pd.DataFrame(positions)
        
        else:
            raise NotImplementedError(f"Weighting scheme '{weighting}' not implemented")
    
    def build_portfolio(
        self,
        predictions: pd.Series,
        prices: pd.Series,
        date: pd.Timestamp,
        exclude_tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Build complete portfolio with long and short positions.
        
        Args:
            predictions: Predicted returns (ticker index)
            prices: Current prices (ticker index)
            date: Portfolio date
            exclude_tickers: Tickers to exclude
            
        Returns:
            DataFrame with full portfolio details
        """
        logger.info(f"Building portfolio for {date.strftime('%Y-%m-%d')}")
        
        # Rank stocks
        ranking = self.rank_stocks(predictions, date)
        
        # Select long/short
        long_tickers, short_tickers = self.select_long_short(ranking, exclude_tickers)
        
        # Calculate position sizes
        long_positions = self.calculate_position_sizes(
            long_tickers, prices, side='long', weighting='equal'
        )
        
        short_positions = self.calculate_position_sizes(
            short_tickers, prices, side='short', weighting='equal'
        )
        
        # Combine positions
        portfolio = pd.concat([long_positions, short_positions], ignore_index=True)
        portfolio['date'] = date
        
        # Calculate portfolio metrics
        long_exposure = long_positions['capital'].sum() if not long_positions.empty else 0
        short_exposure = short_positions['capital'].sum() if not short_positions.empty else 0
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure
        
        logger.info(
            f"Portfolio built: "
            f"Long ${long_exposure:,.0f}, "
            f"Short ${short_exposure:,.0f}, "
            f"Net ${net_exposure:,.0f}, "
            f"Gross ${gross_exposure:,.0f}"
        )
        
        return portfolio
    
    def apply_risk_filters(
        self,
        universe: pd.DataFrame,
        prices: pd.Series,
        volumes: pd.Series,
        min_price: float = 5.0,
        min_volume: float = 1_000_000
    ) -> List[str]:
        """
        Apply risk filters to exclude problematic stocks.
        
        Args:
            universe: DataFrame with stock metadata
            prices: Current prices
            volumes: Recent average volumes
            min_price: Minimum price (exclude penny stocks)
            min_volume: Minimum average volume
            
        Returns:
            List of tickers to exclude
        """
        exclude_tickers = []
        
        # Penny stock filter
        penny_stocks = prices[prices < min_price].index.tolist()
        exclude_tickers.extend(penny_stocks)
        
        # Low volume filter
        illiquid_stocks = volumes[volumes < min_volume].index.tolist()
        exclude_tickers.extend(illiquid_stocks)
        
        # Remove duplicates
        exclude_tickers = list(set(exclude_tickers))
        
        logger.info(
            f"Risk filters: excluded {len(exclude_tickers)} stocks "
            f"(penny: {len(penny_stocks)}, illiquid: {len(illiquid_stocks)})"
        )
        
        return exclude_tickers
    
    def rebalance_portfolio(
        self,
        current_portfolio: pd.DataFrame,
        target_portfolio: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate trades needed to rebalance from current to target portfolio.
        
        Args:
            current_portfolio: Current positions
            target_portfolio: Desired positions
            
        Returns:
            DataFrame with trades to execute
        """
        # Merge current and target
        current = current_portfolio.set_index('ticker')['shares'].to_dict()
        target = target_portfolio.set_index('ticker')['shares'].to_dict()
        
        # Calculate trades
        all_tickers = set(current.keys()).union(set(target.keys()))
        trades = []
        
        for ticker in all_tickers:
            current_shares = current.get(ticker, 0)
            target_shares = target.get(ticker, 0)
            trade_shares = target_shares - current_shares
            
            if trade_shares != 0:
                trades.append({
                    'ticker': ticker,
                    'current_shares': current_shares,
                    'target_shares': target_shares,
                    'trade_shares': trade_shares,
                    'action': 'BUY' if trade_shares > 0 else 'SELL'
                })
        
        trades_df = pd.DataFrame(trades)
        
        logger.info(f"Rebalancing: {len(trades_df)} trades required")
        
        return trades_df
    
    def calculate_portfolio_exposure(
        self,
        portfolio: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio exposure metrics.
        
        Args:
            portfolio: Portfolio DataFrame
            
        Returns:
            Dictionary of exposure metrics
        """
        long_mask = portfolio['side'] == 'long'
        short_mask = portfolio['side'] == 'short'
        
        long_capital = portfolio[long_mask]['capital'].sum()
        short_capital = portfolio[short_mask]['capital'].sum()
        
        metrics = {
            'long_exposure': long_capital,
            'short_exposure': short_capital,
            'net_exposure': long_capital - short_capital,
            'gross_exposure': long_capital + short_capital,
            'n_long': long_mask.sum(),
            'n_short': short_mask.sum(),
            'total_positions': len(portfolio)
        }
        
        return metrics
    
    def generate_staggered_portfolios(
        self,
        predictions_history: Dict[pd.Timestamp, pd.Series],
        prices_history: Dict[pd.Timestamp, pd.Series],
        holding_period: int = 3
    ) -> List[pd.DataFrame]:
        """
        Generate staggered portfolios that start on different days.
        
        Args:
            predictions_history: Dictionary mapping dates to predictions
            prices_history: Dictionary mapping dates to prices
            holding_period: Days to hold positions
            
        Returns:
            List of portfolio DataFrames for each stagger
        """
        portfolios = []
        dates = sorted(predictions_history.keys())
        
        for stagger_idx in range(self.n_portfolios):
            # Select dates for this stagger
            stagger_dates = dates[stagger_idx::self.n_portfolios]
            
            stagger_portfolio = []
            for date in stagger_dates:
                portfolio = self.build_portfolio(
                    predictions_history[date],
                    prices_history[date],
                    date
                )
                portfolio['stagger'] = stagger_idx
                portfolio['close_date'] = date + pd.Timedelta(days=holding_period)
                stagger_portfolio.append(portfolio)
            
            if stagger_portfolio:
                portfolios.append(pd.concat(stagger_portfolio, ignore_index=True))
            
            logger.info(
                f"Stagger {stagger_idx}: {len(stagger_portfolio)} portfolio iterations"
            )
        
        return portfolios


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample predictions and prices
    tickers = [f'STOCK{i}' for i in range(100)]
    predictions = pd.Series(
        np.random.randn(100) * 0.02,  # Random returns
        index=tickers
    )
    prices = pd.Series(
        np.random.uniform(10, 200, 100),
        index=tickers
    )
    
    # Build portfolio
    builder = PortfolioBuilder(n_long=20, n_short=20)
    portfolio = builder.build_portfolio(
        predictions,
        prices,
        pd.Timestamp('2024-01-15')
    )
    
    print(f"\nPortfolio shape: {portfolio.shape}")
    print(f"\nPortfolio columns: {portfolio.columns.tolist()}")
    print(f"\nSample positions:\n{portfolio.head(10)}")
    
    # Calculate exposure metrics
    metrics = builder.calculate_portfolio_exposure(portfolio)
    print(f"\nExposure metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:,.2f}")
