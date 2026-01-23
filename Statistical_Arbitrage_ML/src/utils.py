"""Utility functions for the statistical arbitrage strategy."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / 252)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio (downside deviation).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    
    return (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: Series of portfolio values
        
    Returns:
        Maximum drawdown as negative percentage
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def calculate_calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Args:
        returns: Series of returns
        equity_curve: Series of portfolio values
        
    Returns:
        Calmar ratio
    """
    annual_return = returns.mean() * 252
    max_dd = abs(calculate_max_drawdown(equity_curve))
    
    return annual_return / max_dd if max_dd > 0 else 0


def calculate_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate information ratio.
    
    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information ratio
    """
    excess_returns = strategy_returns - benchmark_returns
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency string."""
    return f"${value:,.{decimals}f}"


def get_trading_dates(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Get list of trading dates (excluding weekends).
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of trading dates
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    # Filter out weekends (Saturday=5, Sunday=6)
    trading_dates = [d for d in dates if d.weekday() < 5]
    return trading_dates


def align_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, join_type: str = 'inner') -> tuple:
    """
    Align two dataframes by index.
    
    Args:
        df1: First dataframe
        df2: Second dataframe
        join_type: Type of join ('inner', 'outer', 'left', 'right')
        
    Returns:
        Tuple of aligned dataframes
    """
    aligned_df1, aligned_df2 = df1.align(df2, join=join_type, axis=0)
    return aligned_df1, aligned_df2


def winsorize_series(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """
    Winsorize series to handle outliers.
    
    Args:
        series: Input series
        lower_pct: Lower percentile to clip
        upper_pct: Upper percentile to clip
        
    Returns:
        Winsorized series
    """
    lower_bound = series.quantile(lower_pct)
    upper_bound = series.quantile(upper_pct)
    return series.clip(lower=lower_bound, upper=upper_bound)


def calculate_turnover(portfolio_current: pd.DataFrame, portfolio_target: pd.DataFrame) -> float:
    """
    Calculate portfolio turnover.
    
    Args:
        portfolio_current: Current portfolio
        portfolio_target: Target portfolio
        
    Returns:
        Turnover as percentage of portfolio value
    """
    current_positions = portfolio_current.set_index('ticker')['capital'].to_dict()
    target_positions = portfolio_target.set_index('ticker')['capital'].to_dict()
    
    all_tickers = set(current_positions.keys()).union(set(target_positions.keys()))
    
    turnover = 0
    for ticker in all_tickers:
        current_value = current_positions.get(ticker, 0)
        target_value = target_positions.get(ticker, 0)
        turnover += abs(target_value - current_value)
    
    total_value = sum(current_positions.values())
    return turnover / total_value if total_value > 0 else 0


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    shuffle: bool = False
) -> tuple:
    """
    Split dataframe into train and test sets.
    
    Args:
        df: Input dataframe
        test_size: Fraction of data for testing
        shuffle: Whether to shuffle before splitting
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if shuffle:
        df = df.sample(frac=1, random_state=42)
    
    split_idx = int(len(df) * (1 - test_size))
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample returns
    returns = pd.Series(np.random.randn(252) * 0.01)
    equity = (1 + returns).cumprod() * 100000
    
    print("Performance Metrics:")
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(returns):.2f}")
    print(f"Sortino Ratio: {calculate_sortino_ratio(returns):.2f}")
    print(f"Max Drawdown: {format_percentage(calculate_max_drawdown(equity))}")
    print(f"Calmar Ratio: {calculate_calmar_ratio(returns, equity):.2f}")
