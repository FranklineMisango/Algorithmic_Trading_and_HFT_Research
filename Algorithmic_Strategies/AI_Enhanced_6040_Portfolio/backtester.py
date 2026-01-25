"""
Backtester Module for AI-Enhanced 60/40 Portfolio

This module implements portfolio backtesting with performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class PortfolioBacktester:
    """Backtest portfolio strategies and calculate performance metrics."""
    
    def __init__(self, config: Dict):
        """
        Initialize backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.initial_capital = config['backtest']['initial_capital']
        self.transaction_cost = config['portfolio']['transaction_cost']
        self.risk_free_rate = config['portfolio']['risk_free_rate']
        
    def calculate_portfolio_returns(self,
                                    allocations: pd.DataFrame,
                                    returns: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio returns from allocations and asset returns.
        
        Args:
            allocations: DataFrame with portfolio allocations
            returns: DataFrame with asset returns
            
        Returns:
            Series with portfolio returns
        """
        # Align indices
        common_idx = allocations.index.intersection(returns.index)
        alloc = allocations.loc[common_idx]
        rets = returns.loc[common_idx]
        
        # Calculate weighted returns
        portfolio_returns = (alloc * rets).sum(axis=1)
        
        return portfolio_returns
    
    def calculate_transaction_costs(self,
                                    allocations: pd.DataFrame) -> pd.Series:
        """
        Calculate transaction costs from portfolio rebalancing.
        
        Args:
            allocations: DataFrame with portfolio allocations
            
        Returns:
            Series with transaction costs
        """
        # Calculate changes in allocations
        allocation_changes = allocations.diff().abs()
        
        # Sum absolute changes and multiply by cost
        total_turnover = allocation_changes.sum(axis=1)
        costs = total_turnover * self.transaction_cost
        
        # First period has no transaction cost
        costs.iloc[0] = 0
        
        return costs
    
    def backtest_strategy(self,
                         allocations: pd.DataFrame,
                         returns: pd.DataFrame,
                         prices: pd.DataFrame) -> pd.DataFrame:
        """
        Backtest a portfolio strategy.
        
        Args:
            allocations: DataFrame with portfolio allocations
            returns: DataFrame with asset returns
            prices: DataFrame with asset prices
            
        Returns:
            DataFrame with portfolio value over time
        """
        # Calculate portfolio returns
        portfolio_returns = self.calculate_portfolio_returns(allocations, returns)
        
        # Calculate transaction costs
        transaction_costs = self.calculate_transaction_costs(allocations)
        
        # Net returns after costs
        net_returns = portfolio_returns - transaction_costs
        
        # Calculate cumulative portfolio value
        portfolio_value = self.initial_capital * (1 + net_returns).cumprod()
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Portfolio_Value': portfolio_value,
            'Returns': net_returns,
            'Gross_Returns': portfolio_returns,
            'Transaction_Costs': transaction_costs
        })
        
        return results
    
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 12) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Series with returns
            periods_per_year: Number of periods per year (12 for monthly)
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - self.risk_free_rate / periods_per_year
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        
        return sharpe
    
    def calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int = 12) -> float:
        """
        Calculate annualized Sortino ratio.
        
        Args:
            returns: Series with returns
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - self.risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
        
        return sortino
    
    def calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            portfolio_value: Series with portfolio value over time
            
        Returns:
            Maximum drawdown (as a positive percentage)
        """
        cummax = portfolio_value.cummax()
        drawdown = (portfolio_value - cummax) / cummax
        max_dd = abs(drawdown.min())
        
        return max_dd
    
    def calculate_cagr(self, portfolio_value: pd.Series, periods_per_year: int = 12) -> float:
        """
        Calculate Compound Annual Growth Rate.
        
        Args:
            portfolio_value: Series with portfolio value over time
            periods_per_year: Number of periods per year
            
        Returns:
            CAGR
        """
        n_periods = len(portfolio_value)
        n_years = n_periods / periods_per_year
        
        if n_years == 0:
            return 0.0
        
        total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0]
        cagr = (total_return ** (1 / n_years)) - 1
        
        return cagr
    
    def calculate_calmar_ratio(self, cagr: float, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            cagr: Compound Annual Growth Rate
            max_drawdown: Maximum drawdown
            
        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0
        
        return cagr / max_drawdown
    
    def calculate_volatility(self, returns: pd.Series, periods_per_year: int = 12) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Series with returns
            periods_per_year: Number of periods per year
            
        Returns:
            Annualized volatility
        """
        return returns.std() * np.sqrt(periods_per_year)
    
    def calculate_all_metrics(self, backtest_results: pd.DataFrame) -> Dict:
        """
        Calculate all performance metrics.
        
        Args:
            backtest_results: DataFrame from backtest_strategy
            
        Returns:
            Dictionary with all metrics
        """
        returns = backtest_results['Returns']
        portfolio_value = backtest_results['Portfolio_Value']
        
        metrics = {
            'Sharpe Ratio': self.calculate_sharpe_ratio(returns),
            'Sortino Ratio': self.calculate_sortino_ratio(returns),
            'Max Drawdown': self.calculate_max_drawdown(portfolio_value),
            'CAGR': self.calculate_cagr(portfolio_value),
            'Volatility': self.calculate_volatility(returns),
            'Total Return': (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1),
            'Final Value': portfolio_value.iloc[-1],
            'Avg Monthly Return': returns.mean(),
            'Monthly Volatility': returns.std(),
            'Best Month': returns.max(),
            'Worst Month': returns.min(),
            'Positive Months': (returns > 0).sum() / len(returns),
            'Total Transaction Costs': backtest_results['Transaction_Costs'].sum()
        }
        
        # Calculate Calmar ratio
        metrics['Calmar Ratio'] = self.calculate_calmar_ratio(
            metrics['CAGR'], 
            metrics['Max Drawdown']
        )
        
        return metrics
    
    def compare_strategies(self,
                          strategies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            strategies: Dictionary of strategy_name: backtest_results
            
        Returns:
            DataFrame comparing all strategies
        """
        comparison = {}
        
        for name, results in strategies.items():
            metrics = self.calculate_all_metrics(results)
            comparison[name] = metrics
        
        comparison_df = pd.DataFrame(comparison).T
        
        return comparison_df
    
    def create_benchmark_strategy(self,
                                 returns: pd.DataFrame,
                                 prices: pd.DataFrame,
                                 benchmark_ticker: str = 'SPY') -> pd.DataFrame:
        """
        Create a buy-and-hold benchmark strategy.
        
        Args:
            returns: DataFrame with asset returns
            prices: DataFrame with asset prices
            benchmark_ticker: Ticker for benchmark
            
        Returns:
            Backtest results for benchmark
        """
        # Create 100% allocation to benchmark
        allocations = pd.DataFrame(
            0.0,
            index=returns.index,
            columns=returns.columns
        )
        allocations[benchmark_ticker] = 1.0
        
        # Backtest
        results = self.backtest_strategy(allocations, returns, prices)
        
        return results
    
    def create_traditional_6040(self,
                               returns: pd.DataFrame,
                               prices: pd.DataFrame,
                               stock_ticker: str = 'SPY',
                               bond_ticker: str = 'TLT') -> pd.DataFrame:
        """
        Create a traditional 60/40 portfolio.
        
        Args:
            returns: DataFrame with asset returns
            prices: DataFrame with asset prices
            stock_ticker: Stock ticker
            bond_ticker: Bond ticker
            
        Returns:
            Backtest results for 60/40
        """
        # Create 60/40 allocation
        allocations = pd.DataFrame(
            0.0,
            index=returns.index,
            columns=returns.columns
        )
        allocations[stock_ticker] = 0.6
        allocations[bond_ticker] = 0.4
        
        # Backtest
        results = self.backtest_strategy(allocations, returns, prices)
        
        return results
    
    def plot_portfolio_value(self,
                            strategies: Dict[str, pd.DataFrame],
                            figsize: Tuple[int, int] = (14, 7)):
        """
        Plot portfolio value over time for multiple strategies.
        
        Args:
            strategies: Dictionary of strategy_name: backtest_results
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for name, results in strategies.items():
            plt.plot(results.index, results['Portfolio_Value'], label=name, linewidth=2)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_drawdown(self,
                     strategies: Dict[str, pd.DataFrame],
                     figsize: Tuple[int, int] = (14, 7)):
        """
        Plot drawdown over time for multiple strategies.
        
        Args:
            strategies: Dictionary of strategy_name: backtest_results
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for name, results in strategies.items():
            portfolio_value = results['Portfolio_Value']
            cummax = portfolio_value.cummax()
            drawdown = (portfolio_value - cummax) / cummax
            
            plt.plot(drawdown.index, drawdown * 100, label=name, linewidth=2)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_monthly_returns(self,
                           backtest_results: pd.DataFrame,
                           figsize: Tuple[int, int] = (14, 7)):
        """
        Plot monthly returns distribution.
        
        Args:
            backtest_results: Backtest results DataFrame
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        returns = backtest_results['Returns']
        
        # Histogram
        ax1.hist(returns * 100, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(returns.mean() * 100, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {returns.mean()*100:.2f}%')
        ax1.set_xlabel('Monthly Return (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Monthly Returns', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Time series
        ax2.bar(returns.index, returns * 100, 
               color=['g' if r > 0 else 'r' for r in returns])
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Monthly Return (%)', fontsize=12)
        ax2.set_title('Monthly Returns Over Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def plot_allocations(self,
                        allocations: pd.DataFrame,
                        figsize: Tuple[int, int] = (14, 7)):
        """
        Plot portfolio allocations over time.
        
        Args:
            allocations: DataFrame with allocations
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Stacked area plot
        plt.stackplot(allocations.index, 
                     allocations.T, 
                     labels=allocations.columns,
                     alpha=0.8)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Allocation', fontsize=12)
        plt.title('Portfolio Allocations Over Time', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()


if __name__ == "__main__":
    # Test the module
    import yaml
    from data_acquisition import DataAcquisition
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Fetch data
    data_acq = DataAcquisition(config)
    prices, returns, indicators = data_acq.get_full_dataset()
    
    # Create backtester
    backtester = PortfolioBacktester(config)
    
    # Test with benchmark
    benchmark_results = backtester.create_benchmark_strategy(returns, prices, 'SPY')
    benchmark_metrics = backtester.calculate_all_metrics(benchmark_results)
    
    print("\n" + "="*50)
    print("Benchmark (Buy & Hold SPY) Performance")
    print("="*50)
    for metric, value in benchmark_metrics.items():
        if isinstance(value, float):
            if 'Ratio' in metric or 'Return' in metric or 'Volatility' in metric:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: ${value:,.2f}" if 'Value' in metric or 'Cost' in metric 
                      else f"{metric}: {value:.4f}")
