"""
Backtesting module for Foreign Market Lead-Lag ML Strategy.
Calculates performance metrics and risk-adjusted returns.
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class Backtester:
    """Comprehensive backtesting and performance analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.benchmark_ticker = config['backtesting']['benchmark']
        
    def calculate_metrics(self, results_df: pd.DataFrame, 
                         benchmark_returns: pd.Series = None) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            results_df: DataFrame with portfolio simulation results
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Calculating performance metrics...")
        
        returns = results_df['net_return']
        
        # Basic metrics
        total_return = (results_df['portfolio_value'].iloc[-1] / 
                       results_df['portfolio_value'].iloc[0]) - 1
        
        num_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / num_years) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Average turnover
        avg_turnover = results_df['turnover'].mean()
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_turnover': avg_turnover,
            'num_trades': len(results_df),
            'avg_long_positions': results_df['num_long'].mean(),
            'avg_short_positions': results_df['num_short'].mean()
        }
        
        # Benchmark comparison
        if benchmark_returns is not None:
            aligned_dates = returns.index.intersection(benchmark_returns.index)
            if len(aligned_dates) > 0:
                bench_returns = benchmark_returns.loc[aligned_dates]
                strat_returns = returns.loc[aligned_dates]
                
                bench_total = (1 + bench_returns).prod() - 1
                bench_annual = (1 + bench_total) ** (1 / num_years) - 1
                bench_vol = bench_returns.std() * np.sqrt(252)
                
                alpha = annual_return - bench_annual
                beta = np.cov(strat_returns, bench_returns)[0, 1] / np.var(bench_returns)
                
                metrics['benchmark_return'] = bench_annual
                metrics['benchmark_volatility'] = bench_vol
                metrics['alpha'] = alpha
                metrics['beta'] = beta
                metrics['information_ratio'] = alpha / (strat_returns - bench_returns).std() / np.sqrt(252)
        
        logger.info("Metrics calculation complete")
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print performance metrics in formatted table."""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\nReturns:")
        print(f"  Total Return:        {metrics['total_return']:>10.2%}")
        print(f"  Annual Return:       {metrics['annual_return']:>10.2%}")
        
        print(f"\nRisk:")
        print(f"  Volatility:          {metrics['volatility']:>10.2%}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        
        print(f"\nRisk-Adjusted:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        
        print(f"\nTrading:")
        print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"  Avg Turnover:        {metrics['avg_turnover']:>10.2f}")
        print(f"  Avg Long Positions:  {metrics['avg_long_positions']:>10.1f}")
        print(f"  Avg Short Positions: {metrics['avg_short_positions']:>10.1f}")
        
        if 'alpha' in metrics:
            print(f"\nBenchmark Comparison:")
            print(f"  Benchmark Return:    {metrics['benchmark_return']:>10.2%}")
            print(f"  Alpha:               {metrics['alpha']:>10.2%}")
            print(f"  Beta:                {metrics['beta']:>10.2f}")
            print(f"  Information Ratio:   {metrics['information_ratio']:>10.2f}")
        
        print("="*60 + "\n")
    
    def plot_performance(self, results_df: pd.DataFrame, 
                        benchmark_returns: pd.Series = None,
                        save_path: str = None):
        """
        Create comprehensive performance visualization.
        
        Args:
            results_df: Portfolio simulation results
            benchmark_returns: Optional benchmark returns
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Cumulative returns
        cumulative = (1 + results_df['net_return']).cumprod()
        axes[0, 0].plot(cumulative.index, cumulative.values, label='Strategy', linewidth=2)
        
        if benchmark_returns is not None:
            aligned_dates = cumulative.index.intersection(benchmark_returns.index)
            bench_cum = (1 + benchmark_returns.loc[aligned_dates]).cumprod()
            axes[0, 0].plot(bench_cum.index, bench_cum.values, 
                          label='Benchmark', linewidth=2, alpha=0.7)
        
        axes[0, 0].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        cumulative = (1 + results_df['net_return']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, 
                               alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe ratio (252-day window)
        rolling_returns = results_df['net_return'].rolling(252)
        rolling_sharpe = (rolling_returns.mean() * 252 - 0.02) / (rolling_returns.std() * np.sqrt(252))
        
        axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Rolling Sharpe Ratio (1Y)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Return distribution
        axes[1, 1].hist(results_df['net_return'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=results_df['net_return'].mean(), color='red', 
                          linestyle='--', label='Mean', linewidth=2)
        axes[1, 1].set_title('Return Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Turnover over time
        axes[2, 0].plot(results_df.index, results_df['turnover'], linewidth=1, alpha=0.7)
        axes[2, 0].axhline(y=results_df['turnover'].mean(), color='red', 
                          linestyle='--', label='Mean', linewidth=2)
        axes[2, 0].set_title('Portfolio Turnover', fontsize=12, fontweight='bold')
        axes[2, 0].set_ylabel('Turnover')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Monthly returns heatmap
        monthly_returns = results_df['net_return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot = monthly_returns.to_frame('return')
        monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
        heatmap_data = monthly_returns_pivot.pivot(index='year', columns='month', values='return')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='RdYlGn', 
                   center=0, ax=axes[2, 1], cbar_kws={'label': 'Return'})
        axes[2, 1].set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Month')
        axes[2, 1].set_ylabel('Year')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.show()
    
    def run_backtest(self, results_df: pd.DataFrame, 
                    benchmark_returns: pd.Series = None) -> Dict:
        """
        Run complete backtest analysis.
        
        Args:
            results_df: Portfolio simulation results
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Running backtest analysis...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(results_df, benchmark_returns)
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Plot performance
        self.plot_performance(results_df, benchmark_returns, 
                            save_path='results/performance.png')
        
        return metrics


if __name__ == "__main__":
    import yaml
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy results for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    results_df = pd.DataFrame({
        'date': dates,
        'gross_return': np.random.randn(len(dates)) * 0.01 + 0.0005,
        'net_return': np.random.randn(len(dates)) * 0.01 + 0.0003,
        'turnover': np.random.rand(len(dates)) * 0.5,
        'num_long': np.random.randint(20, 30, len(dates)),
        'num_short': np.random.randint(20, 30, len(dates))
    }).set_index('date')
    
    results_df['portfolio_value'] = 1000000 * (1 + results_df['net_return']).cumprod()
    
    # Run backtest
    backtester = Backtester(config)
    metrics = backtester.run_backtest(results_df)
