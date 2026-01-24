"""
Performance Evaluator for Music Royalties Strategy
Calculates comprehensive performance metrics and statistical tests
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Evaluates strategy performance with comprehensive metrics
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """
        Calculate returns from equity curve
        
        Args:
            equity_curve: DataFrame with portfolio_value column
            
        Returns:
            Series of returns
        """
        returns = equity_curve['portfolio_value'].pct_change().dropna()
        return returns
    
    def calculate_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            equity_curve: DataFrame with date and portfolio_value
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating performance metrics...")
        
        returns = self.calculate_returns(equity_curve)
        
        # Basic return metrics
        total_return = (equity_curve['portfolio_value'].iloc[-1] / 
                       equity_curve['portfolio_value'].iloc[0]) - 1
        
        # Annualized metrics
        n_periods = len(equity_curve)
        years = (equity_curve['date'].iloc[-1] - equity_curve['date'].iloc[0]).days / 365.25
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        annualized_vol = returns.std() * np.sqrt(12)  # Assuming monthly returns
        
        # Risk-adjusted returns
        sharpe_ratio = (cagr - self.risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
        sortino_ratio = (cagr - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        dd_series = self._calculate_drawdown(equity_curve['portfolio_value'])
        max_drawdown = dd_series.min()
        
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            # Return metrics
            'total_return': total_return,
            'cagr': cagr,
            'annualized_return': cagr,
            
            # Risk metrics
            'annualized_volatility': annualized_vol,
            'downside_volatility': downside_vol,
            'max_drawdown': max_drawdown,
            
            # Risk-adjusted
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Other
            'win_rate': (returns > 0).mean(),
            'avg_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
            'avg_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0,
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else 0,
            
            # Portfolio
            'n_periods': n_periods,
            'years': years,
        }
        
        return metrics
    
    def _calculate_drawdown(self, portfolio_value: pd.Series) -> pd.Series:
        """
        Calculate drawdown series
        
        Args:
            portfolio_value: Series of portfolio values
            
        Returns:
            Series of drawdowns
        """
        running_max = portfolio_value.cummax()
        drawdown = (portfolio_value - running_max) / running_max
        return drawdown
    
    def calculate_benchmark_comparison(self, equity_curve: pd.DataFrame,
                                      benchmark_returns: pd.Series = None) -> Dict:
        """
        Compare strategy to benchmark (S&P 500)
        
        Args:
            equity_curve: Strategy equity curve
            benchmark_returns: Benchmark returns (if None, uses synthetic)
            
        Returns:
            Dictionary of comparison metrics
        """
        strategy_returns = self.calculate_returns(equity_curve)
        
        # Use synthetic benchmark if not provided
        if benchmark_returns is None:
            # Synthetic S&P 500: 10% annual return, 16% volatility
            n_periods = len(strategy_returns)
            benchmark_returns = pd.Series(
                np.random.normal(0.10/12, 0.16/np.sqrt(12), n_periods),
                index=strategy_returns.index
            )
        
        # Align series
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strat_ret = strategy_returns.loc[common_idx]
        bench_ret = benchmark_returns.loc[common_idx]
        
        # Calculate beta and alpha
        covariance = np.cov(strat_ret, bench_ret)[0, 1]
        benchmark_variance = bench_ret.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        strat_mean = strat_ret.mean() * 12  # Annualized
        bench_mean = bench_ret.mean() * 12
        alpha = strat_mean - (self.risk_free_rate + beta * (bench_mean - self.risk_free_rate))
        
        # Correlation
        correlation = strat_ret.corr(bench_ret)
        
        comparison = {
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'tracking_error': (strat_ret - bench_ret).std() * np.sqrt(12),
            'information_ratio': (strat_mean - bench_mean) / ((strat_ret - bench_ret).std() * np.sqrt(12)) if (strat_ret - bench_ret).std() > 0 else 0
        }
        
        return comparison
    
    def perform_statistical_tests(self, equity_curve: pd.DataFrame,
                                 benchmark_returns: pd.Series = None) -> Dict:
        """
        Perform statistical significance tests
        
        Args:
            equity_curve: Strategy equity curve
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary of test results
        """
        logger.info("Performing statistical tests...")
        
        strategy_returns = self.calculate_returns(equity_curve)
        
        tests = {}
        
        # 1. T-test: Are returns significantly > 0?
        t_stat_zero, p_value_zero = stats.ttest_1samp(strategy_returns, 0)
        tests['ttest_vs_zero'] = {
            't_statistic': t_stat_zero,
            'p_value': p_value_zero,
            'significant': p_value_zero < 0.05,
            'interpretation': 'Returns are significantly different from zero' if p_value_zero < 0.05 else 'Returns not significantly different from zero'
        }
        
        # 2. T-test: Are returns > benchmark?
        if benchmark_returns is not None:
            common_idx = strategy_returns.index.intersection(benchmark_returns.index)
            strat_ret = strategy_returns.loc[common_idx]
            bench_ret = benchmark_returns.loc[common_idx]
            
            t_stat_bench, p_value_bench = stats.ttest_rel(strat_ret, bench_ret)
            tests['ttest_vs_benchmark'] = {
                't_statistic': t_stat_bench,
                'p_value': p_value_bench,
                'significant': p_value_bench < 0.05 and t_stat_bench > 0,
                'interpretation': 'Strategy outperforms benchmark significantly' if (p_value_bench < 0.05 and t_stat_bench > 0) else 'No significant outperformance'
            }
            
            # 3. Correlation significance test
            correlation = strat_ret.corr(bench_ret)
            n = len(strat_ret)
            t_stat_corr = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2) if abs(correlation) < 1 else np.inf
            p_value_corr = 2 * (1 - stats.t.cdf(abs(t_stat_corr), n - 2))
            
            tests['correlation_test'] = {
                'correlation': correlation,
                't_statistic': t_stat_corr,
                'p_value': p_value_corr,
                'significant': p_value_corr < 0.05,
                'interpretation': f'Correlation ({correlation:.3f}) is {"significant" if p_value_corr < 0.05 else "not significant"}'
            }
        
        # 4. Normality test (Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(strategy_returns)
        tests['normality_test'] = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'normal': jb_pvalue > 0.05,
            'interpretation': 'Returns are normally distributed' if jb_pvalue > 0.05 else 'Returns are NOT normally distributed'
        }
        
        return tests
    
    def calculate_transaction_cost_impact(self, trades_df: pd.DataFrame,
                                        initial_capital: float) -> Dict:
        """
        Analyze transaction cost impact
        
        Args:
            trades_df: DataFrame of all trades
            initial_capital: Initial portfolio capital
            
        Returns:
            Dictionary of transaction cost metrics
        """
        if len(trades_df) == 0:
            return {'total_costs': 0, 'cost_ratio': 0}
        
        total_buyer_fees = trades_df['buyer_fee'].sum()
        total_seller_commissions = trades_df['seller_commission'].sum()
        total_slippage = trades_df['slippage'].sum()
        total_costs = trades_df['total_cost'].sum()
        
        cost_analysis = {
            'total_costs': total_costs,
            'buyer_fees': total_buyer_fees,
            'seller_commissions': total_seller_commissions,
            'slippage': total_slippage,
            'cost_ratio': total_costs / initial_capital,
            'cost_per_trade': total_costs / len(trades_df),
            'n_trades': len(trades_df),
            'buys': len(trades_df[trades_df['action'] == 'buy']),
            'sells': len(trades_df[trades_df['action'].isin(['sell', 'sell_partial'])])
        }
        
        return cost_analysis
    
    def generate_performance_report(self, equity_curve: pd.DataFrame,
                                   trades_df: pd.DataFrame,
                                   initial_capital: float) -> str:
        """
        Generate comprehensive text performance report
        
        Args:
            equity_curve: Strategy equity curve
            trades_df: All trades
            initial_capital: Initial capital
            
        Returns:
            Formatted report string
        """
        metrics = self.calculate_metrics(equity_curve)
        comparison = self.calculate_benchmark_comparison(equity_curve)
        tests = self.perform_statistical_tests(equity_curve)
        costs = self.calculate_transaction_cost_impact(trades_df, initial_capital)
        
        report = f"""
{'='*80}
MUSIC ROYALTIES STRATEGY - PERFORMANCE REPORT
{'='*80}

RETURN METRICS
{'-'*80}
Total Return:              {metrics['total_return']*100:>8.2f}%
CAGR:                      {metrics['cagr']*100:>8.2f}%
Annualized Volatility:     {metrics['annualized_volatility']*100:>8.2f}%

RISK-ADJUSTED RETURNS
{'-'*80}
Sharpe Ratio:              {metrics['sharpe_ratio']:>8.3f}
Sortino Ratio:             {metrics['sortino_ratio']:>8.3f}
Calmar Ratio:              {metrics['calmar_ratio']:>8.3f}

DRAWDOWN ANALYSIS
{'-'*80}
Maximum Drawdown:          {metrics['max_drawdown']*100:>8.2f}%
Downside Volatility:       {metrics['downside_volatility']*100:>8.2f}%

TRADING STATISTICS
{'-'*80}
Win Rate:                  {metrics['win_rate']*100:>8.2f}%
Average Win:               {metrics['avg_win']*100:>8.2f}%
Average Loss:              {metrics['avg_loss']*100:>8.2f}%
Profit Factor:             {metrics['profit_factor']:>8.3f}

BENCHMARK COMPARISON (vs S&P 500)
{'-'*80}
Beta:                      {comparison['beta']:>8.3f}
Alpha:                     {comparison['alpha']*100:>8.2f}%
Correlation:               {comparison['correlation']:>8.3f}
Tracking Error:            {comparison['tracking_error']*100:>8.2f}%
Information Ratio:         {comparison['information_ratio']:>8.3f}

TRANSACTION COSTS
{'-'*80}
Total Costs:               ${costs['total_costs']:>12,.0f}
Cost Ratio:                {costs['cost_ratio']*100:>8.2f}%
Number of Trades:          {costs['n_trades']:>8d}
Cost per Trade:            ${costs['cost_per_trade']:>12,.0f}

STATISTICAL TESTS
{'-'*80}
Returns vs Zero:           {tests['ttest_vs_zero']['interpretation']}
  (t={tests['ttest_vs_zero']['t_statistic']:.3f}, p={tests['ttest_vs_zero']['p_value']:.4f})

Normality:                 {tests['normality_test']['interpretation']}
  (JB={tests['normality_test']['statistic']:.3f}, p={tests['normality_test']['p_value']:.4f})

PORTFOLIO SUMMARY
{'-'*80}
Initial Capital:           ${initial_capital:>12,.0f}
Final Value:               ${equity_curve['portfolio_value'].iloc[-1]:>12,.0f}
Number of Periods:         {metrics['n_periods']:>8d}
Years:                     {metrics['years']:>8.2f}

{'='*80}
"""
        
        return report


def evaluate_strategy_performance(equity_curve: pd.DataFrame, trades_df: pd.DataFrame,
                                 config: Dict) -> Dict:
    """
    Convenience function to evaluate full strategy performance
    
    Args:
        equity_curve: Strategy equity curve
        trades_df: All trades
        config: Configuration dictionary
        
    Returns:
        Dictionary of all performance metrics
    """
    evaluator = PerformanceEvaluator(config)
    
    metrics = evaluator.calculate_metrics(equity_curve)
    comparison = evaluator.calculate_benchmark_comparison(equity_curve)
    tests = evaluator.perform_statistical_tests(equity_curve)
    costs = evaluator.calculate_transaction_cost_impact(
        trades_df, 
        config['backtest']['initial_capital']
    )
    
    # Generate report
    report = evaluator.generate_performance_report(
        equity_curve, 
        trades_df, 
        config['backtest']['initial_capital']
    )
    
    print(report)
    
    return {
        'metrics': metrics,
        'benchmark_comparison': comparison,
        'statistical_tests': tests,
        'transaction_costs': costs,
        'report': report
    }


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from data_loader import load_and_prepare_data
    from feature_engineering import engineer_all_features
    from model_trainer import train_and_validate_model
    from portfolio_constructor import PortfolioConstructor
    from backtester import RoyaltyBacktester, prepare_universe_by_date
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and prepare data
    data_splits = load_and_prepare_data(config)
    
    # Engineer features
    train_df = engineer_all_features(data_splits['train'], config)
    val_df = engineer_all_features(data_splits['validation'], config)
    test_df = engineer_all_features(data_splits['test'], config)
    
    # Train model
    model, _ = train_and_validate_model(train_df, val_df, config)
    
    # Prepare universe and run backtest
    universe_by_date = prepare_universe_by_date(test_df, config['portfolio']['rebalancing_frequency'])
    backtester = RoyaltyBacktester(config)
    constructor = PortfolioConstructor(config)
    equity_curve = backtester.run_backtest(universe_by_date, model, constructor)
    
    # Evaluate performance
    trades_df = backtester.get_trades_df()
    performance = evaluate_strategy_performance(equity_curve, trades_df, config)
