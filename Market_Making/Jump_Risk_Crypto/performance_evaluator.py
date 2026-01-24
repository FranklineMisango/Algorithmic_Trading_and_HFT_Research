"""
Performance Evaluator
Calculates metrics and statistical tests for strategy comparison
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Evaluate and compare strategy performance
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_config = config['metrics']
        self.confidence_level = self.metrics_config['statistical_tests']['confidence_level']
        
    def evaluate_strategy(self, results_df: pd.DataFrame, strategy_name: str) -> Dict:
        """
        Calculate all metrics for a strategy
        
        Args:
            results_df: Backtest results DataFrame
            strategy_name: Name of strategy to evaluate
            
        Returns:
            Dictionary of metrics
        """
        strategy_data = results_df[results_df['strategy'] == strategy_name].copy()
        
        returns = strategy_data['returns'].dropna()
        portfolio_values = strategy_data['portfolio_value']
        
        # Return metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Drawdown
        cummax = portfolio_values.cummax()
        drawdown = (portfolio_values / cummax - 1)
        max_drawdown = drawdown.min()
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        metrics = {
            'strategy': strategy_name,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'downside_volatility': downside_vol,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        return metrics
    
    def calculate_jump_metrics(
        self, 
        results_df: pd.DataFrame, 
        df_with_jumps: pd.DataFrame, 
        strategy_name: str,
        weights_history: List[Dict]
    ) -> Dict:
        """
        Calculate jump-specific metrics
        
        Args:
            results_df: Backtest results
            df_with_jumps: DataFrame with jump indicators
            strategy_name: Strategy name
            weights_history: History of portfolio weights
            
        Returns:
            Dictionary of jump metrics
        """
        # Get portfolio jump exposure over time
        strategy_data = results_df[results_df['strategy'] == strategy_name].copy()
        
        # Pivot jump data
        jump_matrix = df_with_jumps.pivot(
            index='date', columns='asset', values='is_jump'
        ).fillna(False)
        
        # Calculate weighted jump exposure
        jump_exposures = []
        
        for weight_record in weights_history:
            date = weight_record['date']
            weights = weight_record['weights']
            
            if date in jump_matrix.index:
                # Get jumps on this date
                day_jumps = jump_matrix.loc[date]
                
                # Calculate weighted exposure
                weight_array = np.array([weights.get(asset, 0) for asset in jump_matrix.columns])
                jump_array = day_jumps.astype(float).values
                
                exposure = (weight_array * jump_array).sum()
                jump_exposures.append(exposure)
        
        # Calculate co-jump frequency
        cojump_matrix = (jump_matrix.sum(axis=1) >= 2)  # At least 2 assets jump
        cojump_frequency = cojump_matrix.sum() / len(cojump_matrix)
        
        metrics = {
            'avg_jump_exposure': np.mean(jump_exposures) if jump_exposures else 0,
            'max_jump_exposure': np.max(jump_exposures) if jump_exposures else 0,
            'cojump_frequency': cojump_frequency
        }
        
        return metrics
    
    def compare_strategies(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare all strategies
        
        Args:
            results_df: Combined backtest results
            
        Returns:
            Comparison DataFrame
        """
        logger.info("Comparing strategy performance...")
        
        all_metrics = []
        
        for strategy in results_df['strategy'].unique():
            metrics = self.evaluate_strategy(results_df, strategy)
            all_metrics.append(metrics)
        
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.set_index('strategy')
        
        # Sort by Sharpe ratio
        comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
        
        # Log results
        logger.info("\n=== Performance Comparison ===")
        for strategy in comparison_df.index:
            sharpe = comparison_df.loc[strategy, 'sharpe_ratio']
            annual_ret = comparison_df.loc[strategy, 'annualized_return'] * 100
            vol = comparison_df.loc[strategy, 'volatility'] * 100
            logger.info(f"{strategy:20s}: Sharpe={sharpe:.3f}, Return={annual_ret:+.1f}%, Vol={vol:.1f}%")
        
        return comparison_df
    
    def statistical_test_sharpe(
        self, 
        returns1: np.ndarray, 
        returns2: np.ndarray
    ) -> Dict:
        """
        Test if Sharpe ratio difference is statistically significant
        Uses Jobson-Korkie test with Memmel correction
        
        Args:
            returns1: Returns for strategy 1
            returns2: Returns for strategy 2
            
        Returns:
            Dictionary with test results
        """
        # Calculate Sharpe ratios
        sharpe1 = returns1.mean() / returns1.std() if returns1.std() > 0 else 0
        sharpe2 = returns2.mean() / returns2.std() if returns2.std() > 0 else 0
        
        # Calculate correlation
        corr = np.corrcoef(returns1, returns2)[0, 1]
        
        # Jobson-Korkie test statistic with Memmel correction
        n = len(returns1)
        
        var1 = returns1.var()
        var2 = returns2.var()
        
        # Variance of Sharpe ratio difference
        theta = (
            (1 / (2 * n)) * (
                (1 + 0.5 * sharpe1**2) + 
                (1 + 0.5 * sharpe2**2) - 
                2 * corr * (1 + 0.5 * sharpe1 * sharpe2)
            )
        )
        
        # Test statistic
        if theta > 0:
            t_stat = (sharpe1 - sharpe2) / np.sqrt(theta)
        else:
            t_stat = 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        # Significance at confidence level
        alpha = 1 - self.confidence_level
        is_significant = p_value < alpha
        
        return {
            'sharpe_diff': sharpe1 - sharpe2,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_level': self.confidence_level
        }
    
    def run_statistical_tests(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run statistical tests comparing strategies
        
        Args:
            results_df: Combined backtest results
            
        Returns:
            DataFrame with test results
        """
        logger.info(f"\nRunning statistical tests (confidence level: {self.confidence_level})...")
        
        strategies = results_df['strategy'].unique()
        
        # Compare jump_adjusted vs all benchmarks
        test_results = []
        
        if 'jump_adjusted' not in strategies:
            logger.warning("jump_adjusted strategy not found")
            return pd.DataFrame()
        
        jump_adjusted_returns = results_df[
            results_df['strategy'] == 'jump_adjusted'
        ]['returns'].dropna().values
        
        for benchmark in strategies:
            if benchmark != 'jump_adjusted':
                benchmark_returns = results_df[
                    results_df['strategy'] == benchmark
                ]['returns'].dropna().values
                
                # Align lengths
                min_len = min(len(jump_adjusted_returns), len(benchmark_returns))
                returns1 = jump_adjusted_returns[:min_len]
                returns2 = benchmark_returns[:min_len]
                
                # Run test
                test_result = self.statistical_test_sharpe(returns1, returns2)
                test_result['comparison'] = f'jump_adjusted vs {benchmark}'
                test_results.append(test_result)
        
        test_df = pd.DataFrame(test_results)
        
        # Log significant results
        significant = test_df[test_df['is_significant']]
        if len(significant) > 0:
            logger.info(f"\nSignificant improvements at {self.confidence_level*100}% confidence:")
            for _, row in significant.iterrows():
                logger.info(f"  {row['comparison']}: Sharpe Δ={row['sharpe_diff']:+.3f} "
                          f"(p={row['p_value']:.4f})")
        else:
            logger.info(f"No significant improvements at {self.confidence_level*100}% confidence level")
        
        return test_df
    
    def generate_performance_report(
        self, 
        results_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive performance report
        
        Args:
            results_df: Backtest results
            test_df: Statistical test results
            
        Returns:
            Dictionary with full report
        """
        # Strategy comparison
        comparison = self.compare_strategies(results_df)
        
        # Identify best strategy
        best_strategy = comparison['sharpe_ratio'].idxmax()
        
        report = {
            'comparison': comparison,
            'statistical_tests': test_df,
            'best_strategy': best_strategy,
            'best_sharpe': comparison.loc[best_strategy, 'sharpe_ratio'],
            'significant_improvements': test_df[test_df['is_significant']].to_dict('records')
        }
        
        logger.info(f"\n=== Performance Report ===")
        logger.info(f"Best Strategy: {best_strategy}")
        logger.info(f"Best Sharpe Ratio: {report['best_sharpe']:.3f}")
        logger.info(f"Significant Improvements: {len(report['significant_improvements'])}")
        
        return report


def evaluate_performance(results_df: pd.DataFrame, config: Dict) -> Dict:
    """
    Convenience function for full performance evaluation
    
    Args:
        results_df: Backtest results
        config: Configuration dictionary
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = PerformanceEvaluator(config)
    
    # Compare strategies
    comparison = evaluator.compare_strategies(results_df)
    
    # Run statistical tests
    test_results = evaluator.run_statistical_tests(results_df)
    
    # Generate report
    report = evaluator.generate_performance_report(results_df, test_results)
    
    return report


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from data_loader import load_and_prepare_data
    from jump_detector import detect_and_analyze_jumps
    from backtester import run_backtest
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data and run backtest
    data_splits = load_and_prepare_data(config)
    test_df = data_splits['test']
    df_with_jumps, metrics, cojump_df = detect_and_analyze_jumps(test_df, config)
    
    backtest_results = run_backtest(df_with_jumps, config)
    combined_results = backtest_results['combined_results']
    
    # Evaluate performance
    performance_report = evaluate_performance(combined_results, config)
    
    print("\n=== Evaluation Complete ===")
    print("\nStrategy Rankings:")
    print(performance_report['comparison'][['sharpe_ratio', 'annualized_return', 'max_drawdown']])
