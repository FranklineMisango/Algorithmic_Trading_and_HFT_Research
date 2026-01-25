"""
FX Carry Strategy - Backtesting Module

Performance evaluation with:
1. Transaction costs (bid-ask spreads)
2. Risk metrics (Sharpe, max drawdown, VaR)
3. Stress period analysis
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class FXCarryBacktester:
    """Backtest FX carry strategy with realistic costs"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.bid_ask_bps = self.config['backtesting']['bid_ask_spread_bps']
        self.stress_periods = self.config['backtesting']['stress_periods']
        
    def apply_transaction_costs(self, returns: pd.Series, weights: pd.DataFrame) -> pd.Series:
        """
        Deduct transaction costs when positions change
        
        Args:
            returns: Portfolio returns
            weights: Portfolio weights
            
        Returns:
            Returns after transaction costs
        """
        print("\n" + "="*60)
        print("BACKTESTING")
        print("="*60)
        print(f"Bid-ask spread: {self.bid_ask_bps} bps")
        
        # Calculate turnover (sum of absolute weight changes)
        weight_changes = weights.diff().abs()
        turnover = weight_changes.sum(axis=1)
        
        # Transaction cost = turnover * bid-ask spread
        tc_bps = turnover * self.bid_ask_bps
        tc_returns = -tc_bps / 10000  # Convert bps to decimal
        
        # Apply costs
        net_returns = returns + tc_returns
        
        print(f"Average daily turnover: {turnover.mean():.2%}")
        print(f"Average daily TC: {tc_returns.mean() * 10000:.2f} bps")
        print(f"\nGross Sharpe: {returns.mean() / returns.std() * np.sqrt(252):.2f}")
        print(f"Net Sharpe: {net_returns.mean() / net_returns.std() * np.sqrt(252):.2f}")
        
        return net_returns
    
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        cumulative = (1 + returns).cumprod()
        
        metrics = {
            # Returns
            'total_return': cumulative.iloc[-1] - 1,
            'ann_return': returns.mean() * 252,
            'ann_volatility': returns.std() * np.sqrt(252),
            
            # Risk-adjusted
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'sortino_ratio': returns.mean() / returns[returns < 0].std() * np.sqrt(252),
            
            # Drawdown
            'max_drawdown': (cumulative / cumulative.cummax() - 1).min(),
            'avg_drawdown': (cumulative / cumulative.cummax() - 1).mean(),
            
            # Distribution
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
            
            # Win rate
            'win_rate': (returns > 0).sum() / len(returns),
            'avg_win': returns[returns > 0].mean(),
            'avg_loss': returns[returns < 0].mean(),
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in formatted table"""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\nReturns:")
        print(f"  Total Return:        {metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {metrics['ann_return']:>10.2%}")
        print(f"  Annualized Volatility: {metrics['ann_volatility']:>8.2%}")
        
        print(f"\nRisk-Adjusted:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        
        print(f"\nDrawdown:")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        print(f"  Average Drawdown:    {metrics['avg_drawdown']:>10.2%}")
        
        print(f"\nDistribution:")
        print(f"  Skewness:            {metrics['skewness']:>10.2f}")
        print(f"  Kurtosis:            {metrics['kurtosis']:>10.2f}")
        print(f"  VaR (95%):           {metrics['var_95']:>10.2%}")
        print(f"  CVaR (95%):          {metrics['cvar_95']:>10.2%}")
        
        print(f"\nWin Rate:")
        print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"  Avg Win:             {metrics['avg_win']:>10.4%}")
        print(f"  Avg Loss:            {metrics['avg_loss']:>10.4%}")
    
    def stress_test(self, returns: pd.Series) -> pd.DataFrame:
        """
        Analyze performance during stress periods
        
        Returns:
            DataFrame with metrics for each stress period
        """
        print("\n" + "="*60)
        print("STRESS TESTING")
        print("="*60)
        
        stress_results = []
        
        for start, end in self.stress_periods:
            period_returns = returns.loc[start:end]
            
            if len(period_returns) > 0:
                cumulative = (1 + period_returns).cumprod().iloc[-1] - 1
                vol = period_returns.std() * np.sqrt(252)
                sharpe = period_returns.mean() / period_returns.std() * np.sqrt(252)
                max_dd = (period_returns.cumsum() - period_returns.cumsum().cummax()).min()
                
                stress_results.append({
                    'period': f"{start} to {end}",
                    'return': cumulative,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'max_drawdown': max_dd,
                    'days': len(period_returns)
                })
                
                print(f"\n{start} to {end}:")
                print(f"  Return: {cumulative:>10.2%}")
                print(f"  Sharpe: {sharpe:>10.2f}")
                print(f"  Max DD: {max_dd:>10.2%}")
        
        return pd.DataFrame(stress_results)
    
    def plot_results(self, returns: pd.Series, weights: pd.DataFrame):
        """Generate performance visualizations"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('FX Carry Strategy - Backtest Results', fontsize=16)
        
        # Cumulative returns
        cumulative = (1 + returns).cumprod()
        axes[0, 0].plot(cumulative.index, cumulative.values)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        drawdown = cumulative / cumulative.cummax() - 1
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling Sharpe (6-month)
        rolling_sharpe = returns.rolling(126).mean() / returns.rolling(126).std() * np.sqrt(252)
        axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axes[1, 0].set_title('Rolling 6-Month Sharpe Ratio')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        monthly_returns = returns.resample('M').sum()
        monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, 
                                                  monthly_returns.index.month]).sum().unstack()
        sns.heatmap(monthly_pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0, 
                   ax=axes[1, 1], cbar_kws={'label': 'Monthly Return'})
        axes[1, 1].set_title('Monthly Returns Heatmap')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Year')
        
        # Return distribution
        axes[2, 0].hist(returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[2, 0].axvline(returns.mean(), color='red', linestyle='--', 
                          label=f'Mean: {returns.mean()*10000:.1f} bps')
        axes[2, 0].set_title('Daily Returns Distribution')
        axes[2, 0].set_xlabel('Daily Return')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Position count over time
        position_count = (weights != 0).sum(axis=1)
        axes[2, 1].plot(position_count.index, position_count.values, alpha=0.7)
        axes[2, 1].set_title('Number of Active Positions')
        axes[2, 1].set_ylabel('# Positions')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Results saved to backtest_results.png")
        
        return fig
    
    def run_backtest(self, returns: pd.Series, weights: pd.DataFrame) -> Dict:
        """
        Complete backtesting pipeline
        
        Returns:
            Dictionary of results
        """
        # Apply transaction costs
        net_returns = self.apply_transaction_costs(returns, weights)
        
        # Calculate metrics
        metrics = self.calculate_metrics(net_returns)
        self.print_metrics(metrics)
        
        # Stress testing
        stress_results = self.stress_test(net_returns)
        
        # Plot results
        self.plot_results(net_returns, weights)
        
        print("\n✓ Backtesting complete")
        
        return {
            'metrics': metrics,
            'stress_results': stress_results,
            'returns': net_returns
        }


if __name__ == "__main__":
    # Example usage
    from data_acquisition import FXDataAcquisition
    from signal_generator import CarrySignalGenerator
    from factor_models import FXFactorModel
    from portfolio_constructor import PortfolioConstructor
    
    # Full pipeline
    fx_data = FXDataAcquisition()
    spots, rates, carry, factors = fx_data.load_data()
    
    signal_gen = CarrySignalGenerator()
    zscores, signals, returns = signal_gen.run_signal_generation(carry, spots)
    
    factor_model = FXFactorModel()
    factor_returns = factor_model.calculate_factor_returns(factors)
    neutral_returns = factor_model.neutralize_returns(returns, factor_returns)
    
    portfolio = PortfolioConstructor()
    weights, pf_returns = portfolio.construct_portfolio(neutral_returns, signals)
    
    backtester = FXCarryBacktester()
    results = backtester.run_backtest(pf_returns, weights)
