"""
Performance Evaluation Module

Calculates comprehensive performance metrics for strategy evaluation.

Key Metrics:
- Sharpe Ratio (primary optimization target)
- Maximum Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio
- Sortino Ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PerformanceEvaluator:
    """
    Evaluates strategy performance with comprehensive metrics.
    """
    
    def __init__(self, config: dict):
        """
        Initialize performance evaluator.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.risk_free_rate = 0.0  # Assume 0 for simplicity (can use T-bill rate)
    
    def calculate_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """
        Calculate returns from equity curve.
        
        Parameters
        ----------
        equity_curve : pd.DataFrame
            Equity curve with portfolio_value column
            
        Returns
        -------
        pd.Series
            Returns series
        """
        return equity_curve['portfolio_value'].pct_change().fillna(0)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Sharpe = (Mean Return - Risk Free Rate) / Std(Returns) * sqrt(periods)
        
        Parameters
        ----------
        returns : pd.Series
            Returns series
        periods_per_year : int
            Number of periods per year for annualization
            
        Returns
        -------
        float
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / periods_per_year
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
        return sharpe
    
    def calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sortino ratio (uses downside deviation).
        
        Parameters
        ----------
        returns : pd.Series
            Returns series
        periods_per_year : int
            Number of periods per year
            
        Returns
        -------
        float
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
        return sortino
    
    def calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown.
        
        Parameters
        ----------
        equity_curve : pd.DataFrame
            Equity curve with portfolio_value column
            
        Returns
        -------
        tuple
            (max_drawdown, peak_date, trough_date)
        """
        portfolio_value = equity_curve['portfolio_value']
        
        # Calculate running maximum
        running_max = portfolio_value.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_value - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        
        # Find dates
        max_dd_idx = drawdown.idxmin()
        peak_idx = portfolio_value.loc[:max_dd_idx].idxmax()
        
        return max_dd, peak_idx, max_dd_idx
    
    def calculate_calmar_ratio(
        self, 
        returns: pd.Series, 
        equity_curve: pd.DataFrame,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (Annual Return / Max Drawdown).
        
        Parameters
        ----------
        returns : pd.Series
            Returns series
        equity_curve : pd.DataFrame
            Equity curve
        periods_per_year : int
            Periods per year
            
        Returns
        -------
        float
            Calmar ratio
        """
        annual_return = returns.mean() * periods_per_year
        max_dd, _, _ = self.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """
        Calculate win rate.
        
        Parameters
        ----------
        trades : pd.DataFrame
            Trades dataframe
            
        Returns
        -------
        float
            Win rate (0-1)
        """
        if len(trades) == 0:
            return 0.0
        
        winning_trades = trades[trades['pnl_net'] > 0]
        return len(winning_trades) / len(trades)
    
    def calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """
        Calculate profit factor (Gross Profit / Gross Loss).
        
        Parameters
        ----------
        trades : pd.DataFrame
            Trades dataframe
            
        Returns
        -------
        float
            Profit factor
        """
        if len(trades) == 0:
            return 0.0
        
        winning_trades = trades[trades['pnl_net'] > 0]
        losing_trades = trades[trades['pnl_net'] <= 0]
        
        gross_profit = winning_trades['pnl_net'].sum()
        gross_loss = abs(losing_trades['pnl_net'].sum())
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_average_win_loss(self, trades: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate average win and average loss.
        
        Parameters
        ----------
        trades : pd.DataFrame
            Trades dataframe
            
        Returns
        -------
        tuple
            (avg_win, avg_loss)
        """
        if len(trades) == 0:
            return 0.0, 0.0
        
        winning_trades = trades[trades['pnl_net'] > 0]
        losing_trades = trades[trades['pnl_net'] <= 0]
        
        avg_win = winning_trades['pnl_net'].mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades['pnl_net'].mean() if len(losing_trades) > 0 else 0.0
        
        return avg_win, avg_loss
    
    def calculate_expectancy(self, trades: pd.DataFrame) -> float:
        """
        Calculate trade expectancy.
        
        Expectancy = (Win Rate * Avg Win) + (Loss Rate * Avg Loss)
        
        Parameters
        ----------
        trades : pd.DataFrame
            Trades dataframe
            
        Returns
        -------
        float
            Expectancy
        """
        if len(trades) == 0:
            return 0.0
        
        win_rate = self.calculate_win_rate(trades)
        avg_win, avg_loss = self.calculate_average_win_loss(trades)
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        return expectancy
    
    def evaluate_strategy(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame
    ) -> Dict:
        """
        Calculate all performance metrics.
        
        Parameters
        ----------
        equity_curve : pd.DataFrame
            Equity curve
        trades : pd.DataFrame
            Trades dataframe
            
        Returns
        -------
        dict
            Performance metrics
        """
        print("="*60)
        print("PERFORMANCE EVALUATION")
        print("="*60)
        
        # Calculate returns
        returns = self.calculate_returns(equity_curve)
        
        # Calculate metrics
        metrics = {}
        
        # Returns
        total_return = (equity_curve['portfolio_value'].iloc[-1] / 
                       equity_curve['portfolio_value'].iloc[0] - 1)
        metrics['total_return'] = total_return
        metrics['annualized_return'] = ((1 + total_return) ** (252 / len(equity_curve)) - 1)
        
        # Risk metrics
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        
        max_dd, peak_date, trough_date = self.calculate_max_drawdown(equity_curve)
        metrics['max_drawdown'] = max_dd
        metrics['max_drawdown_peak'] = peak_date
        metrics['max_drawdown_trough'] = trough_date
        
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns, equity_curve)
        
        # Volatility
        metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        
        # Trade metrics
        if len(trades) > 0:
            metrics['total_trades'] = len(trades)
            metrics['win_rate'] = self.calculate_win_rate(trades)
            metrics['profit_factor'] = self.calculate_profit_factor(trades)
            
            avg_win, avg_loss = self.calculate_average_win_loss(trades)
            metrics['average_win'] = avg_win
            metrics['average_loss'] = avg_loss
            metrics['expectancy'] = self.calculate_expectancy(trades)
            
            # Holding periods
            metrics['avg_holding_bars'] = trades['holding_bars'].mean()
            metrics['max_holding_bars'] = trades['holding_bars'].max()
            
            # Costs
            metrics['total_commission'] = trades['commission'].sum()
            metrics['total_slippage'] = (trades['entry_slippage'] + trades['exit_slippage']).sum()
            metrics['total_costs'] = metrics['total_commission'] + metrics['total_slippage']
            metrics['costs_pct_of_capital'] = (metrics['total_costs'] / 
                                               equity_curve['portfolio_value'].iloc[0] * 100)
        
        # Print metrics
        print("\nReturns:")
        print(f"  Total Return: {metrics['total_return']*100:.2f}%")
        print(f"  Annualized Return: {metrics['annualized_return']*100:.2f}%")
        print(f"  Annualized Volatility: {metrics['annualized_volatility']*100:.2f}%")
        
        print("\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        
        print("\nDrawdown:")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Peak: {metrics['max_drawdown_peak']}")
        print(f"  Trough: {metrics['max_drawdown_trough']}")
        
        if len(trades) > 0:
            print("\nTrade Statistics:")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Average Win: ${metrics['average_win']:,.0f}")
            print(f"  Average Loss: ${metrics['average_loss']:,.0f}")
            print(f"  Expectancy: ${metrics['expectancy']:,.0f}")
            print(f"  Avg Holding: {metrics['avg_holding_bars']:.1f} bars")
            
            print("\nTransaction Costs:")
            print(f"  Total Commission: ${metrics['total_commission']:,.0f}")
            print(f"  Total Slippage: ${metrics['total_slippage']:,.0f}")
            print(f"  Total Costs: ${metrics['total_costs']:,.0f}")
            print(f"  Costs as % of Capital: {metrics['costs_pct_of_capital']:.2f}%")
        
        print("="*60)
        
        return metrics


def visualize_performance(equity_curve: pd.DataFrame, trades: pd.DataFrame, metrics: Dict):
    """
    Visualize strategy performance.
    
    Parameters
    ----------
    equity_curve : pd.DataFrame
        Equity curve
    trades : pd.DataFrame
        Trades
    metrics : dict
        Performance metrics
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Equity curve
    ax = axes[0, 0]
    ax.plot(equity_curve.index, equity_curve['portfolio_value'], linewidth=2)
    ax.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=equity_curve['portfolio_value'].iloc[0], color='red', 
               linestyle='--', alpha=0.5, label='Initial Capital')
    ax.legend()
    
    # Drawdown
    ax = axes[0, 1]
    portfolio_value = equity_curve['portfolio_value']
    running_max = portfolio_value.expanding().max()
    drawdown = (portfolio_value - running_max) / running_max * 100
    ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    ax.plot(drawdown.index, drawdown, color='red', linewidth=1.5)
    ax.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    
    # Returns distribution
    ax = axes[1, 0]
    returns = equity_curve['portfolio_value'].pct_change().dropna()
    ax.hist(returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=returns.mean() * 100, color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {returns.mean()*100:.3f}%')
    ax.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative returns
    ax = axes[1, 1]
    cumulative_returns = (1 + returns).cumprod()
    ax.plot(cumulative_returns.index, cumulative_returns, linewidth=2)
    ax.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.grid(True, alpha=0.3)
    
    # Trade P&L
    if len(trades) > 0:
        ax = axes[2, 0]
        ax.scatter(range(len(trades)), trades['pnl_net'], 
                   c=['green' if x > 0 else 'red' for x in trades['pnl_net']],
                   alpha=0.6, s=50)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_title('Trade P&L', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('P&L ($)')
        ax.grid(True, alpha=0.3)
        
        # Monthly returns
        ax = axes[2, 1]
        equity_curve['month'] = equity_curve.index.to_period('M')
        monthly_returns = equity_curve.groupby('month')['portfolio_value'].apply(
            lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100
        )
        monthly_returns.plot(kind='bar', ax=ax, 
                           color=['green' if x > 0 else 'red' for x in monthly_returns])
        ax.set_title('Monthly Returns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/performance_visualization.png', dpi=150, bbox_inches='tight')
    print("\nPerformance visualization saved to results/performance_visualization.png")
    plt.show()


def main():
    """
    Test performance evaluator.
    """
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load backtest results
    equity_curve = pd.read_csv('results/backtest_equity_curve.csv', index_col=0, parse_dates=True)
    trades = pd.read_csv('results/backtest_trades.csv', index_col=0)
    
    # Evaluate performance
    evaluator = PerformanceEvaluator(config)
    metrics = evaluator.evaluate_strategy(equity_curve, trades)
    
    # Visualize
    visualize_performance(equity_curve, trades, metrics)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['Value']
    metrics_df.to_csv('results/performance_metrics.csv')
    print("\nPerformance metrics saved to results/performance_metrics.csv")


if __name__ == "__main__":
    main()
