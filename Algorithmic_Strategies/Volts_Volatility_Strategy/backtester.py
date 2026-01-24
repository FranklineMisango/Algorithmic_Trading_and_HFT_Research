"""
Backtesting Framework for Volts Strategy

Implements a comprehensive backtesting system to evaluate the performance
of volatility-based Granger causality trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    shares: float = 0.0
    direction: int = 1  # 1 for long, -1 for short
    pair_name: str = ""
    commission: float = 0.0
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    
    def close_trade(self, exit_date: pd.Timestamp, exit_price: float) -> None:
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
        # Calculate P&L
        price_change = (exit_price - self.entry_price) * self.direction
        self.pnl = (price_change * self.shares) - (2 * self.commission)  # Entry + exit commission
        self.return_pct = (price_change / self.entry_price) * 100
    
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_date is None


@dataclass
class PortfolioState:
    """Represents the state of the portfolio."""
    cash: float
    positions: Dict[str, Trade] = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)
    dates: List[pd.Timestamp] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    
    def total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            trade.shares * current_prices.get(trade.pair_name.split('->')[1], trade.entry_price) * trade.direction
            for trade in self.positions.values()
        )
        return self.cash + positions_value


class VoltBacktester:
    """
    Backtesting engine for the Volts strategy.
    """
    
    def __init__(
        self,
        initial_capital_per_pair: float = 1000.0,
        commission: float = 9.0,
        slippage: float = 0.0005,
        position_size_pct: float = 1.0
    ):
        """
        Initialize backtester.
        
        Parameters:
        -----------
        initial_capital_per_pair : float
            Initial capital allocated per trading pair
        commission : float
            Commission per trade (in dollars)
        slippage : float
            Slippage as fraction of price
        position_size_pct : float
            Percentage of allocated capital to use (0-1)
        """
        self.initial_capital_per_pair = initial_capital_per_pair
        self.commission = commission
        self.slippage = slippage
        self.position_size_pct = position_size_pct
        
        self.portfolio = None
        self.results = {}
        
    def calculate_position_size(
        self,
        capital: float,
        price: float
    ) -> int:
        """
        Calculate number of shares to trade.
        
        Parameters:
        -----------
        capital : float
            Available capital
        price : float
            Current price
            
        Returns:
        --------
        int : Number of shares
        """
        # Prevent trading with negative or zero capital
        if capital <= 0:
            return 0
        
        # Limit to 95% of capital to prevent over-leverage
        usable_capital = capital * min(self.position_size_pct, 0.95)
        # Account for commission
        max_shares = (usable_capital - self.commission) / (price * (1 + self.slippage))
        return max(int(max_shares), 0)
    
    def execute_trade(
        self,
        date: pd.Timestamp,
        signal: int,
        pair_name: str,
        price: float,
        capital: float
    ) -> Optional[Trade]:
        """
        Execute a trade based on signal.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Trade date
        signal : int
            Signal value (1 for buy, -1 for sell)
        pair_name : str
            Trading pair identifier
        price : float
            Current price
        capital : float
            Available capital
            
        Returns:
        --------
        Optional[Trade] : Executed trade or None
        """
        if signal == 0:
            return None
        
        # Calculate position size
        shares = self.calculate_position_size(capital, price)
        
        if shares <= 0:
            return None
        
        # Apply slippage
        execution_price = price * (1 + self.slippage * signal)
        
        # Create trade
        trade = Trade(
            entry_date=date,
            entry_price=execution_price,
            shares=shares,
            direction=signal,
            pair_name=pair_name,
            commission=self.commission
        )
        
        return trade
    
    def run_backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        signals: Dict[str, pd.DataFrame],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Run backtest for all trading pairs.
        
        Parameters:
        -----------
        price_data : Dict[str, pd.DataFrame]
            Price data for all assets {ticker: DataFrame with OHLC}
        signals : Dict[str, pd.DataFrame]
            Trading signals {pair_name: signals_df}
        start_date : pd.Timestamp, optional
            Backtest start date
        end_date : pd.Timestamp, optional
            Backtest end date
            
        Returns:
        --------
        Dict : Backtest results
        """
        # Initialize portfolio
        n_pairs = len(signals)
        total_capital = self.initial_capital_per_pair * n_pairs
        
        # Create capital allocation for each pair
        capital_allocation = {
            pair: self.initial_capital_per_pair 
            for pair in signals.keys()
        }
        
        # Results tracking
        pair_results = {}
        
        # Run backtest for each pair independently
        for pair_name, signals_df in signals.items():
            target_ticker = pair_name.split('->')[1]
            
            # Filter date range
            if start_date:
                signals_df = signals_df[signals_df.index >= start_date]
            if end_date:
                signals_df = signals_df[signals_df.index <= end_date]
            
            # Get price data for target
            target_prices = price_data[target_ticker]['Close']
            
            # Run backtest for this pair
            result = self._backtest_single_pair(
                pair_name,
                signals_df,
                target_prices,
                capital_allocation[pair_name]
            )
            
            pair_results[pair_name] = result
        
        # Aggregate results
        aggregated_results = self._aggregate_results(pair_results, total_capital)
        
        self.results = {
            'pair_results': pair_results,
            'aggregated': aggregated_results
        }
        
        return self.results
    
    def _backtest_single_pair(
        self,
        pair_name: str,
        signals_df: pd.DataFrame,
        target_prices: pd.Series,
        initial_capital: float
    ) -> Dict:
        """
        Backtest a single trading pair.
        
        Parameters:
        -----------
        pair_name : str
            Trading pair identifier
        signals_df : pd.DataFrame
            Signals for this pair
        target_prices : pd.Series
            Price data for target stock
        initial_capital : float
            Initial capital for this pair
            
        Returns:
        --------
        Dict : Results for this pair
        """
        portfolio = PortfolioState(cash=initial_capital)
        current_trade = None
        
        # Align signals and prices
        common_dates = signals_df.index.intersection(target_prices.index)
        
        for date in common_dates:
            signal = signals_df.loc[date, 'signal']
            price = target_prices.loc[date]
            
            # Update equity curve
            current_value = portfolio.cash
            if current_trade is not None and current_trade.is_open():
                # Calculate unrealized P&L for open position
                price_change = (price - current_trade.entry_price) * current_trade.direction
                unrealized_pnl = price_change * current_trade.shares
                # Add locked capital + unrealized P&L
                current_value += (current_trade.shares * current_trade.entry_price) + unrealized_pnl
            
            portfolio.equity_curve.append(current_value)
            portfolio.dates.append(date)
            
            # Check if we need to close existing trade
            if current_trade is not None and current_trade.is_open():
                # Close if signal changes or is opposite
                if signal != current_trade.direction:
                    # Close trade
                    execution_price = price * (1 - self.slippage * current_trade.direction)
                    current_trade.close_trade(date, execution_price)
                    
                    # Return initial investment + P&L
                    portfolio.cash += (current_trade.shares * current_trade.entry_price)  # Return locked capital
                    portfolio.cash += current_trade.pnl  # Add/subtract P&L (includes commissions)
                    
                    portfolio.closed_trades.append(current_trade)
                    current_trade = None
            
            # Open new trade if signal is not hold and no open position
            if current_trade is None and signal != 0:
                new_trade = self.execute_trade(
                    date,
                    signal,
                    pair_name,
                    price,
                    portfolio.cash
                )
                
                if new_trade is not None:
                    # Deduct cost for both long and short (short requires margin)
                    cost = new_trade.shares * new_trade.entry_price + self.commission
                    portfolio.cash -= cost
                    current_trade = new_trade
        
        # Close any remaining open trade
        if current_trade is not None and current_trade.is_open():
            last_date = common_dates[-1]
            last_price = target_prices.loc[last_date]
            execution_price = last_price * (1 - self.slippage * current_trade.direction)
            current_trade.close_trade(last_date, execution_price)
            
            # Return initial investment + P&L
            portfolio.cash += (current_trade.shares * current_trade.entry_price)
            portfolio.cash += current_trade.pnl
            
            portfolio.closed_trades.append(current_trade)
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio, initial_capital)
        
        return {
            'portfolio': portfolio,
            'metrics': metrics,
            'trades': portfolio.closed_trades
        }
    
    def _calculate_metrics(
        self,
        portfolio: PortfolioState,
        initial_capital: float
    ) -> Dict:
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        portfolio : PortfolioState
            Portfolio state
        initial_capital : float
            Initial capital
            
        Returns:
        --------
        Dict : Performance metrics
        """
        trades = portfolio.closed_trades
        equity_curve = np.array(portfolio.equity_curve)
        
        if len(trades) == 0:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'n_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0
            }
        
        # Basic metrics
        final_value = equity_curve[-1]
        total_return = final_value - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        n_trades = len(trades)
        n_wins = len(winning_trades)
        win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else np.inf
        
        # Drawdown
        cumulative_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Returns
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Calmar ratio
        if max_drawdown > 0:
            # Annualize return
            n_years = len(equity_curve) / 252
            annualized_return = (final_value / initial_capital) ** (1 / n_years) - 1
            calmar_ratio = (annualized_return * 100) / max_drawdown
        else:
            calmar_ratio = 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_value': final_value,
            'n_trades': n_trades,
            'n_wins': n_wins,
            'n_losses': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    def _aggregate_results(
        self,
        pair_results: Dict,
        total_capital: float
    ) -> Dict:
        """
        Aggregate results across all pairs.
        
        Parameters:
        -----------
        pair_results : Dict
            Results for each pair
        total_capital : float
            Total initial capital
            
        Returns:
        --------
        Dict : Aggregated metrics
        """
        # Combine equity curves
        all_dates = set()
        for result in pair_results.values():
            all_dates.update(result['portfolio'].dates)
        
        all_dates = sorted(all_dates)
        
        # Build combined equity curve
        combined_equity = []
        for date in all_dates:
            total_value = 0
            for result in pair_results.values():
                portfolio = result['portfolio']
                if date in portfolio.dates:
                    idx = portfolio.dates.index(date)
                    total_value += portfolio.equity_curve[idx]
                else:
                    # Use last known value
                    dates_before = [d for d in portfolio.dates if d <= date]
                    if dates_before:
                        idx = portfolio.dates.index(max(dates_before))
                        total_value += portfolio.equity_curve[idx]
                    else:
                        total_value += self.initial_capital_per_pair
            
            combined_equity.append(total_value)
        
        # Aggregate trades
        all_trades = []
        for result in pair_results.values():
            all_trades.extend(result['trades'])
        
        # Create temporary portfolio for metric calculation
        temp_portfolio = PortfolioState(cash=0)
        temp_portfolio.equity_curve = combined_equity
        temp_portfolio.dates = all_dates
        temp_portfolio.closed_trades = all_trades
        
        # Calculate aggregated metrics
        aggregated_metrics = self._calculate_metrics(temp_portfolio, total_capital)
        
        return {
            'metrics': aggregated_metrics,
            'equity_curve': combined_equity,
            'dates': all_dates,
            'all_trades': all_trades
        }
    
    def print_results(self) -> None:
        """Print backtest results in a formatted way."""
        if not self.results:
            print("No results available. Run backtest first.")
            return
        
        print("\n" + "="*80)
        print("VOLTS STRATEGY BACKTEST RESULTS")
        print("="*80)
        
        # Aggregated results
        agg_metrics = self.results['aggregated']['metrics']
        print("\nAGGREGATED PERFORMANCE:")
        print(f"Total Return: ${agg_metrics['total_return']:.2f} ({agg_metrics['total_return_pct']:.2f}%)")
        print(f"Final Value: ${agg_metrics['final_value']:.2f}")
        print(f"Number of Trades: {agg_metrics['n_trades']}")
        print(f"Win Rate: {agg_metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {agg_metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {agg_metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {agg_metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {agg_metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {agg_metrics['calmar_ratio']:.2f}")
        
        # Per-pair results
        print("\n" + "-"*80)
        print("PER-PAIR PERFORMANCE:")
        print("-"*80)
        
        for pair_name, result in self.results['pair_results'].items():
            metrics = result['metrics']
            print(f"\n{pair_name}:")
            print(f"  Return: ${metrics['total_return']:.2f} ({metrics['total_return_pct']:.2f}%)")
            print(f"  Trades: {metrics['n_trades']} | Wins: {metrics['n_wins']} | "
                  f"Losses: {metrics['n_losses']} | Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  Max DD: {metrics['max_drawdown']:.2f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
                  f"Sortino: {metrics['sortino_ratio']:.2f}")
    
    def plot_results(self, save_path: str = None) -> None:
        """
        Plot backtest results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save plot
        """
        if not self.results:
            print("No results available. Run backtest first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Equity curves
        agg_data = self.results['aggregated']
        axes[0, 0].plot(agg_data['dates'], agg_data['equity_curve'], 
                       label='Combined', linewidth=2, color='black')
        
        for pair_name, result in self.results['pair_results'].items():
            portfolio = result['portfolio']
            axes[0, 0].plot(portfolio.dates, portfolio.equity_curve, 
                           label=pair_name, alpha=0.7)
        
        axes[0, 0].set_title('Equity Curves', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        equity = np.array(agg_data['equity_curve'])
        cumulative_max = np.maximum.accumulate(equity)
        drawdown = (equity - cumulative_max) / cumulative_max * 100
        
        axes[0, 1].fill_between(agg_data['dates'], drawdown, 0, 
                                alpha=0.3, color='red')
        axes[0, 1].plot(agg_data['dates'], drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Trade P&L distribution
        all_pnls = [trade.pnl for trade in agg_data['all_trades']]
        axes[1, 0].hist(all_pnls, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('P&L ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Cumulative returns comparison
        returns_data = []
        for pair_name, result in self.results['pair_results'].items():
            metrics = result['metrics']
            returns_data.append({
                'pair': pair_name,
                'return_pct': metrics['total_return_pct']
            })
        
        returns_df = pd.DataFrame(returns_data)
        axes[1, 1].bar(range(len(returns_df)), returns_df['return_pct'], 
                      color=['green' if x > 0 else 'red' for x in returns_df['return_pct']],
                      alpha=0.7, edgecolor='black')
        axes[1, 1].set_xticks(range(len(returns_df)))
        axes[1, 1].set_xticklabels(returns_df['pair'], rotation=45, ha='right')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 1].set_title('Returns by Pair', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Return (%)')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    print("Backtester module loaded. Use with main strategy pipeline.")
