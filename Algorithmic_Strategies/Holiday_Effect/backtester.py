"""
Backtester for Holiday Effect Equity Strategy

Event-driven backtest with realistic transaction costs and risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import yaml


class Backtester:
    """Backtest equity long strategy for Holiday Effect."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.portfolio_config = self.config['portfolio']
        self.execution_config = self.config['execution']
        self.risk_config = self.config['risk_management']
        
        self.slippage_bps = self.execution_config['slippage_bps']
        self.commission_bps = self.execution_config['commission_bps']
        
    def calculate_transaction_costs(self, trade_value: float) -> float:
        """
        Calculate realistic transaction costs.
        
        Args:
            trade_value: Dollar value of trade
            
        Returns:
            Total cost in dollars
        """
        slippage = trade_value * (self.slippage_bps / 10000)
        commission = trade_value * (self.commission_bps / 10000)
        
        return slippage + commission
    
    def run_backtest(self,
                     prices: pd.DataFrame,
                     signals: pd.DataFrame,
                     initial_capital: float = None) -> Dict:
        """
        Run backtest for equity long strategy.
        
        Args:
            prices: AMZN price data
            signals: Trading signals
            initial_capital: Starting capital
            
        Returns:
            Dictionary with backtest results
        """
        if initial_capital is None:
            initial_capital = self.portfolio_config['initial_capital']
        
        # Initialize tracking
        cash = initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        positions = []
        
        # Align prices and signals
        common_dates = prices.index.intersection(signals.index)
        prices_aligned = prices.loc[common_dates]
        signals_aligned = signals.loc[common_dates]
        
        for date in common_dates:
            current_price = prices_aligned.loc[date, 'Adj Close']
            signal = signals_aligned.loc[date, 'in_window']
            
            # Track daily position
            position_value = shares * current_price
            portfolio_value = cash + position_value
            
            portfolio_values.append({
                'date': date,
                'cash': cash,
                'shares': shares,
                'position_value': position_value,
                'portfolio_value': portfolio_value,
                'price': current_price
            })
            
            # Trading logic
            if signal == 1 and shares == 0:
                # Enter position (buy)
                trade_value = cash * 0.99  # Leave 1% for costs
                transaction_costs = self.calculate_transaction_costs(trade_value)
                
                # Buy shares
                shares_to_buy = int((trade_value - transaction_costs) / current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price + transaction_costs
                    cash -= cost
                    shares += shares_to_buy
                    
                    trades.append({
                        'date': date,
                        'action': 'buy',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'value': cost,
                        'costs': transaction_costs
                    })
            
            elif signal == 0 and shares > 0:
                # Exit position (sell)
                trade_value = shares * current_price
                transaction_costs = self.calculate_transaction_costs(trade_value)
                
                # Sell shares
                proceeds = trade_value - transaction_costs
                cash += proceeds
                
                trades.append({
                    'date': date,
                    'action': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'value': proceeds,
                    'costs': transaction_costs
                })
                
                shares = 0
        
        # Final liquidation if holding position
        if shares > 0:
            final_date = common_dates[-1]
            final_price = prices_aligned.loc[final_date, 'Adj Close']
            trade_value = shares * final_price
            transaction_costs = self.calculate_transaction_costs(trade_value)
            
            proceeds = trade_value - transaction_costs
            cash += proceeds
            
            trades.append({
                'date': final_date,
                'action': 'sell',
                'price': final_price,
                'shares': shares,
                'value': proceeds,
                'costs': transaction_costs
            })
            
            shares = 0
        
        # Convert to DataFrames
        portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            portfolio_df,
            trades_df,
            initial_capital
        )
        
        results = {
            'portfolio': portfolio_df,
            'trades': trades_df,
            'metrics': metrics,
            'initial_capital': initial_capital,
            'final_value': cash
        }
        
        return results
    
    def _calculate_metrics(self,
                           portfolio_df: pd.DataFrame,
                           trades_df: pd.DataFrame,
                           initial_capital: float) -> Dict:
        """Calculate performance metrics."""
        
        # Returns
        portfolio_values = portfolio_df['portfolio_value']
        returns = portfolio_values.pct_change().dropna()
        
        total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
        
        # Annualized return
        years = len(portfolio_df) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if len(trades_df) > 0:
            # Match buy/sell pairs
            buy_trades = trades_df[trades_df['action'] == 'buy']
            sell_trades = trades_df[trades_df['action'] == 'sell']
            
            num_trades = min(len(buy_trades), len(sell_trades))
            
            if num_trades > 0:
                pnls = []
                for i in range(num_trades):
                    buy_value = buy_trades.iloc[i]['value']
                    sell_value = sell_trades.iloc[i]['value']
                    pnl = sell_value - buy_value
                    pnls.append(pnl)
                
                winning_trades = sum(1 for pnl in pnls if pnl > 0)
                win_rate = winning_trades / num_trades
            else:
                num_trades = 0
                win_rate = 0
        else:
            num_trades = 0
            win_rate = 0
        
        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'total_costs': trades_df['costs'].sum() if len(trades_df) > 0 else 0
        }
        
        return metrics
    
    def compare_to_benchmark(self,
                             strategy_results: Dict,
                             benchmark_prices: pd.DataFrame) -> Dict:
        """
        Compare strategy to buy-and-hold benchmark.
        
        Args:
            strategy_results: Strategy backtest results
            benchmark_prices: SPY price data
            
        Returns:
            Comparison metrics
        """
        # Align dates
        strategy_dates = strategy_results['portfolio'].index
        benchmark_aligned = benchmark_prices.loc[strategy_dates, 'Adj Close']
        
        # Calculate benchmark returns
        benchmark_returns = benchmark_aligned.pct_change().dropna()
        benchmark_total_return = (benchmark_aligned.iloc[-1] - benchmark_aligned.iloc[0]) / benchmark_aligned.iloc[0]
        
        years = len(strategy_dates) / 252
        benchmark_annualized = (1 + benchmark_total_return) ** (1 / years) - 1 if years > 0 else 0
        
        benchmark_sharpe = (benchmark_returns.mean() / benchmark_returns.std()) * np.sqrt(252)
        
        # Comparison
        comparison = {
            'strategy_total_return': strategy_results['metrics']['total_return_pct'],
            'benchmark_total_return': benchmark_total_return * 100,
            'excess_return': (strategy_results['metrics']['total_return'] - benchmark_total_return) * 100,
            'strategy_sharpe': strategy_results['metrics']['sharpe_ratio'],
            'benchmark_sharpe': benchmark_sharpe,
            'sharpe_improvement': strategy_results['metrics']['sharpe_ratio'] - benchmark_sharpe
        }
        
        return comparison


if __name__ == "__main__":
    # Test backtester
    from data_acquisition import DataAcquisition
    from signal_generator import SignalGenerator
    
    # Fetch data
    data_acq = DataAcquisition()
    dataset = data_acq.fetch_full_dataset()
    
    # Generate signals
    signal_gen = SignalGenerator()
    signals, windows = signal_gen.generate_signal_series(dataset['amzn_prices'].index)
    
    # Apply filters
    filtered_signals = signal_gen.apply_market_filters(
        signals,
        dataset['spy_prices']['Adj Close'],
        dataset['vix']
    )
    
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest(
        dataset['amzn_prices'],
        filtered_signals
    )
    
    print("=== Backtest Results ===")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['metrics']['total_return_pct']:.2f}%")
    print(f"Annualized Return: {results['metrics']['annualized_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown_pct']:.2f}%")
    print(f"Number of Trades: {results['metrics']['num_trades']}")
    print(f"Win Rate: {results['metrics']['win_rate']*100:.1f}%")
    
    # Compare to benchmark
    comparison = backtester.compare_to_benchmark(
        results,
        dataset['spy_prices']
    )
    
    print("\n=== vs Benchmark (SPY) ===")
    print(f"Strategy Return: {comparison['strategy_total_return']:.2f}%")
    print(f"Benchmark Return: {comparison['benchmark_total_return']:.2f}%")
    print(f"Excess Return: {comparison['excess_return']:.2f}%")
    print(f"Strategy Sharpe: {comparison['strategy_sharpe']:.2f}")
    print(f"Benchmark Sharpe: {comparison['benchmark_sharpe']:.2f}")
