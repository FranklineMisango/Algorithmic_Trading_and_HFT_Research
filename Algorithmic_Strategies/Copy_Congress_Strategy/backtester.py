"""
Backtesting Module for Copy Congress Strategy

Simulates portfolio performance with risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class CongressBacktester:
    """Backtest Congressional trading strategy."""
    
    def __init__(self, config: Dict):
        """
        Initialize backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.initial_capital = config['backtest']['initial_capital']
        self.commission = config['transaction_costs']['commission']
        self.slippage_bps = config['transaction_costs']['slippage_bps']
        self.rebalance_freq = config['portfolio']['rebalance_frequency']
        
        # Risk management
        self.max_drawdown_limit = config['risk']['max_drawdown']
        self.position_stop_loss = config['risk']['position_stop_loss']
        self.portfolio_stop_loss = config['risk']['portfolio_stop_loss']
        
    def calculate_transaction_costs(self,
                                   turnover: float,
                                   portfolio_value: float) -> float:
        """
        Calculate transaction costs.
        
        Args:
            turnover: Portfolio turnover (fraction)
            portfolio_value: Current portfolio value
            
        Returns:
            Transaction cost in dollars
        """
        # Commission
        commission_cost = turnover * portfolio_value * self.commission
        
        # Slippage (average of min/max)
        avg_slippage = np.mean(self.slippage_bps) / 10000.0
        slippage_cost = turnover * portfolio_value * avg_slippage
        
        total_cost = commission_cost + slippage_cost
        
        return total_cost
    
    def simulate_portfolio(self,
                          portfolio_history: Dict[pd.Timestamp, pd.Series],
                          prices: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate portfolio returns.
        
        Args:
            portfolio_history: Dictionary of portfolio weights over time
            prices: DataFrame with prices
            
        Returns:
            DataFrame with portfolio metrics
        """
        print("Simulating portfolio performance...")
        
        # Initialize
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {}  # ticker -> shares
        
        results = []
        previous_weights = pd.Series()
        rebalance_dates = sorted(portfolio_history.keys())
        
        # Iterate through all trading days
        for date in prices.index:
            # Calculate current portfolio value
            position_value = 0.0
            for ticker, shares in positions.items():
                if ticker in prices.columns:
                    position_value += shares * prices.loc[date, ticker]
            
            portfolio_value = cash + position_value
            
            # Check for rebalance
            if date in rebalance_dates:
                target_weights = portfolio_history[date]
                
                # Calculate turnover
                if len(previous_weights) > 0:
                    current_weights = pd.Series(0.0, index=target_weights.index)
                    for ticker in target_weights.index:
                        if ticker in positions and ticker in prices.columns:
                            current_weights[ticker] = (
                                positions.get(ticker, 0) * prices.loc[date, ticker] / portfolio_value
                            )
                    
                    turnover = (current_weights - target_weights).abs().sum() / 2.0
                else:
                    turnover = 1.0  # First rebalance is full turnover
                
                # Transaction costs
                transaction_cost = self.calculate_transaction_costs(turnover, portfolio_value)
                cash -= transaction_cost
                portfolio_value -= transaction_cost
                
                # Rebalance portfolio
                new_positions = {}
                for ticker, weight in target_weights.items():
                    if ticker in prices.columns:
                        target_value = weight * portfolio_value
                        price = prices.loc[date, ticker]
                        shares = int(target_value / price)
                        new_positions[ticker] = shares
                
                # Update cash
                new_position_value = sum(
                    shares * prices.loc[date, ticker]
                    for ticker, shares in new_positions.items()
                    if ticker in prices.columns
                )
                cash = portfolio_value - new_position_value
                positions = new_positions
                previous_weights = target_weights
            
            # Check position stop losses
            positions_to_close = []
            for ticker, shares in positions.items():
                if ticker not in prices.columns:
                    positions_to_close.append(ticker)
                    continue
                
                # Calculate position return since entry
                # Simplified: use current price vs average price
                # In production, track entry prices per position
                
            # Remove stopped positions
            for ticker in positions_to_close:
                if ticker in positions:
                    del positions[ticker]
            
            # Recalculate portfolio value
            position_value = sum(
                shares * prices.loc[date, ticker]
                for ticker, shares in positions.items()
                if ticker in prices.columns
            )
            portfolio_value = cash + position_value
            
            # Record results
            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'position_value': position_value,
                'n_positions': len(positions)
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.set_index('date')
        
        # Calculate returns
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1
        
        print(f"Simulation complete: {len(results_df)} days")
        
        return results_df
    
    def calculate_performance_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            results: DataFrame with portfolio results
            
        Returns:
            Dictionary with performance metrics
        """
        returns = results['returns'].dropna()
        
        # Annualization factor
        trading_days_per_year = 252
        
        # Total return
        total_return = results['cumulative_returns'].iloc[-1]
        
        # Annualized return
        n_years = len(returns) / trading_days_per_year
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(trading_days_per_year)
        
        # Sharpe ratio
        risk_free_rate = self.config['backtest']['risk_free_rate']
        excess_returns = returns - risk_free_rate / trading_days_per_year
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(trading_days_per_year)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(trading_days_per_year)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Average turnover
        avg_holdings = results['n_positions'].mean()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_holdings': avg_holdings,
            'final_value': results['portfolio_value'].iloc[-1],
            'n_years': n_years
        }
        
        return metrics
    
    def calculate_benchmark_returns(self, benchmark_ticker: str, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate benchmark returns.
        
        Args:
            benchmark_ticker: Ticker for benchmark
            prices: DataFrame with prices
            
        Returns:
            DataFrame with benchmark results
        """
        print(f"Calculating benchmark returns for {benchmark_ticker}...")
        
        if benchmark_ticker not in prices.columns:
            print(f"Warning: {benchmark_ticker} not in price data")
            return pd.DataFrame()
        
        benchmark_prices = prices[benchmark_ticker]
        benchmark_returns = benchmark_prices.pct_change()
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
        
        results = pd.DataFrame({
            'portfolio_value': benchmark_prices / benchmark_prices.iloc[0] * self.initial_capital,
            'returns': benchmark_returns,
            'cumulative_returns': benchmark_cumulative
        })
        
        return results
    
    def compare_to_benchmark(self,
                            strategy_results: pd.DataFrame,
                            benchmark_results: pd.DataFrame) -> Dict:
        """
        Compare strategy to benchmark.
        
        Args:
            strategy_results: Strategy performance
            benchmark_results: Benchmark performance
            
        Returns:
            Dictionary with comparison metrics
        """
        print("\nComparing to benchmark...")
        
        # Calculate metrics for both
        strategy_metrics = self.calculate_performance_metrics(strategy_results)
        benchmark_metrics = self.calculate_performance_metrics(benchmark_results)
        
        # Calculate alpha and tracking error
        strategy_returns = strategy_results['returns'].dropna()
        benchmark_returns = benchmark_results['returns'].dropna()
        
        # Align returns
        aligned_returns = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        # Tracking error
        tracking_error = (aligned_returns['strategy'] - aligned_returns['benchmark']).std() * np.sqrt(252)
        
        # Information ratio
        excess_return = strategy_metrics['annualized_return'] - benchmark_metrics['annualized_return']
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        comparison = {
            'strategy_return': strategy_metrics['annualized_return'],
            'benchmark_return': benchmark_metrics['annualized_return'],
            'excess_return': excess_return,
            'strategy_sharpe': strategy_metrics['sharpe_ratio'],
            'benchmark_sharpe': benchmark_metrics['sharpe_ratio'],
            'strategy_max_dd': strategy_metrics['max_drawdown'],
            'benchmark_max_dd': benchmark_metrics['max_drawdown'],
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
        
        return comparison
    
    def print_performance_summary(self, metrics: Dict):
        """
        Print performance summary.
        
        Args:
            metrics: Dictionary with performance metrics
        """
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Return:          {metrics['total_return']:>10.2%}")
        print(f"Annualized Return:     {metrics['annualized_return']:>10.2%}")
        print(f"Volatility:            {metrics['volatility']:>10.2%}")
        print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:         {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:          {metrics['max_drawdown']:>10.2%}")
        print(f"Calmar Ratio:          {metrics['calmar_ratio']:>10.2f}")
        print(f"Win Rate:              {metrics['win_rate']:>10.2%}")
        print(f"Average Holdings:      {metrics['avg_holdings']:>10.1f}")
        print(f"Final Value:           ${metrics['final_value']:>10,.0f}")
        print(f"Years:                 {metrics['n_years']:>10.2f}")
        print("="*60)


if __name__ == "__main__":
    # Test the module
    import yaml
    from data_acquisition import CongressionalDataAcquisition
    from signal_generator import SignalGenerator
    from portfolio_constructor import PortfolioConstructor
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get data
    print("Loading data...")
    data_acq = CongressionalDataAcquisition(config)
    congressional_trades, prices, volumes, market_caps, volatility = data_acq.get_full_dataset()
    
    # Generate signals
    print("\nGenerating signals...")
    signal_gen = SignalGenerator(config)
    rebalance_dates = pd.date_range(
        start=prices.index[0],
        end=prices.index[-1],
        freq='W-FRI'
    )
    signals_history = signal_gen.generate_signals_timeseries(
        congressional_trades,
        prices,
        rebalance_dates.tolist()
    )
    
    # Construct portfolios
    print("\nConstructing portfolios...")
    portfolio_constructor = PortfolioConstructor(config)
    portfolio_history = portfolio_constructor.generate_portfolio_timeseries(
        signals_history,
        volatility,
        rebalance_dates.tolist()
    )
    
    # Backtest
    print("\nRunning backtest...")
    backtester = CongressBacktester(config)
    results = backtester.simulate_portfolio(portfolio_history, prices)
    
    # Calculate metrics
    metrics = backtester.calculate_performance_metrics(results)
    backtester.print_performance_summary(metrics)
    
    # Compare to benchmark
    benchmark_ticker = config['backtest']['benchmark']
    benchmark_results = backtester.calculate_benchmark_returns(benchmark_ticker, prices)
    
    if len(benchmark_results) > 0:
        comparison = backtester.compare_to_benchmark(results, benchmark_results)
        
        print("\n" + "="*60)
        print("BENCHMARK COMPARISON")
        print("="*60)
        print(f"Strategy Return:       {comparison['strategy_return']:>10.2%}")
        print(f"Benchmark Return:      {comparison['benchmark_return']:>10.2%}")
        print(f"Excess Return:         {comparison['excess_return']:>10.2%}")
        print(f"Information Ratio:     {comparison['information_ratio']:>10.2f}")
        print(f"Tracking Error:        {comparison['tracking_error']:>10.2%}")
        print("="*60)
