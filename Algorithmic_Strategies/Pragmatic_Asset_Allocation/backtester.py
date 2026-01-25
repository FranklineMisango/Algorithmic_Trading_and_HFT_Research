"""
Pragmatic Asset Allocation Model - Backtester Module
Executes the complete strategy with performance tracking, benchmarking, and risk analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PragmaticAssetAllocationBacktester:
    """
    Backtester for Pragmatic Asset Allocation Model.
    Executes the strategy and provides comprehensive performance analysis.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.backtest_config = self.config['backtest']
        self.assets = self.config['assets']

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def run_backtest(self, signals_dict: Dict[str, pd.DataFrame],
                    price_data: pd.DataFrame, portfolio_results: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Run the complete backtest simulation.

        Args:
            signals_dict: Dictionary of signal DataFrames
            price_data: Combined price data for all assets
            portfolio_results: Portfolio construction results

        Returns:
            Dictionary with backtest results and performance metrics
        """
        try:
            logger.info("Starting Pragmatic Asset Allocation backtest")

            # Extract portfolio history
            portfolio_history = portfolio_results.get('portfolio_history')
            if portfolio_history is None or portfolio_history.empty:
                logger.error("No portfolio history available")
                return {}

            # Simulate daily portfolio values between rebalances
            daily_portfolio_values = self._simulate_daily_performance(
                portfolio_history, price_data
            )

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(daily_portfolio_values)

            # Generate benchmark comparisons
            benchmark_results = self._run_benchmarks(daily_portfolio_values.index[0],
                                                   daily_portfolio_values.index[-1],
                                                   price_data)

            # Risk analysis
            risk_metrics = self._calculate_risk_metrics(daily_portfolio_values, benchmark_results)

            # Compile results
            backtest_results = {
                'portfolio_values': daily_portfolio_values,
                'performance_metrics': performance_metrics,
                'benchmark_results': benchmark_results,
                'risk_metrics': risk_metrics,
                'portfolio_history': portfolio_history,
                'signals_summary': self._extract_signals_summary(signals_dict),
                'dates': daily_portfolio_values.index.strftime('%Y-%m-%d').tolist()
            }

            logger.info("Backtest completed successfully")
            return backtest_results

        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {}

    def _simulate_daily_performance(self, portfolio_history: pd.DataFrame,
                                  price_data: pd.DataFrame) -> pd.Series:
        """
        Simulate daily portfolio performance between rebalance dates.

        Args:
            portfolio_history: DataFrame with rebalance information
            price_data: Daily price data for all assets

        Returns:
            Series with daily portfolio values
        """
        try:
            # Get all dates in the backtest period
            start_date = pd.to_datetime(self.backtest_config['start_date'])
            end_date = pd.to_datetime(self.backtest_config['end_date'])
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

            portfolio_values = pd.Series(index=all_dates, dtype=float)

            # Sort portfolio history by date
            portfolio_history = portfolio_history.sort_values('date')

            # Initialize with starting capital
            current_value = self.backtest_config['initial_capital']
            current_positions = {'CASH': current_value}

            for i, date in enumerate(all_dates):
                date_str = date.strftime('%Y-%m-%d')

                # Check if this is a rebalance date
                rebalance_mask = portfolio_history['date'] == date
                if rebalance_mask.any():
                    rebalance_record = portfolio_history[rebalance_mask].iloc[0]
                    current_positions = rebalance_record['positions']
                    current_value = rebalance_record['portfolio_value']

                # Calculate portfolio value for this day
                daily_value = self._calculate_daily_portfolio_value(
                    current_positions, price_data, date_str
                )

                portfolio_values.loc[date] = daily_value

            # Forward fill any missing values
            portfolio_values = portfolio_values.fillna(method='ffill')

            return portfolio_values

        except Exception as e:
            logger.error(f"Error simulating daily performance: {str(e)}")
            return pd.Series()

    def _calculate_daily_portfolio_value(self, positions: Dict[str, float],
                                       price_data: pd.DataFrame, date_str: str) -> float:
        """
        Calculate portfolio value for a specific date.

        Args:
            positions: Current position values
            price_data: Price data
            date_str: Date string

        Returns:
            Portfolio value
        """
        try:
            total_value = 0

            for asset, position_value in positions.items():
                if asset == 'CASH':
                    total_value += position_value
                elif asset in price_data.columns.levels[0]:
                    # Get the price for this asset on this date
                    asset_prices = price_data[asset]['Close']

                    # Find the closest available price (forward fill)
                    price_series = asset_prices.loc[:date_str]
                    if not price_series.empty:
                        current_price = price_series.iloc[-1]

                        # Calculate shares and current value
                        original_price = current_price  # Simplified - should track original purchase price
                        shares = position_value / original_price
                        current_value = shares * current_price
                        total_value += current_value
                    else:
                        # If no price available, keep original position value
                        total_value += position_value
                else:
                    # Unknown asset, keep position value
                    total_value += position_value

            return total_value

        except Exception as e:
            logger.error(f"Error calculating daily portfolio value: {str(e)}")
            return 0

    def _calculate_performance_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            portfolio_values: Daily portfolio values

        Returns:
            Dictionary of performance metrics
        """
        try:
            # Basic return metrics
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            annual_return = total_return / (len(portfolio_values) / 252)

            # Volatility metrics
            daily_returns = portfolio_values.pct_change().dropna()
            annual_volatility = daily_returns.std() * np.sqrt(252)

            # Sharpe ratio
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

            # Drawdown metrics
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = drawdowns.min()

            # Other metrics
            win_rate = (daily_returns > 0).mean()
            sortino_ratio = self._calculate_sortino_ratio(daily_returns, risk_free_rate)
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')

            # Benchmark against buy-and-hold of 60/40 portfolio
            buy_hold_return = annual_return  # Placeholder - would calculate actual benchmark

            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'total_days': len(portfolio_values),
                'total_trades': 0,  # Would track actual trades
                'avg_daily_turnover': 0  # Would calculate from position changes
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        try:
            excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)

            annual_excess_return = excess_returns.mean() * 252

            return annual_excess_return / downside_deviation if downside_deviation != 0 else float('inf')

        except Exception as e:
            return 0

    def _run_benchmarks(self, start_date: datetime, end_date: datetime,
                       price_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Run benchmark strategies for comparison.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            price_data: Price data for benchmark assets

        Returns:
            Dictionary of benchmark results
        """
        try:
            benchmarks = {}

            # 60/40 Benchmark (60% S&P 500, 40% bonds)
            spx_ticker = "^GSPC"  # S&P 500
            bond_ticker = "IEF"   # 10-year Treasury ETF

            if spx_ticker in price_data.columns.levels[0] and bond_ticker in price_data.columns.levels[0]:
                spx_prices = price_data[spx_ticker]['Close'].loc[start_date:end_date]
                bond_prices = price_data[bond_ticker]['Close'].loc[start_date:end_date]

                # Calculate 60/40 returns
                spx_returns = spx_prices.pct_change().fillna(0)
                bond_returns = bond_prices.pct_change().fillna(0)

                benchmark_returns = 0.6 * spx_returns + 0.4 * bond_returns
                benchmark_cumulative = (1 + benchmark_returns).cumprod()

                benchmarks['60_40'] = {
                    'total_return': benchmark_cumulative.iloc[-1] - 1,
                    'annual_return': benchmark_returns.mean() * 252,
                    'annual_volatility': benchmark_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252)),
                    'max_drawdown': self._calculate_max_drawdown(benchmark_cumulative),
                    'daily_returns': benchmark_returns.tolist()
                }

            # Risky assets only benchmark
            risky_tickers = [asset['ticker'] for asset in self.assets['risky']]
            risky_benchmark = self._calculate_equal_weight_benchmark(
                risky_tickers, price_data, start_date, end_date
            )
            if risky_benchmark:
                benchmarks['risky_assets_only'] = risky_benchmark

            # Hedging assets only benchmark
            hedging_tickers = [asset['ticker'] for asset in self.assets['hedging']]
            hedging_benchmark = self._calculate_equal_weight_benchmark(
                hedging_tickers, price_data, start_date, end_date
            )
            if hedging_benchmark:
                benchmarks['hedging_assets_only'] = hedging_benchmark

            return benchmarks

        except Exception as e:
            logger.error(f"Error running benchmarks: {str(e)}")
            return {}

    def _calculate_equal_weight_benchmark(self, tickers: List[str], price_data: pd.DataFrame,
                                        start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate equal-weighted benchmark for given tickers."""
        try:
            available_tickers = [t for t in tickers if t in price_data.columns.levels[0]]
            if not available_tickers:
                return {}

            # Calculate equal-weighted returns
            returns_list = []
            for ticker in available_tickers:
                asset_returns = price_data[ticker]['Close'].loc[start_date:end_date].pct_change().fillna(0)
                returns_list.append(asset_returns)

            if returns_list:
                benchmark_returns = pd.concat(returns_list, axis=1).mean(axis=1)
                benchmark_cumulative = (1 + benchmark_returns).cumprod()

                return {
                    'total_return': benchmark_cumulative.iloc[-1] - 1,
                    'annual_return': benchmark_returns.mean() * 252,
                    'annual_volatility': benchmark_returns.std() * np.sqrt(252),
                    'sharpe_ratio': (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252)),
                    'max_drawdown': self._calculate_max_drawdown(benchmark_cumulative),
                    'daily_returns': benchmark_returns.tolist()
                }

            return {}

        except Exception as e:
            logger.error(f"Error calculating equal weight benchmark: {str(e)}")
            return {}

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        try:
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            return drawdowns.min()
        except Exception:
            return 0

    def _calculate_risk_metrics(self, portfolio_values: pd.Series,
                              benchmark_results: Dict[str, Dict[str, float]]) -> Dict[str, any]:
        """
        Calculate risk metrics and benchmark comparisons.

        Args:
            portfolio_values: Strategy portfolio values
            benchmark_results: Benchmark performance results

        Returns:
            Dictionary of risk metrics
        """
        try:
            daily_returns = portfolio_values.pct_change().dropna()

            risk_metrics = {
                'var_95': np.percentile(daily_returns, 5),  # 95% VaR
                'cvar_95': daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean(),
                'skewness': daily_returns.skew(),
                'kurtosis': daily_returns.kurtosis(),
                'worst_day': daily_returns.min(),
                'best_day': daily_returns.max(),
                'avg_up_day': daily_returns[daily_returns > 0].mean(),
                'avg_down_day': daily_returns[daily_returns < 0].mean()
            }

            # Benchmark comparisons
            if benchmark_results:
                for bench_name, bench_data in benchmark_results.items():
                    if 'daily_returns' in bench_data:
                        bench_returns = pd.Series(bench_data['daily_returns'])
                        # Calculate alpha, beta, etc. (simplified)
                        risk_metrics[f'{bench_name}_alpha'] = (
                            daily_returns.mean() - bench_returns.mean()
                        ) * 252

            return risk_metrics

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}

    def _extract_signals_summary(self, signals_dict: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Extract summary statistics from signals."""
        try:
            summary = {}

            for signal_type, df in signals_dict.items():
                if not df.empty:
                    summary[signal_type] = {
                        'total_signals': len(df),
                        'date_range': [df.index.min().strftime('%Y-%m-%d'),
                                     df.index.max().strftime('%Y-%m-%d')],
                        'columns': df.columns.tolist()
                    }

            return summary

        except Exception as e:
            logger.error(f"Error extracting signals summary: {str(e)}")
            return {}

    def generate_performance_report(self, backtest_results: Dict[str, any],
                                  save_path: str = "results") -> None:
        """
        Generate comprehensive performance report with charts.

        Args:
            backtest_results: Complete backtest results
            save_path: Path to save report files
        """
        try:
            Path(save_path).mkdir(exist_ok=True)

            # Extract data
            portfolio_values = backtest_results.get('portfolio_values', pd.Series())
            performance_metrics = backtest_results.get('performance_metrics', {})
            benchmark_results = backtest_results.get('benchmark_results', {})

            if portfolio_values.empty:
                logger.error("No portfolio values available for report")
                return

            # Create performance charts
            self._create_performance_charts(portfolio_values, benchmark_results, save_path)

            # Generate metrics table
            self._create_metrics_table(performance_metrics, benchmark_results, save_path)

            # Save detailed results
            results_file = Path(save_path) / "backtest_results.json"
            with open(results_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = {}
                for key, value in backtest_results.items():
                    if isinstance(value, pd.Series):
                        json_results[key] = value.tolist()
                    elif isinstance(value, pd.DataFrame):
                        json_results[key] = value.to_dict()
                    elif isinstance(value, dict):
                        json_results[key] = value
                    else:
                        json_results[key] = str(value)

                json.dump(json_results, f, indent=2, default=str)

            logger.info(f"Performance report saved to {save_path}")

        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")

    def _create_performance_charts(self, portfolio_values: pd.Series,
                                 benchmark_results: Dict[str, Dict[str, float]],
                                 save_path: str) -> None:
        """Create performance visualization charts."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Portfolio value over time
            axes[0,0].plot(portfolio_values.index, portfolio_values.values, linewidth=2, label='Strategy')
            axes[0,0].set_title('Portfolio Value Over Time')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Portfolio Value ($)')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

            # Drawdown chart
            cumulative = portfolio_values / portfolio_values.iloc[0]
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max

            axes[0,1].fill_between(portfolio_values.index, 0, drawdowns, color='red', alpha=0.3)
            axes[0,1].set_title('Portfolio Drawdown')
            axes[0,1].set_xlabel('Date')
            axes[0,1].set_ylabel('Drawdown (%)')
            axes[0,1].set_ylim(bottom=-0.5, top=0)
            axes[0,1].grid(True, alpha=0.3)

            # Rolling Sharpe ratio
            daily_returns = portfolio_values.pct_change().dropna()
            rolling_window = 252
            rolling_sharpe = (daily_returns.rolling(rolling_window).mean() /
                            daily_returns.rolling(rolling_window).std()) * np.sqrt(252)

            axes[1,0].plot(portfolio_values.index[rolling_window:], rolling_sharpe[rolling_window:],
                          linewidth=1, color='purple')
            axes[1,0].set_title(f'Rolling Sharpe Ratio ({rolling_window} days)')
            axes[1,0].set_xlabel('Date')
            axes[1,0].set_ylabel('Sharpe Ratio')
            axes[1,0].grid(True, alpha=0.3)

            # Benchmark comparison
            if benchmark_results:
                strategy_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
                dates = portfolio_values.index

                bench_names = []
                bench_returns = []

                for bench_name, bench_data in benchmark_results.items():
                    if 'total_return' in bench_data:
                        bench_names.append(bench_name.replace('_', ' ').title())
                        bench_returns.append(bench_data['total_return'])

                if bench_names and bench_returns:
                    bars = axes[1,1].bar(['Strategy'] + bench_names, [strategy_return] + bench_returns)
                    axes[1,1].set_title('Total Return Comparison')
                    axes[1,1].set_ylabel('Total Return (%)')

                    # Highlight strategy bar
                    bars[0].set_color('darkblue')

                    # Add value labels
                    for bar, value in zip(bars, [strategy_return] + bench_returns):
                        height = bar.get_height()
                        axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                                     f'{value:.1%}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(Path(save_path) / "performance_charts.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error creating performance charts: {str(e)}")

    def _create_metrics_table(self, performance_metrics: Dict[str, float],
                            benchmark_results: Dict[str, Dict[str, float]],
                            save_path: str) -> None:
        """Create performance metrics table."""
        try:
            # Create comparison table
            metrics_data = {
                'Strategy': list(performance_metrics.keys()),
                'Value': list(performance_metrics.values())
            }

            # Add benchmark columns
            for bench_name, bench_data in benchmark_results.items():
                if 'sharpe_ratio' in bench_data:
                    metrics_data[f'{bench_name.title()}'] = [
                        bench_data.get(metric, '') for metric in performance_metrics.keys()
                    ]

            metrics_df = pd.DataFrame(metrics_data)
            metrics_file = Path(save_path) / "performance_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)

            logger.info(f"Metrics table saved to {metrics_file}")

        except Exception as e:
            logger.error(f"Error creating metrics table: {str(e)}")