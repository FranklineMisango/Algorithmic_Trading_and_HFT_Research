"""
Backtesting Module for Deep Learning Options Trading Strategy

Implements comprehensive backtesting with transaction costs, slippage,
and comparison against benchmark strategies (buy-and-hold, momentum, mean-reversion).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from scipy import stats

class OptionsBacktester:
    """
    Backtesting engine for delta-neutral straddle trading strategy.
    Includes transaction costs, slippage, and benchmark comparisons.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        self.results = {}

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler()
            ]
        )

    def run_backtest(self, positions_df: pd.DataFrame, prices_df: pd.DataFrame,
                    options_df: pd.DataFrame) -> dict:
        """
        Run complete backtest of the LSTM strategy.

        Args:
            positions_df: DataFrame with position signals from model
            prices_df: Underlying price data
            options_df: Options data for cost calculation

        Returns:
            Dictionary with backtest results
        """
        self.logger.info("Starting backtest execution")

        # Initialize portfolio
        portfolio = self._initialize_portfolio()

        # Process each trading day
        trading_dates = sorted(positions_df['date'].unique())

        for date in trading_dates:
            try:
                # Get positions for this date
                day_positions = positions_df[positions_df['date'] == date]

                # Execute trades
                portfolio = self._execute_trades(portfolio, day_positions,
                                               options_df, date)

                # Update portfolio value
                portfolio = self._update_portfolio_value(portfolio, prices_df, date)

                # Check risk limits
                portfolio = self._apply_risk_management(portfolio, date)

            except Exception as e:
                self.logger.error(f"Error processing date {date}: {e}")
                continue

        # Calculate performance metrics
        results = self._calculate_performance_metrics(portfolio)

        # Run benchmark comparisons
        benchmark_results = self._run_benchmarks(positions_df, prices_df, options_df)
        results.update(benchmark_results)

        self.results = results
        self.logger.info("Backtest completed")

        return results

    def _initialize_portfolio(self) -> dict:
        """Initialize portfolio with starting capital."""
        return {
            'cash': self.config['backtest']['initial_capital'],
            'positions': {},  # ticker -> position details
            'portfolio_value': self.config['backtest']['initial_capital'],
            'trade_log': [],
            'daily_values': [],
            'date': None
        }

    def _execute_trades(self, portfolio: dict, positions: pd.DataFrame,
                       options_df: pd.DataFrame, date: datetime) -> dict:
        """
        Execute trades based on model signals.

        Args:
            portfolio: Current portfolio state
            positions: Position signals for the day
            options_df: Options data for pricing
            date: Trading date

        Returns:
            Updated portfolio
        """
        total_value = portfolio['portfolio_value']

        for _, position in positions.iterrows():
            ticker = position['ticker']
            signal = position['position_signal']

            # Get current options price
            options_price = self._get_options_price(options_df, ticker, date)

            if options_price is None:
                continue

            # Calculate position size
            position_value = min(
                total_value * self.config['backtest']['max_position_size'],
                total_value * self.config['backtest']['max_single_stock_exposure']
            )

            # Number of straddles to trade
            num_straddles = int(position_value / options_price)

            if num_straddles == 0:
                continue

            # Current position
            current_position = portfolio['positions'].get(ticker, 0)

            # Target position
            target_position = signal * num_straddles

            # Trade size
            trade_size = target_position - current_position

            if abs(trade_size) < 1:  # Minimum trade size
                continue

            # Calculate transaction costs
            trade_cost = self._calculate_transaction_cost(trade_size, options_price)

            # Apply slippage
            execution_price = self._apply_slippage(options_price, trade_size)

            # Execute trade
            trade_value = abs(trade_size) * execution_price + trade_cost

            if trade_value > portfolio['cash']:
                # Scale down trade if insufficient cash
                scale_factor = portfolio['cash'] / trade_value
                trade_size *= scale_factor
                trade_value *= scale_factor

            # Update portfolio
            portfolio['cash'] -= trade_value
            portfolio['positions'][ticker] = target_position

            # Log trade
            trade_record = {
                'date': date,
                'ticker': ticker,
                'signal': signal,
                'trade_size': trade_size,
                'execution_price': execution_price,
                'trade_cost': trade_cost,
                'trade_value': trade_value
            }
            portfolio['trade_log'].append(trade_record)

        portfolio['date'] = date
        return portfolio

    def _get_options_price(self, options_df: pd.DataFrame, ticker: str,
                          date: datetime) -> float:
        """Get straddle price for ticker on given date."""
        try:
            day_options = options_df[
                (options_df['ticker'] == ticker) &
                (options_df['date'] == date)
            ]

            if day_options.empty:
                return None

            # Use ATM straddle price (moneyness closest to 1.0)
            atm_option = day_options.iloc[(day_options['moneyness'] - 1.0).abs().argmin()]
            return atm_option['straddle_price']

        except Exception:
            return None

    def _calculate_transaction_cost(self, trade_size: float, price: float) -> float:
        """Calculate total transaction costs for a trade."""
        # Per contract cost
        per_contract_cost = self.config['backtest']['transaction_cost_per_contract']

        # Bid-ask spread cost
        spread_cost = abs(trade_size) * price * self.config['backtest']['bid_ask_spread']

        # Total cost
        total_cost = abs(trade_size) * per_contract_cost + spread_cost

        return total_cost

    def _apply_slippage(self, price: float, trade_size: float) -> float:
        """Apply slippage model to execution price."""
        slippage_model = self.config['backtest']['slippage_model']

        if slippage_model == 'conservative':
            # Worse price for market orders
            slippage = 0.001  # 0.1% slippage
        elif slippage_model == 'aggressive':
            slippage = 0.0005  # 0.05% slippage
        else:
            return price

        # Apply slippage (worse price for buyer)
        if trade_size > 0:  # Buying
            return price * (1 + slippage)
        else:  # Selling
            return price * (1 - slippage)

    def _update_portfolio_value(self, portfolio: dict, prices_df: pd.DataFrame,
                               date: datetime) -> dict:
        """Update portfolio value based on current positions and prices."""
        total_value = portfolio['cash']

        for ticker, position in portfolio['positions'].items():
            if position == 0:
                continue

            # Get current straddle price
            options_price = self._get_options_price_from_prices(prices_df, ticker, date)

            if options_price is not None:
                position_value = position * options_price
                total_value += position_value

        portfolio['portfolio_value'] = total_value
        portfolio['daily_values'].append({
            'date': date,
            'portfolio_value': total_value,
            'cash': portfolio['cash']
        })

        return portfolio

    def _get_options_price_from_prices(self, prices_df: pd.DataFrame,
                                     ticker: str, date: datetime) -> float:
        """Get options price from prices DataFrame (simplified)."""
        # This is a placeholder - in real implementation, would have options prices
        # For now, return a synthetic price based on underlying volatility
        try:
            underlying_price = prices_df.loc[(ticker, date), 'Adj Close']
            # Synthetic straddle price (simplified model)
            return underlying_price * 0.05  # 5% of underlying price
        except KeyError:
            return None

    def _apply_risk_management(self, portfolio: dict, date: datetime) -> dict:
        """Apply risk management rules."""
        # Check drawdown limit
        if len(portfolio['daily_values']) > 1:
            initial_value = self.config['backtest']['initial_capital']
            current_value = portfolio['portfolio_value']
            drawdown = (initial_value - current_value) / initial_value

            if drawdown > self.config['backtest']['max_drawdown_stop']:
                self.logger.warning(f"Drawdown limit reached ({drawdown:.2%}) on {date}")
                # Close all positions
                portfolio = self._close_all_positions(portfolio, date)

        return portfolio

    def _close_all_positions(self, portfolio: dict, date: datetime) -> dict:
        """Close all positions (risk management)."""
        for ticker in list(portfolio['positions'].keys()):
            portfolio['positions'][ticker] = 0

        self.logger.info(f"All positions closed on {date} for risk management")
        return portfolio

    def _calculate_performance_metrics(self, portfolio: dict) -> dict:
        """Calculate comprehensive performance metrics."""
        daily_values = pd.DataFrame(portfolio['daily_values'])
        trade_log = pd.DataFrame(portfolio['trade_log'])

        if daily_values.empty:
            return {}

        # Basic returns
        daily_values['returns'] = daily_values['portfolio_value'].pct_change()
        daily_values = daily_values.dropna()

        # Annualized metrics
        total_return = (daily_values['portfolio_value'].iloc[-1] /
                       daily_values['portfolio_value'].iloc[0] - 1)

        annual_return = (1 + total_return) ** (252 / len(daily_values)) - 1
        annual_volatility = daily_values['returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        # Drawdown analysis
        cumulative = (1 + daily_values['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Trading metrics
        total_trades = len(trade_log)
        winning_trades = len(trade_log[trade_log['trade_value'] > 0]) if not trade_log.empty else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Turnover
        total_turnover = trade_log['trade_value'].abs().sum() if not trade_log.empty else 0
        avg_turnover = total_turnover / len(daily_values) if len(daily_values) > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_daily_turnover': avg_turnover,
            'daily_returns': daily_values['returns'].tolist(),
            'portfolio_values': daily_values['portfolio_value'].tolist(),
            'dates': daily_values['date'].tolist()
        }

    def _run_benchmarks(self, positions_df: pd.DataFrame, prices_df: pd.DataFrame,
                       options_df: pd.DataFrame) -> dict:
        """Run benchmark strategies for comparison."""
        benchmarks = {}

        if self.config['benchmarks']['buy_and_hold_options']:
            benchmarks['buy_and_hold'] = self._benchmark_buy_and_hold(positions_df, options_df)

        if self.config['benchmarks']['momentum_strategy']:
            benchmarks['momentum'] = self._benchmark_momentum(prices_df, options_df)

        if self.config['benchmarks']['mean_reversion_strategy']:
            benchmarks['mean_reversion'] = self._benchmark_mean_reversion(prices_df, options_df)

        return {'benchmarks': benchmarks}

    def _benchmark_buy_and_hold(self, positions_df: pd.DataFrame,
                               options_df: pd.DataFrame) -> dict:
        """Buy and hold options strategy benchmark."""
        # Simple strategy: buy straddles and hold
        self.logger.info("Running buy-and-hold benchmark")

        portfolio = self._initialize_portfolio()
        trading_dates = sorted(positions_df['date'].unique())

        # Buy on first date
        first_date = trading_dates[0]
        day_positions = positions_df[positions_df['date'] == first_date]

        for _, position in day_positions.iterrows():
            ticker = position['ticker']
            options_price = self._get_options_price(options_df, ticker, first_date)

            if options_price is not None:
                # Buy one straddle per ticker
                portfolio['positions'][ticker] = 1
                portfolio['cash'] -= options_price

        # Hold positions
        for date in trading_dates:
            portfolio = self._update_portfolio_value(portfolio, options_df, date)

        return self._calculate_performance_metrics(portfolio)

    def _benchmark_momentum(self, prices_df: pd.DataFrame,
                          options_df: pd.DataFrame) -> dict:
        """Momentum-based options strategy."""
        # Bet on continuation of underlying momentum
        self.logger.info("Running momentum benchmark")

        portfolio = self._initialize_portfolio()
        trading_dates = sorted(prices_df.index.get_level_values('Date').unique())

        for date in trading_dates:
            try:
                # Calculate momentum signal
                momentum_signals = self._calculate_momentum_signals(prices_df, date)

                # Convert to positions
                for ticker, signal in momentum_signals.items():
                    options_price = self._get_options_price(options_df, ticker, date)

                    if options_price is not None:
                        position_size = signal * (portfolio['portfolio_value'] * 0.1 / options_price)
                        portfolio['positions'][ticker] = position_size

                portfolio = self._update_portfolio_value(portfolio, options_df, date)

            except Exception as e:
                continue

        return self._calculate_performance_metrics(portfolio)

    def _benchmark_mean_reversion(self, prices_df: pd.DataFrame,
                                options_df: pd.DataFrame) -> dict:
        """Mean-reversion based options strategy."""
        # Bet against extreme moves (as mentioned in transcript)
        self.logger.info("Running mean-reversion benchmark")

        portfolio = self._initialize_portfolio()
        trading_dates = sorted(prices_df.index.get_level_values('Date').unique())

        for date in trading_dates:
            try:
                # Calculate mean-reversion signal
                mr_signals = self._calculate_mean_reversion_signals(prices_df, date)

                # Convert to positions
                for ticker, signal in mr_signals.items():
                    options_price = self._get_options_price(options_df, ticker, date)

                    if options_price is not None:
                        position_size = signal * (portfolio['portfolio_value'] * 0.1 / options_price)
                        portfolio['positions'][ticker] = position_size

                portfolio = self._update_portfolio_value(portfolio, options_df, date)

            except Exception as e:
                continue

        return self._calculate_performance_metrics(portfolio)

    def _calculate_momentum_signals(self, prices_df: pd.DataFrame, date: datetime) -> dict:
        """Calculate momentum signals for underlying stocks."""
        signals = {}

        try:
            # 1-month momentum
            end_date = date
            start_date = end_date - timedelta(days=30)

            for ticker in prices_df.index.get_level_values('ticker').unique():
                try:
                    price_data = prices_df.loc[ticker, start_date:end_date]
                    if len(price_data) >= 20:  # At least 20 trading days
                        momentum = (price_data['Adj Close'].iloc[-1] /
                                  price_data['Adj Close'].iloc[0] - 1)
                        signals[ticker] = 1 if momentum > 0 else -1
                except KeyError:
                    continue

        except Exception:
            pass

        return signals

    def _calculate_mean_reversion_signals(self, prices_df: pd.DataFrame, date: datetime) -> dict:
        """Calculate mean-reversion signals for underlying stocks."""
        signals = {}

        try:
            # Check for extreme moves over past week
            end_date = date
            start_date = end_date - timedelta(days=5)

            for ticker in prices_df.index.get_level_values('ticker').unique():
                try:
                    price_data = prices_df.loc[ticker, start_date:end_date]
                    if len(price_data) >= 3:
                        weekly_return = (price_data['Adj Close'].iloc[-1] /
                                       price_data['Adj Close'].iloc[0] - 1)

                        # Signal to bet against extreme moves
                        if weekly_return > 0.05:  # Strong upward move
                            signals[ticker] = -1  # Bet on reversion down
                        elif weekly_return < -0.05:  # Strong downward move
                            signals[ticker] = 1   # Bet on reversion up
                        else:
                            signals[ticker] = 0

                except KeyError:
                    continue

        except Exception:
            pass

        return signals

    def plot_results(self, save_path: str = "results/backtest_results.png"):
        """Plot backtest results and benchmark comparison."""
        if not self.results:
            self.logger.warning("No results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value over time
        dates = pd.to_datetime(self.results.get('dates', []))
        portfolio_values = self.results.get('portfolio_values', [])

        if dates.any() and portfolio_values:
            axes[0, 0].plot(dates, portfolio_values)
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # Drawdown chart
        if portfolio_values:
            cumulative = pd.Series(portfolio_values) / portfolio_values[0]
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max

            axes[0, 1].fill_between(range(len(drawdowns)), 0, drawdowns, color='red', alpha=0.3)
            axes[0, 1].set_title('Portfolio Drawdown')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].set_ylim(bottom=-0.5, top=0)

        # Benchmark comparison
        benchmarks = self.results.get('benchmarks', {})
        strategy_sharpe = self.results.get('sharpe_ratio', 0)

        benchmark_names = list(benchmarks.keys()) + ['LSTM Strategy']
        sharpe_ratios = [benchmarks[name].get('sharpe_ratio', 0) for name in benchmarks.keys()] + [strategy_sharpe]

        axes[1, 0].bar(benchmark_names, sharpe_ratios, color=['blue'] * len(benchmarks) + ['green'])
        axes[1, 0].set_title('Sharpe Ratio Comparison')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Returns distribution
        daily_returns = self.results.get('daily_returns', [])
        if daily_returns:
            axes[1, 1].hist(daily_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 1].axvline(np.mean(daily_returns), color='red', linestyle='--', label='Mean')
            axes[1, 1].set_title('Daily Returns Distribution')
            axes[1, 1].set_xlabel('Daily Return')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()

        plt.tight_layout()

        # Save plot
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Results plot saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # Example usage
    backtester = OptionsBacktester()

    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    positions_data = []
    for date in dates:
        for ticker in tickers:
            positions_data.append({
                'date': date,
                'ticker': ticker,
                'position_signal': np.random.uniform(-1, 1)
            })

    positions_df = pd.DataFrame(positions_data)

    # Mock prices and options data
    prices_data = []
    options_data = []

    for ticker in tickers:
        for date in dates:
            prices_data.append({
                'ticker': ticker,
                'Date': date,
                'Adj Close': 100 + np.random.normal(0, 5),
                'return_1d': np.random.normal(0, 0.02)
            })

            options_data.append({
                'date': date,
                'ticker': ticker,
                'straddle_price': 5 + np.random.normal(0, 1)
            })

    prices_df = pd.DataFrame(prices_data).set_index(['ticker', 'Date'])
    options_df = pd.DataFrame(options_data)

    # Run backtest
    results = backtester.run_backtest(positions_df, prices_df, options_df)
    print(f"Backtest Sharpe Ratio: {results.get('sharpe_ratio', 'N/A')}")

    # Plot results
    backtester.plot_results()