"""
Enhanced Backtesting Engine - Lean/QuantConnect Style Architecture

Features:
- Event-driven architecture with proper order management
- Realistic execution modeling with slippage and market impact
- Portfolio management with margin requirements
- Risk management with position limits and drawdown controls
- Walk-forward optimization framework
- Multi-asset portfolio support
- Comprehensive performance analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from abc import ABC, abstractmethod
import warnings
import logging
from enum import Enum
import yaml
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order object with full lifecycle management."""
    order_id: str
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position tracking with P&L calculations."""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: pd.Timestamp
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    market_value: float = 0.0
    last_update: pd.Timestamp = None

    def update_market_price(self, price: float, timestamp: pd.Timestamp):
        """Update position with new market price."""
        self.market_value = price * abs(self.quantity) * self.get_multiplier()
        if self.quantity > 0:  # Long position
            self.unrealized_pnl = (price - self.entry_price) * self.quantity * self.get_multiplier()
        else:  # Short position
            self.unrealized_pnl = (self.entry_price - price) * abs(self.quantity) * self.get_multiplier()
        self.last_update = timestamp

    def get_multiplier(self) -> int:
        """Get contract multiplier for symbol."""
        multipliers = {'ES': 50, 'NQ': 20}
        return multipliers.get(self.symbol, 1)


@dataclass
class Portfolio:
    """Portfolio management with margin and risk controls."""
    initial_capital: float
    cash: float = 0.0
    total_value: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    margin_used: float = 0.0
    margin_available: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self):
        self.cash = self.initial_capital
        self.total_value = self.initial_capital
        self.margin_available = self.initial_capital

    def update_portfolio_value(self, market_data: Dict[str, float], timestamp: pd.Timestamp):
        """Update portfolio value based on current market prices."""
        total_value = self.cash

        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.update_market_price(market_data[symbol], timestamp)
                total_value += position.market_value

        self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.total_value = total_value + self.unrealized_pnl

    def get_margin_requirement(self, symbol: str, quantity: int, price: float) -> float:
        """Calculate margin requirement for a position."""
        margin_rates = {'ES': 12000, 'NQ': 16000}  # Intraday margins
        base_margin = margin_rates.get(symbol, 10000)
        notional = abs(quantity) * price * Position(symbol, 0, 0, pd.Timestamp.now()).get_multiplier()
        return min(base_margin * abs(quantity), notional * 0.1)  # Min of fixed or 10% of notional


class ExecutionModel:
    """Realistic execution model with slippage and market impact."""

    def __init__(self, config: dict):
        self.config = config
        self.costs = config['strategy']['transaction_costs']

    def execute_order(self, order: Order, market_price: float, timestamp: pd.Timestamp) -> Tuple[float, float, float]:
        """
        Execute order with realistic slippage and commission.

        Returns:
            (execution_price, slippage, commission)
        """
        # Calculate slippage
        slippage_ticks = self.costs['slippage_ticks']
        tick_size = self.costs[order.symbol]['tick_size']

        # Base slippage
        slippage = slippage_ticks * tick_size

        # Market impact for large orders (simplified)
        market_impact = min(order.quantity / 100, 1.0) * tick_size

        total_slippage = slippage + market_impact

        # Apply slippage to execution price
        if order.side == OrderSide.BUY:
            execution_price = market_price + total_slippage
        else:  # SELL
            execution_price = market_price - total_slippage

        # Calculate commission
        commission = self.costs['commission_per_contract'] * abs(order.quantity)

        return execution_price, total_slippage, commission


class RiskManager:
    """Risk management with position limits and drawdown controls."""

    def __init__(self, config: dict, portfolio: Portfolio):
        self.config = config
        self.portfolio = portfolio
        self.risk_limits = config['strategy']['risk_management']

        # Risk tracking
        self.daily_pnl = 0.0
        self.peak_value = portfolio.initial_capital
        self.current_drawdown = 0.0
        self.daily_reset_time = None

    def check_risk_limits(self, timestamp: pd.Timestamp) -> bool:
        """Check if any risk limits are breached."""
        # Daily loss limit
        if self.daily_pnl < -self.risk_limits['max_daily_loss_absolute']:
            logger.warning(f"Daily loss limit breached: ${abs(self.daily_pnl):,.0f}")
            return False

        # Drawdown limit
        current_value = self.portfolio.total_value
        self.current_drawdown = (self.peak_value - current_value) / self.peak_value

        if self.current_drawdown > self.risk_limits['max_drawdown_pct']:
            logger.warning(f"Drawdown limit breached: {self.current_drawdown:.1%}")
            return False

        return True

    def update_daily_pnl(self, timestamp: pd.Timestamp):
        """Reset daily P&L at market open."""
        current_date = timestamp.date()
        if self.daily_reset_time is None or current_date != self.daily_reset_time.date():
            self.daily_pnl = 0.0
            self.daily_reset_time = timestamp

    def validate_order(self, order: Order, market_price: float) -> bool:
        """Validate order against risk limits."""
        # Check margin requirements
        margin_required = self.portfolio.get_margin_requirement(
            order.symbol, order.quantity, market_price
        )

        if margin_required > self.portfolio.margin_available:
            logger.warning(f"Insufficient margin for order {order.order_id}")
            return False

        # Check position limits
        max_contracts = self.config['strategy']['position_sizing']['max_contracts_per_instrument']
        current_position = self.portfolio.positions.get(order.symbol, Position(order.symbol, 0, 0, pd.Timestamp.now()))
        new_quantity = current_position.quantity + order.quantity

        if abs(new_quantity) > max_contracts:
            logger.warning(f"Position limit exceeded for {order.symbol}")
            return False

        return True


class EnhancedBacktester:
    """
    Enhanced event-driven backtester with Lean/QuantConnect-style architecture.
    """

    def __init__(self, config: dict):
        self.config = config

        # Core components
        self.portfolio = Portfolio(config['strategy']['portfolio']['initial_capital'])
        self.execution_model = ExecutionModel(config)
        self.risk_manager = RiskManager(config, self.portfolio)

        # Order management
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0

        # Trading data
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_timestamp: pd.Timestamp = None

        # Results tracking
        self.equity_curve: List[Dict] = []
        self.trades: List[Dict] = []
        self.performance_metrics: Dict = {}

        # Logging
        self.logger = logging.getLogger(f"{__class__.__name__}")

    def load_market_data(self, data: Dict[str, pd.DataFrame]):
        """Load market data for backtesting."""
        self.market_data = data
        self.logger.info(f"Loaded market data for {len(data)} instruments")

    def generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"ORD_{self.order_counter:06d}"

    def submit_order(self, symbol: str, side: OrderSide, quantity: int,
                    order_type: OrderType = OrderType.MARKET,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    tags: Optional[Dict] = None) -> str:
        """
        Submit order to execution system.

        Returns order ID.
        """
        order_id = self.generate_order_id()

        order = Order(
            order_id=order_id,
            timestamp=self.current_timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            tags=tags or {}
        )

        # Validate order
        if not self.market_data.get(symbol) is not None:
            self.logger.error(f"No market data for {symbol}")
            order.status = OrderStatus.REJECTED
            return order_id

        market_price = self.market_data[symbol].loc[self.current_timestamp, 'Close']

        if not self.risk_manager.validate_order(order, market_price):
            order.status = OrderStatus.REJECTED
            return order_id

        # Queue order for execution
        self.orders[order_id] = order
        self.logger.debug(f"Order submitted: {order_id} - {side.value} {quantity} {symbol}")

        return order_id

    def execute_pending_orders(self):
        """Execute all pending orders."""
        for order_id, order in list(self.orders.items()):
            if order.status == OrderStatus.PENDING:
                self._execute_order(order)

    def _execute_order(self, order: Order):
        """Execute individual order."""
        try:
            # Get market price
            market_price = self.market_data[order.symbol].loc[self.current_timestamp, 'Close']

            # Execute through execution model
            execution_price, slippage, commission = self.execution_model.execute_order(
                order, market_price, self.current_timestamp
            )

            # Update order
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.slippage = slippage
            order.commission = commission
            order.status = OrderStatus.FILLED

            # Update portfolio
            self._update_portfolio_from_order(order)

            self.logger.debug(f"Order filled: {order.order_id} at {execution_price:.2f}")

        except Exception as e:
            self.logger.error(f"Order execution failed: {order.order_id} - {str(e)}")
            order.status = OrderStatus.REJECTED

    def _update_portfolio_from_order(self, order: Order):
        """Update portfolio state from filled order."""
        symbol = order.symbol
        quantity = order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity
        price = order.filled_price
        cost = order.commission + order.slippage

        # Update cash
        notional = abs(quantity) * price * Position(symbol, 0, 0, pd.Timestamp.now()).get_multiplier()
        self.portfolio.cash -= notional + cost

        # Update or create position
        if symbol in self.portfolio.positions:
            position = self.portfolio.positions[symbol]
            # Handle position changes (could be increase, decrease, or reversal)
            if (position.quantity > 0 and quantity > 0) or (position.quantity < 0 and quantity < 0):
                # Increasing position - average entry price
                total_quantity = position.quantity + quantity
                total_cost = (position.quantity * position.entry_price) + (quantity * price)
                position.entry_price = total_cost / total_quantity
                position.quantity = total_quantity
            elif abs(quantity) < abs(position.quantity):
                # Reducing position - record realized P&L
                realized_pnl = (price - position.entry_price) * quantity * position.get_multiplier()
                if position.quantity < 0:  # Short position
                    realized_pnl = -realized_pnl
                self.portfolio.realized_pnl += realized_pnl
                position.quantity += quantity
            else:
                # Closing or reversing position
                realized_pnl = (price - position.entry_price) * position.quantity * position.get_multiplier()
                if position.quantity < 0:  # Short position
                    realized_pnl = -realized_pnl
                self.portfolio.realized_pnl += realized_pnl

                if abs(quantity) > abs(position.quantity):
                    # Reversal - close old and open new
                    remaining_quantity = quantity - position.quantity
                    position.quantity = remaining_quantity
                    position.entry_price = price
                    position.entry_time = self.current_timestamp
                else:
                    # Full close
                    del self.portfolio.positions[symbol]
        else:
            # New position
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_time=self.current_timestamp
            )

    def update_portfolio(self):
        """Update portfolio value and risk metrics."""
        # Get current market prices
        market_prices = {}
        for symbol, data in self.market_data.items():
            if self.current_timestamp in data.index:
                market_prices[symbol] = data.loc[self.current_timestamp, 'Close']

        # Update portfolio
        self.portfolio.update_portfolio_value(market_prices, self.current_timestamp)

        # Update risk manager
        self.risk_manager.update_daily_pnl(self.current_timestamp)

        # Record equity curve
        self.equity_curve.append({
            'timestamp': self.current_timestamp,
            'portfolio_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'unrealized_pnl': self.portfolio.unrealized_pnl,
            'realized_pnl': self.portfolio.realized_pnl,
            'drawdown': self.risk_manager.current_drawdown
        })

    def run_backtest(self, signals_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Run backtest with signal data.

        Parameters:
            signals_data: Dict of DataFrames with signals for each strategy

        Returns:
            Equity curve DataFrame
        """
        self.logger.info("="*60)
        self.logger.info("ENHANCED BACKTEST EXECUTION")
        self.logger.info("="*60)

        # Get common timestamp index
        all_indices = [df.index for df in signals_data.values()]
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)

        self.logger.info(f"Backtesting {len(common_index)} bars from {common_index[0]} to {common_index[-1]}")

        # Event loop
        for i, timestamp in enumerate(common_index):
            self.current_timestamp = timestamp

            # Process signals and generate orders
            self._process_signals(signals_data, timestamp)

            # Execute pending orders
            self.execute_pending_orders()

            # Update portfolio
            self.update_portfolio()

            # Check risk limits
            if not self.risk_manager.check_risk_limits(timestamp):
                self.logger.warning("Risk limits breached - halting backtest")
                break

            if i % 1000 == 0:
                self.logger.info(f"Processed {i}/{len(common_index)} bars...")

        # Close all positions at end
        self._close_all_positions()

        # Calculate final metrics
        self._calculate_performance_metrics()

        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)

        self.logger.info("Backtest completed successfully")
        self._print_summary()

        return equity_df

    def _process_signals(self, signals_data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp):
        """Process trading signals and generate orders."""
        for strategy_name, signals_df in signals_data.items():
            if timestamp not in signals_df.index:
                continue

            signal_row = signals_df.loc[timestamp]

            # Check for entry signals
            if signal_row.get('entry_signal', False):
                symbol = signal_row.get('symbol', 'ES')  # Default to ES
                side = OrderSide.BUY if signal_row['signal'] == 1 else OrderSide.SELL
                quantity = abs(int(signal_row.get('position_size', 1)))

                if quantity > 0:
                    self.submit_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        tags={'strategy': strategy_name, 'signal_type': 'entry'}
                    )

            # Check for exit signals
            elif signal_row.get('exit_signal', False):
                # Close position for this strategy
                self._close_strategy_position(strategy_name)

    def _close_strategy_position(self, strategy_name: str):
        """Close position for specific strategy."""
        # Find position associated with this strategy
        # This is a simplified approach - in practice you'd track strategy-position mapping
        for symbol, position in self.portfolio.positions.items():
            if position.quantity != 0:
                quantity = -position.quantity  # Close entire position
                side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY

                self.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=abs(quantity),
                    tags={'strategy': strategy_name, 'signal_type': 'exit'}
                )
                break  # Only close one position per strategy for now

    def _close_all_positions(self):
        """Close all open positions at end of backtest."""
        for symbol, position in list(self.portfolio.positions.items()):
            if position.quantity != 0:
                quantity = -position.quantity
                side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY

                self.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=abs(quantity),
                    tags={'signal_type': 'end_of_backtest'}
                )

    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        if len(self.equity_curve) == 0:
            return

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)

        # Basic returns
        total_return = (equity_df['portfolio_value'].iloc[-1] / self.portfolio.initial_capital) - 1
        annual_return = total_return * (252 / len(equity_df))  # Assuming daily data

        # Risk metrics
        daily_returns = equity_df['portfolio_value'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        max_drawdown = (equity_df['portfolio_value'] / equity_df['portfolio_value'].cummax() - 1).min()

        # Trading metrics
        total_trades = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        winning_trades = len([t for t in self.trades if t.get('pnl_net', 0) > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        self.performance_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_portfolio_value': equity_df['portfolio_value'].iloc[-1],
            'total_pnl': self.portfolio.realized_pnl + self.portfolio.unrealized_pnl
        }

    def _print_summary(self):
        """Print backtest summary."""
        m = self.performance_metrics

        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(f"Initial Capital: ${self.portfolio.initial_capital:,.0f}")
        print(f"Final Portfolio Value: ${m['final_portfolio_value']:,.0f}")
        print(f"Total Return: {m['total_return']:.2%}")
        print(f"Annual Return: {m['annual_return']:.2%}")
        print(f"Volatility: {m['volatility']:.2%}")
        print(f"Sharpe Ratio: {m['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {m['max_drawdown']:.2%}")
        print(f"Total Trades: {m['total_trades']}")
        print(f"Win Rate: {m['win_rate']:.1%}")
        print(f"Total P&L: ${m['total_pnl']:,.0f}")
        print("="*60)

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def save_results(self, output_dir: str = "results"):
        """Save backtest results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(f"{output_dir}/equity_curve.csv", index=False)

        # Save trades
        trades_df = self.get_trades_dataframe()
        trades_df.to_csv(f"{output_dir}/trades.csv", index=False)

        # Save metrics
        with open(f"{output_dir}/performance_metrics.json", 'w') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)

        # Save config
        with open(f"{output_dir}/config_used.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        self.logger.info(f"Results saved to {output_dir}/")


# Parameter Optimization Framework
class ParameterOptimizer:
    """Walk-forward parameter optimization framework."""

    def __init__(self, config: dict, param_ranges: Dict[str, List]):
        self.config = config
        self.param_ranges = param_ranges
        self.results = []

    def optimize_parameters(self, data: Dict[str, pd.DataFrame],
                          start_date: str, end_date: str,
                          metric: str = 'sharpe_ratio') -> Dict:
        """Optimize parameters for given date range."""
        best_params = {}
        best_metric = -np.inf

        # Grid search over parameter combinations
        from itertools import product

        param_keys = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())

        for param_combo in product(*param_values):
            param_dict = dict(zip(param_keys, param_combo))

            # Update config with parameters
            test_config = self._update_config_params(self.config.copy(), param_dict)

            # Run backtest
            backtester = EnhancedBacktester(test_config)
            backtester.load_market_data(data)

            # Generate signals (simplified - would need actual signal generation)
            signals_data = self._generate_signals_for_params(test_config, data, start_date, end_date)

            equity_curve = backtester.run_backtest(signals_data)

            # Evaluate
            current_metric = backtester.performance_metrics.get(metric, -np.inf)

            if current_metric > best_metric:
                best_metric = current_metric
                best_params = param_dict

            self.results.append({
                'params': param_dict,
                'metric': current_metric,
                'equity_curve': equity_curve
            })

        return best_params

    def _update_config_params(self, config: dict, params: Dict) -> dict:
        """Update config with parameter values."""
        # This would need to be customized based on parameter structure
        for param, value in params.items():
            if param in config['strategy']['noise_area']:
                config['strategy']['noise_area'][param] = value
            elif param in config['strategy']['position_sizing']:
                config['strategy']['position_sizing'][param] = value
        return config

    def _generate_signals_for_params(self, config: dict, data: Dict[str, pd.DataFrame],
                                   start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Generate signals with given parameters (placeholder)."""
        # This would integrate with the actual signal generation pipeline
        # For now, return empty signals
        return {}


# Political Season Analysis Framework
class PoliticalSeasonAnalyzer:
    """Analyze strategy performance across political seasons."""

    def __init__(self, config: dict):
        self.config = config
        self.seasons = {
            '2012-2016': {'start': '2012-01-01', 'end': '2016-12-31'},
            '2016-2021': {'start': '2016-01-01', 'end': '2021-12-31'},
            '2021-2024': {'start': '2021-01-01', 'end': '2024-12-31'},
            '2024-2026': {'start': '2024-01-01', 'end': '2026-12-31'}
        }

    def run_seasonal_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Run analysis for each political season."""
        results = {}

        for season_name, dates in self.seasons.items():
            self.logger.info(f"Analyzing season: {season_name}")

            # Filter data for season
            season_data = self._filter_data_by_dates(data, dates['start'], dates['end'])

            if len(season_data) == 0:
                self.logger.warning(f"No data available for season {season_name}")
                continue

            # Optimize parameters for this season
            optimizer = ParameterOptimizer(self.config, self._get_param_ranges())
            best_params = optimizer.optimize_parameters(
                season_data, dates['start'], dates['end']
            )

            # Run backtest with optimal parameters
            season_config = self._update_config_params(self.config.copy(), best_params)
            backtester = EnhancedBacktester(season_config)
            backtester.load_market_data(season_data)

            # Generate signals and run backtest
            signals_data = self._generate_season_signals(season_config, season_data)
            equity_curve = backtester.run_backtest(signals_data)

            results[season_name] = {
                'best_params': best_params,
                'equity_curve': equity_curve,
                'performance_metrics': backtester.performance_metrics,
                'trades': backtester.get_trades_dataframe(),
                'config': season_config
            }

            # Save season results
            self._save_season_results(season_name, results[season_name])

        return results

    def _filter_data_by_dates(self, data: Dict[str, pd.DataFrame],
                            start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Filter data for date range."""
        filtered_data = {}
        for symbol, df in data.items():
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_data[symbol] = df[mask].copy()
        return filtered_data

    def _get_param_ranges(self) -> Dict[str, List]:
        """Get parameter ranges for optimization."""
        return {
            'lookback_days': [10, 20, 30, 60],
            'target_daily_volatility': [0.05, 0.10, 0.15, 0.20],
            'atr_multiplier': [1.0, 1.5, 2.0, 2.5],
            'confirmation_bars': [1, 2, 3],
            'volume_threshold_percentile': [50, 60, 70, 80]
        }

    def _update_config_params(self, config: dict, params: Dict) -> dict:
        """Update config with optimized parameters."""
        param_mapping = {
            'lookback_days': ['strategy', 'noise_area', 'lookback_days'],
            'target_daily_volatility': ['strategy', 'position_sizing', 'target_daily_volatility'],
            'atr_multiplier': ['strategy', 'noise_area', 'atr_multiplier'],
            'confirmation_bars': ['strategy', 'entry_exit', 'confirmation_bars'],
            'volume_threshold_percentile': ['strategy', 'entry_exit', 'volume_threshold_percentile']
        }

        for param, value in params.items():
            if param in param_mapping:
                path = param_mapping[param]
                self._set_nested_value(config, path, value)

        return config

    def _set_nested_value(self, d: dict, path: List[str], value: Any):
        """Set nested dictionary value."""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _generate_season_signals(self, config: dict, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate signals for season (placeholder - integrate with actual signal generation)."""
        # This would integrate with the actual signal generation pipeline
        return {}

    def _save_season_results(self, season_name: str, results: Dict):
        """Save results for a season."""
        output_dir = f"results/political_seasons/{season_name.replace('-', '_')}"
        os.makedirs(output_dir, exist_ok=True)

        # Save equity curve
        results['equity_curve'].to_csv(f"{output_dir}/equity_curve.csv")

        # Save trades
        results['trades'].to_csv(f"{output_dir}/trades.csv", index=False)

        # Save metrics
        with open(f"{output_dir}/performance_metrics.json", 'w') as f:
            json.dump(results['performance_metrics'], f, indent=2, default=str)

        # Save parameters
        with open(f"{output_dir}/optimal_parameters.yaml", 'w') as f:
            yaml.dump(results['best_params'], f, default_flow_style=False)

        # Save config
        with open(f"{output_dir}/config_used.yaml", 'w') as f:
            yaml.dump(results['config'], f, default_flow_style=False)


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize analyzer
    analyzer = PoliticalSeasonAnalyzer(config)

    # Load data (example)
    # data = load_market_data()
    # results = analyzer.run_seasonal_analysis(data)