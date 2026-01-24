"""
Backtester
==========

Realistic backtesting with transaction costs, slippage, and risk metrics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List, Optional


class Backtester:
    """
    Perform realistic backtesting of trading strategies.
    
    Includes transaction costs, slippage, market impact, and comprehensive risk metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("FuturesPrediction.Backtester")
        self.backtest_config = config.get("backtesting", {})
        
        self.initial_capital = self.backtest_config.get("initial_capital", 10000)
        self.transaction_cost = self.backtest_config.get("transaction_cost", 0.001)
        self.position_size = self.backtest_config.get("position_size", 0.1)
        self.prediction_threshold = self.backtest_config.get("prediction_threshold", 0.001)
        self.use_realistic_slippage = self.backtest_config.get("use_realistic_slippage", True)
        self.slippage_bps = self.backtest_config.get("slippage_bps", 2)
        self.market_impact_factor = self.backtest_config.get("market_impact_factor", 0.0001)
        
    def backtest_strategy(self, predictions: np.ndarray, y_true: np.ndarray,
                         timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest with realistic constraints.
        
        Args:
            predictions: Model predictions
            y_true: True price changes
            timestamps: Optional timestamps for time-series analysis
            
        Returns:
            Dictionary of backtest results and metrics
        """
        self.logger.info("Running backtest...")
        
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        positions_history = []
        
        for i, (pred, true_change) in enumerate(zip(predictions, y_true)):
            # Trading logic with threshold
            signal = 0
            if pred > self.prediction_threshold:
                signal = 1  # Buy signal
            elif pred < -self.prediction_threshold:
                signal = -1  # Sell signal
            
            # Position change
            if signal != 0 and signal != np.sign(position):
                # Close existing position if opposite signal
                if position != 0:
                    # Exit trade
                    exit_pnl = position * true_change * capital
                    capital += exit_pnl
                    
                    # Transaction costs
                    cost = self.transaction_cost * abs(position) * capital
                    capital -= cost
                    
                    # Slippage
                    if self.use_realistic_slippage:
                        slippage_cost = (self.slippage_bps / 10000) * abs(position) * capital
                        capital -= slippage_cost
                    
                    trades.append({
                        'index': i,
                        'type': 'exit',
                        'position': position,
                        'pnl': exit_pnl,
                        'cost': cost,
                        'capital': capital
                    })
                    
                    position = 0
                
                # Enter new position
                new_position = signal * self.position_size
                
                # Transaction costs
                cost = self.transaction_cost * abs(new_position) * capital
                capital -= cost
                
                # Market impact (larger positions have bigger impact)
                if self.use_realistic_slippage:
                    impact = self.market_impact_factor * abs(new_position) * capital
                    capital -= impact
                
                position = new_position
                
                trades.append({
                    'index': i,
                    'type': 'entry',
                    'position': position,
                    'prediction': pred,
                    'cost': cost,
                    'capital': capital
                })
            
            # Mark-to-market P&L
            if position != 0:
                mtm_pnl = position * true_change * capital
                capital += mtm_pnl
            
            equity_curve.append(capital)
            positions_history.append(position)
        
        # Close any open position at the end
        if position != 0:
            cost = self.transaction_cost * abs(position) * capital
            capital -= cost
            trades.append({
                'index': len(predictions) - 1,
                'type': 'exit',
                'position': position,
                'cost': cost,
                'capital': capital
            })
        
        equity_curve.append(capital)
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, trades, timestamps)
        
        results = {
            'final_capital': capital,
            'equity_curve': equity_curve,
            'trades': trades,
            'positions': positions_history,
            'metrics': metrics
        }
        
        self._log_results(results)
        
        return results
    
    def _calculate_metrics(self, equity_curve: List[float], trades: List[Dict],
                          timestamps: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: List of capital over time
            trades: List of trade dictionaries
            timestamps: Optional timestamps
            
        Returns:
            Dictionary of metrics
        """
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Basic metrics
        total_return = (equity_array[-1] - self.initial_capital) / self.initial_capital
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe(returns)
        max_drawdown = self._calculate_max_drawdown(equity_array)
        
        # Trade statistics
        num_trades = len([t for t in trades if t['type'] == 'entry'])
        winning_trades = len([t for t in trades if t['type'] == 'exit' and t.get('pnl', 0) > 0])
        losing_trades = len([t for t in trades if t['type'] == 'exit' and t.get('pnl', 0) < 0])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        # Profit factor
        total_wins = sum([t.get('pnl', 0) for t in trades if t['type'] == 'exit' and t.get('pnl', 0) > 0])
        total_losses = abs(sum([t.get('pnl', 0) for t in trades if t['type'] == 'exit' and t.get('pnl', 0) < 0]))
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        # Volatility
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        sortino_ratio = self._calculate_sortino(returns)
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': total_return / num_trades if num_trades > 0 else 0
        }
        
        return metrics
    
    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_sortino(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        return np.mean(excess_returns) / downside_std * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Array of equity values
            
        Returns:
            Maximum drawdown (negative value)
        """
        if len(equity_curve) == 0:
            return 0.0
        
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cummax) / cummax
        return np.min(drawdowns)
    
    def _log_results(self, results: Dict[str, Any]):
        """Log backtest results."""
        metrics = results['metrics']
        
        self.logger.info("=" * 60)
        self.logger.info("BACKTEST RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"Final Capital: ${results['final_capital']:,.2f}")
        self.logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
        self.logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        self.logger.info(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        self.logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        self.logger.info(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        self.logger.info(f"Volatility: {metrics['volatility']:.4f}")
        self.logger.info(f"Number of Trades: {metrics['num_trades']}")
        self.logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        self.logger.info(f"Profit Factor: {metrics['profit_factor']:.4f}")
        self.logger.info("=" * 60)
    
    def compare_strategies(self, strategies: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple strategies side-by-side.
        
        Args:
            strategies: Dictionary of strategy_name -> backtest_results
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for name, results in strategies.items():
            metrics = results['metrics'].copy()
            metrics['strategy'] = name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        df = df.set_index('strategy')
        
        self.logger.info("\nStrategy Comparison:")
        self.logger.info(df.to_string())
        
        return df
