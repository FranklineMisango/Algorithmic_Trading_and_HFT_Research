"""
Backtester for Crypto Macro-Fundamental Strategy

Vectorized backtest with transaction costs, position sizing, and leverage rules.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
except ImportError:
    print("Warning: scipy not installed")


class CryptoBacktester:
    """Backtest crypto macro-fundamental strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize backtester."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.backtest_config = self.config['backtest']
        self.signals_config = self.config['signals']
        self.risk_config = self.config['risk']
    
    def calculate_position_sizes(
        self,
        predictions: np.ndarray,
        realized_vol: float,
        features: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate position sizes with volatility scaling and leverage rules.
        
        Args:
            predictions: Model predictions
            realized_vol: Realized volatility
            features: Feature DataFrame (for risk-off conditions)
        
        Returns:
            Position sizes (as fraction of capital, can exceed 1.0 for leverage)
        """
        target_vol = self.signals_config['sizing']['target_volatility']
        max_leverage = self.signals_config['leverage']['max_leverage']
        
        # Volatility scaler
        K = target_vol / (realized_vol + 1e-8)
        
        # Base position = K * predicted_return
        positions = K * predictions
        
        # Apply leverage limits
        positions = np.clip(positions, -max_leverage, max_leverage)
        
        # Risk-off condition: reduce to 1x if both macro and risk premium are extreme
        if 'external_macro' in features.columns and 'crypto_risk_premium_5d' in features.columns:
            macro_quartile = pd.qcut(features['external_macro'], 4, labels=[1,2,3,4], duplicates='drop')
            risk_quartile = pd.qcut(features['crypto_risk_premium_5d'], 4, labels=[1,2,3,4], duplicates='drop')
            
            # Check if both in top quartile (extreme risk-off)
            risk_off_mask = (macro_quartile == 4) & (risk_quartile == 4)
            
            # Reduce leverage
            positions[risk_off_mask] = np.sign(positions[risk_off_mask]) * np.minimum(
                np.abs(positions[risk_off_mask]), 1.0
            )
        
        return positions
    
    def run_backtest(
        self,
        results_df: pd.DataFrame,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        initial_capital: float = None
    ) -> Dict:
        """
        Run vectorized backtest.
        
        Args:
            results_df: DataFrame with 'actual' and 'predicted' returns
            prices: Price DataFrame
            features: Feature DataFrame (for position sizing)
            initial_capital: Starting capital
        
        Returns:
            Backtest results dict
        """
        if initial_capital is None:
            initial_capital = self.backtest_config['initial_capital']
        
        # Calculate realized volatility (20-day rolling)
        btc_returns = prices['BTC-USD'].pct_change()
        realized_vol = btc_returns.rolling(20).std().mean() * np.sqrt(252)
        
        # Calculate position sizes
        predictions = results_df['predicted'].values
        features_aligned = features.reindex(results_df.index).ffill()
        
        positions = self.calculate_position_sizes(
            predictions,
            realized_vol,
            features_aligned
        )
        
        # Actual returns
        actual_returns = results_df['actual'].values
        
        # Calculate position changes (for transaction costs)
        position_changes = np.abs(np.diff(positions, prepend=0))
        
        # Transaction costs
        commission_bps = self.backtest_config['costs']['commission_bps']
        slippage_bps = self.backtest_config['costs']['slippage_bps']
        total_cost_bps = commission_bps + slippage_bps
        
        transaction_costs = position_changes * (total_cost_bps / 10000)
        
        # Strategy returns (before costs)
        gross_returns = positions * actual_returns
        
        # Net returns (after costs)
        net_returns = gross_returns - transaction_costs
        
        # Portfolio value over time
        portfolio_values = initial_capital * np.cumprod(1 + net_returns)
        
        # Calculate drawdowns
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (cummax - portfolio_values) / cummax
        
        # Stop-loss rule: 10% drawdown triggers 50% position reduction
        stop_loss_threshold = self.risk_config['stop_loss']['threshold']
        
        stop_loss_triggered = drawdowns > stop_loss_threshold
        
        if np.any(stop_loss_triggered):
            first_trigger = np.where(stop_loss_triggered)[0][0]
            print(f"\nWarning: Stop-loss triggered at index {first_trigger}")
            print(f"  Drawdown: {drawdowns[first_trigger]:.2%}")
            
            # Reduce positions by 50% after trigger
            positions[first_trigger:] *= 0.5
            
            # Recalculate returns with reduced positions
            gross_returns = positions * actual_returns
            position_changes = np.abs(np.diff(positions, prepend=0))
            transaction_costs = position_changes * (total_cost_bps / 10000)
            net_returns = gross_returns - transaction_costs
            portfolio_values = initial_capital * np.cumprod(1 + net_returns)
            cummax = np.maximum.accumulate(portfolio_values)
            drawdowns = (cummax - portfolio_values) / cummax
        
        # Create results DataFrame
        backtest_df = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'gross_return': gross_returns,
            'net_return': net_returns,
            'position': positions,
            'transaction_cost': transaction_costs,
            'drawdown': drawdowns
        }, index=results_df.index)
        
        # Calculate metrics
        metrics = self.calculate_metrics(backtest_df, initial_capital)
        
        return {
            'backtest_df': backtest_df,
            'metrics': metrics
        }
    
    def calculate_metrics(
        self,
        backtest_df: pd.DataFrame,
        initial_capital: float
    ) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            backtest_df: Backtest results DataFrame
            initial_capital: Initial capital
        
        Returns:
            Metrics dict
        """
        portfolio_values = backtest_df['portfolio_value'].values
        returns = backtest_df['net_return'].values
        
        # Total return
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        
        # Annualized return
        n_days = len(portfolio_values)
        n_years = n_days / 252
        annualized_return = (portfolio_values[-1] / initial_capital) ** (1 / n_years) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = annualized_return / (volatility + 1e-8)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        sortino = annualized_return / downside_std
        
        # Max drawdown
        max_drawdown = np.max(backtest_df['drawdown'].values)
        
        # Calmar ratio
        calmar = annualized_return / (max_drawdown + 1e-8)
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_days': n_days
        }
    
    def backtest_buy_and_hold(
        self,
        prices: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        initial_capital: float = None
    ) -> Dict:
        """
        Backtest simple buy-and-hold benchmark.
        
        Args:
            prices: Price DataFrame
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
        
        Returns:
            Benchmark results
        """
        if initial_capital is None:
            initial_capital = self.backtest_config['initial_capital']
        
        # Filter dates
        prices_filtered = prices.loc[start_date:end_date]
        
        # BTC buy-and-hold
        btc_returns = prices_filtered['BTC-USD'].pct_change().fillna(0)
        portfolio_values = initial_capital * np.cumprod(1 + btc_returns)
        
        # Create DataFrame
        backtest_df = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'net_return': btc_returns,
            'drawdown': (np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values)
        }, index=prices_filtered.index)
        
        # Calculate metrics
        metrics = self.calculate_metrics(backtest_df, initial_capital)
        
        return {
            'backtest_df': backtest_df,
            'metrics': metrics
        }
    
    def statistical_tests(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict:
        """
        Perform statistical tests.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
        
        Returns:
            Test results
        """
        # One-sided t-test
        t_stat, p_value = stats.ttest_ind(
            strategy_returns,
            benchmark_returns,
            alternative='greater'
        )
        
        is_significant = p_value < 0.05
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'mean_diff': np.mean(strategy_returns) - np.mean(benchmark_returns)
        }
    
    def capacity_analysis(
        self,
        backtest_df: pd.DataFrame,
        capital_levels: List[float] = None
    ) -> pd.DataFrame:
        """
        Analyze strategy capacity with price impact.
        
        Args:
            backtest_df: Backtest results
            capital_levels: List of capital levels to test
        
        Returns:
            DataFrame with capacity analysis
        """
        if capital_levels is None:
            capital_levels = self.config['evaluation']['capacity']['test_capital_levels']
        
        price_impact_bps = self.config['evaluation']['capacity']['price_impact_bps']
        
        baseline_sharpe = self.calculate_metrics(
            backtest_df,
            capital_levels[0]
        )['sharpe_ratio']
        
        results = []
        
        for capital in capital_levels:
            # Estimate average daily trade size
            avg_position = np.mean(np.abs(backtest_df['position'].values))
            avg_trade_size = capital * avg_position
            
            # Calculate price impact
            # Impact = 0.20% per $1M traded
            impact_bps = (avg_trade_size / 1e6) * price_impact_bps
            impact_pct = impact_bps / 10000
            
            # Adjust returns for impact
            adjusted_returns = backtest_df['net_return'].values - impact_pct
            
            # Recalculate Sharpe
            adjusted_sharpe = (
                np.mean(adjusted_returns) * 252 / (np.std(adjusted_returns) * np.sqrt(252) + 1e-8)
            )
            
            # Degradation
            degradation = (baseline_sharpe - adjusted_sharpe) / baseline_sharpe
            
            results.append({
                'capital': capital,
                'avg_trade_size': avg_trade_size,
                'price_impact_bps': impact_bps,
                'sharpe_ratio': adjusted_sharpe,
                'degradation_pct': degradation * 100
            })
        
        return pd.DataFrame(results)


# Test code
if __name__ == "__main__":
    from data_acquisition import DataAcquisition
    from feature_engineering import FeatureEngineer
    from ml_model import CryptoMLModel
    
    # Load and prepare data
    data_acq = DataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset()
    
    engineer = FeatureEngineer('config.yaml')
    features = engineer.engineer_all_features(dataset['prices'], dataset['events'])
    target = engineer.create_target_variable(dataset['prices'])
    
    # Train model (simplified for testing)
    model = CryptoMLModel('config.yaml')
    ml_results = model.walk_forward_validation(features, target, optimize_hyperparams=False)
    
    # Backtest
    backtester = CryptoBacktester('config.yaml')
    
    print("\nRunning strategy backtest...")
    strategy_results = backtester.run_backtest(
        ml_results['results_df'],
        dataset['prices'],
        features
    )
    
    print("\n" + "="*60)
    print("STRATEGY PERFORMANCE")
    print("="*60)
    for metric, value in strategy_results['metrics'].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Buy-and-hold benchmark
    print("\nRunning buy-and-hold benchmark...")
    start_date = ml_results['results_df'].index[0]
    end_date = ml_results['results_df'].index[-1]
    
    benchmark_results = backtester.backtest_buy_and_hold(
        dataset['prices'],
        start_date,
        end_date
    )
    
    print("\n" + "="*60)
    print("BUY-AND-HOLD BENCHMARK")
    print("="*60)
    for metric, value in benchmark_results['metrics'].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
