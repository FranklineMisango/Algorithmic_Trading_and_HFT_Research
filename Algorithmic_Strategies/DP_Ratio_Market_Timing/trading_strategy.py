"""
Trading Strategy and Backtesting for Dividend-Price Ratio Model

This module implements:
1. Trading rules based on Δlog(D/P) signal
2. Position sizing and risk management
3. Transaction cost modeling
4. Performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class DPRatioTradingStrategy:
    """
    Implements trading strategy based on D/P ratio signals.
    """
    
    def __init__(self, config: dict, model):
        """
        Initialize trading strategy.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        model : DPRatioOLSModel
            Trained OLS model
        """
        self.config = config
        self.model = model
        self.initial_capital = config['backtest']['initial_capital']
        self.commission = config['costs']['commission_pct']
        self.slippage = config['costs']['slippage_pct']
        self.total_cost = config['costs']['total_cost_pct']
        
    def generate_signals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from model predictions.
        
        Signal Logic:
        - Positive Δlog(D/P) → Predict higher returns → LONG
        - Negative Δlog(D/P) → Predict lower returns → REDUCE/CASH
        
        Parameters
        ----------
        X : pd.DataFrame
            Features (delta_log_dp)
            
        Returns
        -------
        pd.DataFrame
            Signals with positions
        """
        signals = pd.DataFrame(index=X.index)
        
        # Get raw signal (Δlog(D/P))
        signals['signal'] = X[X.columns[0]].values
        
        # Get model predictions
        signals['predicted_return'] = self.model.predict(X)
        
        # Trading rule: Long if predicted return > 0, else cash
        threshold = self.config['strategy']['signal_threshold']
        
        signals['position'] = 0.0  # Default: cash
        
        # Long position when prediction is positive
        signals.loc[signals['predicted_return'] > threshold, 'position'] = \
            self.config['strategy']['long_allocation']
        
        # Cash position when prediction is negative
        signals.loc[signals['predicted_return'] <= threshold, 'position'] = \
            self.config['strategy']['short_allocation']
        
        # Calculate position changes (for transaction costs)
        signals['position_change'] = signals['position'].diff().abs()
        
        return signals
    
    def backtest_strategy(self, signals: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """
        Backtest the strategy with transaction costs.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Trading signals
        returns : pd.Series
            Actual returns
            
        Returns
        -------
        pd.DataFrame
            Backtest results
        """
        print("\n" + "="*60)
        print("BACKTESTING STRATEGY")
        print("="*60)
        
        # Align signals and returns
        backtest = signals.copy()
        backtest['market_return'] = returns
        
        # Calculate strategy returns
        # Position from previous period determines current return
        backtest['strategy_return_gross'] = backtest['position'].shift(1) * backtest['market_return']
        
        # Transaction costs (applied when position changes)
        backtest['transaction_costs'] = backtest['position_change'] * self.total_cost
        
        # Net strategy returns
        backtest['strategy_return_net'] = (
            backtest['strategy_return_gross'] - backtest['transaction_costs']
        )
        
        # Buy-and-hold benchmark (100% in market)
        backtest['benchmark_return'] = backtest['market_return']
        
        # Cumulative returns
        backtest['strategy_cumulative'] = (1 + backtest['strategy_return_net']).cumprod()
        backtest['benchmark_cumulative'] = (1 + backtest['benchmark_return']).cumprod()
        
        # Portfolio value
        backtest['portfolio_value'] = self.initial_capital * backtest['strategy_cumulative']
        backtest['benchmark_value'] = self.initial_capital * backtest['benchmark_cumulative']
        
        # Drawdown
        backtest['strategy_peak'] = backtest['portfolio_value'].cummax()
        backtest['strategy_drawdown'] = (
            (backtest['portfolio_value'] - backtest['strategy_peak']) / backtest['strategy_peak']
        )
        
        backtest['benchmark_peak'] = backtest['benchmark_value'].cummax()
        backtest['benchmark_drawdown'] = (
            (backtest['benchmark_value'] - backtest['benchmark_peak']) / backtest['benchmark_peak']
        )
        
        # Track trades
        backtest['trade'] = backtest['position_change'] > 0
        
        print(f"  Period: {backtest.index[0].strftime('%Y-%m-%d')} to {backtest.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Total months: {len(backtest)}")
        print(f"  Number of trades: {backtest['trade'].sum()}")
        
        print("="*60)
        
        return backtest
    
    def calculate_performance_metrics(self, backtest: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Parameters
        ----------
        backtest : pd.DataFrame
            Backtest results
            
        Returns
        -------
        Dict
            Performance metrics
        """
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        # Get returns
        strategy_returns = backtest['strategy_return_net'].dropna()
        benchmark_returns = backtest['benchmark_return'].dropna()
        
        # Time period
        n_months = len(strategy_returns)
        n_years = n_months / 12
        
        # Total returns
        strategy_total = backtest['strategy_cumulative'].iloc[-1] - 1
        benchmark_total = backtest['benchmark_cumulative'].iloc[-1] - 1
        
        # Annualized returns
        strategy_annual = (1 + strategy_total) ** (1 / n_years) - 1
        benchmark_annual = (1 + benchmark_total) ** (1 / n_years) - 1
        
        # Volatility (annualized)
        strategy_vol = strategy_returns.std() * np.sqrt(12)
        benchmark_vol = benchmark_returns.std() * np.sqrt(12)
        
        # Sharpe ratio
        rf_rate = self.config['backtest']['risk_free_rate']
        strategy_sharpe = (strategy_annual - rf_rate) / strategy_vol
        benchmark_sharpe = (benchmark_annual - rf_rate) / benchmark_vol
        
        # Sortino ratio (downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(12)
        strategy_sortino = (strategy_annual - rf_rate) / downside_std if downside_std > 0 else np.nan
        
        # Maximum drawdown
        strategy_max_dd = backtest['strategy_drawdown'].min()
        benchmark_max_dd = backtest['benchmark_drawdown'].min()
        
        # Calmar ratio
        strategy_calmar = strategy_annual / abs(strategy_max_dd) if strategy_max_dd != 0 else np.nan
        
        # Win rate
        win_rate = (strategy_returns > 0).mean()
        
        # Profit factor
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else np.nan
        
        # Number of trades
        num_trades = backtest['trade'].sum()
        
        # Final values
        strategy_final = backtest['portfolio_value'].iloc[-1]
        benchmark_final = backtest['benchmark_value'].iloc[-1]
        
        # Alpha (excess return vs benchmark)
        alpha_annual = strategy_annual - benchmark_annual
        
        # Print metrics
        print(f"\nReturns:")
        print(f"  Strategy Total:        {strategy_total:+.2%}")
        print(f"  Benchmark Total:       {benchmark_total:+.2%}")
        print(f"  Strategy Annualized:   {strategy_annual:+.2%}")
        print(f"  Benchmark Annualized:  {benchmark_annual:+.2%}")
        print(f"  Alpha:                 {alpha_annual:+.2%}")
        
        print(f"\nRisk:")
        print(f"  Strategy Volatility:   {strategy_vol:.2%}")
        print(f"  Benchmark Volatility:  {benchmark_vol:.2%}")
        print(f"  Strategy Sharpe:       {strategy_sharpe:.3f}")
        print(f"  Benchmark Sharpe:      {benchmark_sharpe:.3f}")
        print(f"  Strategy Sortino:      {strategy_sortino:.3f}")
        print(f"  Strategy Max DD:       {strategy_max_dd:.2%}")
        print(f"  Benchmark Max DD:      {benchmark_max_dd:.2%}")
        print(f"  Calmar Ratio:          {strategy_calmar:.3f}")
        
        print(f"\nTrading:")
        print(f"  Win Rate:              {win_rate:.2%}")
        print(f"  Profit Factor:         {profit_factor:.3f}")
        print(f"  Number of Trades:      {num_trades}")
        print(f"  Avg Trades/Year:       {num_trades/n_years:.1f}")
        
        print(f"\nFinal Values:")
        print(f"  Strategy:              ${strategy_final:,.0f}")
        print(f"  Benchmark:             ${benchmark_final:,.0f}")
        print(f"  Difference:            ${strategy_final - benchmark_final:+,.0f}")
        
        print("="*60)
        
        return {
            'total_return_strategy': strategy_total,
            'total_return_benchmark': benchmark_total,
            'annual_return_strategy': strategy_annual,
            'annual_return_benchmark': benchmark_annual,
            'alpha': alpha_annual,
            'volatility_strategy': strategy_vol,
            'volatility_benchmark': benchmark_vol,
            'sharpe_strategy': strategy_sharpe,
            'sharpe_benchmark': benchmark_sharpe,
            'sortino_strategy': strategy_sortino,
            'max_drawdown_strategy': strategy_max_dd,
            'max_drawdown_benchmark': benchmark_max_dd,
            'calmar_ratio': strategy_calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'final_value_strategy': strategy_final,
            'final_value_benchmark': benchmark_final,
            'n_months': n_months,
            'n_years': n_years
        }
    
    def analyze_signal_effectiveness(self, backtest: pd.DataFrame) -> Dict:
        """
        Analyze how well the signal predicts returns.
        
        Parameters
        ----------
        backtest : pd.DataFrame
            Backtest results
            
        Returns
        -------
        Dict
            Signal analysis metrics
        """
        print("\n" + "="*60)
        print("SIGNAL EFFECTIVENESS ANALYSIS")
        print("="*60)
        
        # Correlation between signal and next return
        corr_signal_return = backtest[['signal', 'market_return']].corr().iloc[0, 1]
        
        # Correlation between predicted and actual returns
        corr_pred_actual = backtest[['predicted_return', 'market_return']].corr().iloc[0, 1]
        
        # Directional accuracy
        actual_direction = (backtest['market_return'] > 0).astype(int)
        pred_direction = (backtest['predicted_return'] > 0).astype(int)
        directional_accuracy = (actual_direction == pred_direction).mean()
        
        # Returns by signal quartile
        backtest['signal_quartile'] = pd.qcut(backtest['signal'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        returns_by_quartile = backtest.groupby('signal_quartile')['market_return'].mean()
        
        print(f"\nCorrelations:")
        print(f"  Signal vs Actual Return:     {corr_signal_return:.4f}")
        print(f"  Predicted vs Actual Return:  {corr_pred_actual:.4f}")
        
        print(f"\nDirectional Accuracy:")
        print(f"  Correct Direction:           {directional_accuracy:.2%}")
        
        print(f"\nAvg Return by Signal Quartile:")
        for q, ret in returns_by_quartile.items():
            print(f"  {q} (signal):  {ret:+.2%}")
        
        print("="*60)
        
        return {
            'corr_signal_return': corr_signal_return,
            'corr_pred_actual': corr_pred_actual,
            'directional_accuracy': directional_accuracy,
            'returns_by_quartile': returns_by_quartile.to_dict()
        }


def main():
    """
    Test trading strategy.
    """
    import yaml
    from data_acquisition import DividendPriceDataFetcher
    from feature_engineering import DPRatioFeatureEngineer
    from ols_model import DPRatioOLSModel
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get data
    fetcher = DividendPriceDataFetcher(config)
    data = fetcher.fetch_and_prepare_data()
    
    # Engineer features
    engineer = DPRatioFeatureEngineer(config)
    data = engineer.engineer_all_features(data)
    X, y = engineer.prepare_model_dataset(data)
    
    # Train model
    model = DPRatioOLSModel(config)
    X_in, X_out, y_in, y_out = model.split_data(X, y)
    model.train_ols_model(X_in, y_in)
    
    # Initialize strategy
    strategy = DPRatioTradingStrategy(config, model)
    
    # Test on out-of-sample period
    signals_out = strategy.generate_signals(X_out)
    backtest_results = strategy.backtest_strategy(signals_out, y_out)
    
    # Calculate metrics
    performance = strategy.calculate_performance_metrics(backtest_results)
    signal_analysis = strategy.analyze_signal_effectiveness(backtest_results)
    
    # Save results
    backtest_results.to_csv('results/backtest_results.csv')
    print("\nBacktest results saved to results/backtest_results.csv")


if __name__ == "__main__":
    main()
