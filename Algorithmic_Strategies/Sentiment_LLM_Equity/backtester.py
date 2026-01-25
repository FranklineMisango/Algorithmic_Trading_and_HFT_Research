"""
Backtester for Sentiment-Based Equity Strategy

Includes transaction costs, market impact, and performance metrics.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class SentimentBacktester:
    """Backtest sentiment-based long-short strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize backtester."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.backtest_config = self.config['backtest']
        self.execution_config = self.config['execution']
        self.evaluation_config = self.config['evaluation']
    
    def calculate_transaction_costs(
        self,
        position_changes: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate transaction costs.
        
        Args:
            position_changes: Position changes (turnover)
            market_data: Market data for market cap classification
        
        Returns:
            Transaction costs per trade
        """
        # Merge to get market cap
        merged = position_changes.merge(
            market_data[['ticker', 'date', 'market_cap']],
            on=['ticker', 'date'],
            how='left'
        )
        
        # Classify stocks (large-cap vs small-cap)
        # Simplified: use median market cap as threshold
        median_mcap = merged['market_cap'].median()
        is_large_cap = merged['market_cap'] >= median_mcap
        
        # Get cost parameters
        large_cap_cost = self.execution_config['transaction_costs']['large_cap']['total_bps'] / 10000
        small_cap_cost = self.execution_config['transaction_costs']['small_cap']['total_bps'] / 10000
        
        # Calculate costs
        costs = np.where(is_large_cap, large_cap_cost, small_cap_cost)
        transaction_costs = merged['turnover'].abs() * costs
        
        return transaction_costs
    
    def run_backtest(
        self,
        portfolio: pd.DataFrame,
        market_data: pd.DataFrame,
        initial_capital: float = None
    ) -> Dict:
        """
        Run vectorized backtest.
        
        Args:
            portfolio: Portfolio with weights
            market_data: Market data with returns
            initial_capital: Starting capital
        
        Returns:
            Backtest results dict
        """
        if initial_capital is None:
            initial_capital = self.backtest_config['initial_capital']
        
        # Merge portfolio with returns
        backtest_data = portfolio.merge(
            market_data[['ticker', 'date', 'return']],
            on=['ticker', 'date'],
            how='inner'
        )
        
        # Calculate portfolio returns (weight * return)
        backtest_data['contribution'] = backtest_data['weight'] * backtest_data['return']
        
        # Aggregate by date
        daily_returns = backtest_data.groupby('date')['contribution'].sum().reset_index()
        daily_returns.rename(columns={'contribution': 'gross_return'}, inplace=True)
        
        # Calculate turnover
        portfolio_sorted = portfolio.sort_values(['ticker', 'date'])
        portfolio_sorted['prev_weight'] = portfolio_sorted.groupby('ticker')['weight'].shift(1)
        portfolio_sorted['turnover'] = (portfolio_sorted['weight'] - portfolio_sorted['prev_weight'].fillna(0)).abs()
        
        daily_turnover = portfolio_sorted.groupby('date')['turnover'].sum().reset_index()
        
        # Merge
        daily_returns = daily_returns.merge(daily_turnover, on='date', how='left')
        daily_returns['turnover'] = daily_returns['turnover'].fillna(0)
        
        # Calculate transaction costs (simplified)
        cost_bps = self.execution_config['backtest_costs'][self.execution_config['backtest_costs']['default']]
        daily_returns['transaction_cost'] = daily_returns['turnover'] * (cost_bps / 10000)
        
        # Net returns
        daily_returns['net_return'] = daily_returns['gross_return'] - daily_returns['transaction_cost']
        
        # Portfolio value
        daily_returns['portfolio_value'] = initial_capital * (1 + daily_returns['net_return']).cumprod()
        
        # Drawdown
        cummax = daily_returns['portfolio_value'].cummax()
        daily_returns['drawdown'] = (cummax - daily_returns['portfolio_value']) / cummax
        
        # Calculate metrics
        metrics = self.calculate_metrics(daily_returns, initial_capital)
        
        return {
            'daily_returns': daily_returns,
            'metrics': metrics
        }
    
    def calculate_metrics(self, daily_returns: pd.DataFrame, initial_capital: float) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            daily_returns: Daily returns DataFrame
            initial_capital: Initial capital
        
        Returns:
            Metrics dict
        """
        portfolio_values = daily_returns['portfolio_value'].values
        returns = daily_returns['net_return'].values
        
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
        max_drawdown = daily_returns['drawdown'].max()
        
        # Calmar ratio
        calmar = annualized_return / (max_drawdown + 1e-8)
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        # Turnover
        avg_turnover = daily_returns['turnover'].mean()
        annualized_turnover = avg_turnover * 252
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'annualized_turnover': annualized_turnover,
            'n_days': n_days
        }
    
    def factor_regression(self, strategy_returns: pd.Series, factor_returns: pd.DataFrame) -> Dict:
        """
        Perform Fama-French factor regression.
        
        Args:
            strategy_returns: Strategy returns
            factor_returns: Factor returns (Mkt-RF, SMB, HML, etc.)
        
        Returns:
            Regression results
        """
        # Align data
        aligned = pd.DataFrame({
            'strategy': strategy_returns
        }).join(factor_returns, how='inner')
        
        # Regression
        from sklearn.linear_model import LinearRegression
        
        X = aligned[factor_returns.columns].values
        y = aligned['strategy'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Alpha (intercept)
        alpha = model.intercept_
        
        # Betas (coefficients)
        betas = dict(zip(factor_returns.columns, model.coef_))
        
        # R-squared
        r_squared = model.score(X, y)
        
        # T-stat for alpha
        residuals = y - model.predict(X)
        se_alpha = np.std(residuals) / np.sqrt(len(y))
        t_stat = alpha / se_alpha
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), len(y) - len(factor_returns.columns) - 1))
        
        return {
            'alpha': alpha,
            'alpha_annualized': alpha * 252,
            'betas': betas,
            'r_squared': r_squared,
            't_stat': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }


# Test code
if __name__ == "__main__":
    from data_acquisition import SentimentDataAcquisition
    from portfolio_construction import SentimentPortfolioConstructor
    
    # Load data
    data_acq = SentimentDataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset("2023-01-01", "2023-03-31")
    
    # Generate placeholder sentiment scores
    sentiment_scores = dataset['text_data'].copy()
    sentiment_scores['sentiment'] = np.random.randn(len(sentiment_scores))
    
    # Construct portfolio
    constructor = SentimentPortfolioConstructor('config.yaml')
    portfolio = constructor.construct_portfolio(sentiment_scores, dataset['market_data'])
    
    # Backtest
    backtester = SentimentBacktester('config.yaml')
    results = backtester.run_backtest(portfolio, dataset['market_data'])
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    for metric, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
