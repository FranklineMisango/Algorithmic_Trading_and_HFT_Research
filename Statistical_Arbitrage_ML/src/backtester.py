"""
Backtesting Engine for Statistical Arbitrage Strategy

Simulates historical performance with realistic transaction costs and metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns


class Backtester:
    """
    Backtests the statistical arbitrage strategy.
    
    Key features:
    - Simulates portfolio returns over time
    - Accounts for transaction costs
    - Calculates performance metrics
    - Factor analysis (Fama-French)
    - Visualization
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        transaction_cost_bps: float = 10.0,  # 10 basis points
        holding_period: int = 3,  # days
        market_benchmark: str = 'SPY'  # S&P 500 proxy
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_cost_bps: Transaction cost in basis points (0.01%)
            holding_period: Days to hold positions
            market_benchmark: Ticker for market benchmark
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.transaction_cost_pct = transaction_cost_bps / 10000.0
        self.holding_period = holding_period
        self.market_benchmark = market_benchmark
        
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
        
        logger.info(
            f"Backtester initialized: ${initial_capital:,.0f} capital, "
            f"{transaction_cost_bps} bps costs, "
            f"{holding_period} day holding period"
        )
    
    def calculate_trade_pnl(
        self,
        portfolio: pd.DataFrame,
        entry_prices: pd.Series,
        exit_prices: pd.Series,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Calculate P&L for a portfolio over holding period.
        
        Args:
            portfolio: Portfolio DataFrame with positions
            entry_prices: Prices at entry
            exit_prices: Prices at exit
            entry_date: Entry date
            exit_date: Exit date
            
        Returns:
            DataFrame with trade results
        """
        trades = []
        
        for _, position in portfolio.iterrows():
            ticker = position['ticker']
            shares = position['shares']
            side = position['side']
            
            if ticker not in entry_prices.index or ticker not in exit_prices.index:
                logger.warning(f"Missing prices for {ticker}, skipping")
                continue
            
            entry_price = entry_prices[ticker]
            exit_price = exit_prices[ticker]
            
            # Calculate return
            if side == 'long':
                trade_return = (exit_price - entry_price) / entry_price
            else:  # short
                trade_return = (entry_price - exit_price) / entry_price
            
            # Calculate P&L
            position_value = abs(shares) * entry_price
            gross_pnl = position_value * trade_return
            
            # Transaction costs (entry + exit)
            transaction_costs = position_value * self.transaction_cost_pct * 2
            net_pnl = gross_pnl - transaction_costs
            
            trades.append({
                'ticker': ticker,
                'side': side,
                'shares': shares,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_value': position_value,
                'trade_return': trade_return,
                'gross_pnl': gross_pnl,
                'transaction_costs': transaction_costs,
                'net_pnl': net_pnl
            })
        
        return pd.DataFrame(trades)
    
    def run_backtest(
        self,
        portfolios: List[pd.DataFrame],
        price_data: pd.DataFrame
    ) -> Dict:
        """
        Run backtest on series of portfolios.
        
        Args:
            portfolios: List of portfolio DataFrames over time
            price_data: Historical price data (date, ticker) multi-index
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest with {len(portfolios)} portfolio iterations...")
        
        all_trades = []
        equity_curve = [self.initial_capital]
        dates = [portfolios[0]['date'].iloc[0] - timedelta(days=1)]  # Start date
        
        for portfolio in portfolios:
            entry_date = portfolio['date'].iloc[0]
            exit_date = entry_date + timedelta(days=self.holding_period)
            
            # Get entry and exit prices
            try:
                entry_prices = price_data.loc[entry_date]['Close']
                exit_prices = price_data.loc[exit_date]['Close']
            except KeyError:
                logger.warning(f"Missing price data for {entry_date} or {exit_date}")
                continue
            
            # Calculate trade P&L
            trades = self.calculate_trade_pnl(
                portfolio, entry_prices, exit_prices, entry_date, exit_date
            )
            
            if trades.empty:
                continue
            
            all_trades.append(trades)
            
            # Update equity
            period_pnl = trades['net_pnl'].sum()
            new_equity = equity_curve[-1] + period_pnl
            
            equity_curve.append(new_equity)
            dates.append(exit_date)
            
            logger.debug(
                f"{entry_date.strftime('%Y-%m-%d')}: "
                f"P&L=${period_pnl:,.2f}, Equity=${new_equity:,.2f}"
            )
        
        # Combine all trades
        if all_trades:
            all_trades_df = pd.concat(all_trades, ignore_index=True)
        else:
            all_trades_df = pd.DataFrame()
        
        # Create equity curve dataframe
        equity_df = pd.DataFrame({
            'date': dates,
            'equity': equity_curve
        })
        equity_df = equity_df.set_index('date')
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(equity_df, all_trades_df)
        
        results = {
            'trades': all_trades_df,
            'equity_curve': equity_df,
            'metrics': metrics
        }
        
        logger.info("Backtest complete")
        
        return results
    
    def calculate_performance_metrics(
        self,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_df: Equity curve with returns
            trades_df: All trades executed
            
        Returns:
            Dictionary of performance metrics
        """
        returns = equity_df['returns'].dropna()
        
        if len(returns) == 0:
            logger.warning("No returns to calculate metrics")
            return {}
        
        # Basic return metrics
        total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1
        n_years = len(returns) / 252  # Approximate trading days per year
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Risk metrics
        returns_std = returns.std()
        sharpe_ratio = (returns.mean() / returns_std) * np.sqrt(252) if returns_std > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['net_pnl'] > 0]
            losing_trades = trades_df[trades_df['net_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df)
            avg_win = winning_trades['net_pnl'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['net_pnl'].mean() if not losing_trades.empty else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            total_trades = len(trades_df)
            total_pnl = trades_df['net_pnl'].sum()
            avg_trade_pnl = trades_df['net_pnl'].mean()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            total_trades = 0
            total_pnl = 0
            avg_trade_pnl = 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': returns_std * np.sqrt(252),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_trade_pnl': avg_trade_pnl
        }
        
        logger.info(
            f"Performance metrics calculated: "
            f"Annual Return: {annual_return*100:.2f}%, "
            f"Sharpe: {sharpe_ratio:.2f}, "
            f"Max DD: {max_drawdown*100:.2f}%"
        )
        
        return metrics
    
    def calculate_market_correlation(
        self,
        strategy_returns: pd.Series,
        market_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate correlation with market benchmark.
        
        Args:
            strategy_returns: Strategy returns
            market_returns: Market benchmark returns
            
        Returns:
            Dictionary with correlation metrics
        """
        # Align returns
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned) < 2:
            logger.warning("Insufficient data for correlation calculation")
            return {}
        
        correlation = aligned['strategy'].corr(aligned['market'])
        beta = aligned['strategy'].cov(aligned['market']) / aligned['market'].var()
        
        return {
            'market_correlation': correlation,
            'market_beta': beta
        }
    
    def fama_french_analysis(
        self,
        strategy_returns: pd.Series,
        ff_factors: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Perform Fama-French 3-factor analysis.
        
        Args:
            strategy_returns: Strategy returns
            ff_factors: DataFrame with Fama-French factors (Mkt-RF, SMB, HML, RF)
                       If None, simplified analysis is performed
            
        Returns:
            Dictionary with factor loadings and alpha
        """
        if ff_factors is None:
            logger.warning("Fama-French factors not provided, skipping factor analysis")
            return {}
        
        # Merge returns with factors
        merged = pd.merge(
            strategy_returns.to_frame('returns'),
            ff_factors,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        if len(merged) < 10:
            logger.warning("Insufficient data for factor analysis")
            return {}
        
        # Calculate excess returns
        merged['excess_returns'] = merged['returns'] - merged['RF']
        
        # Regression: excess_returns = alpha + beta_mkt * Mkt-RF + beta_smb * SMB + beta_hml * HML
        from sklearn.linear_model import LinearRegression
        
        X = merged[['Mkt-RF', 'SMB', 'HML']].values
        y = merged['excess_returns'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        alpha = model.intercept_ * 252  # Annualized
        betas = model.coef_
        r_squared = model.score(X, y)
        
        results = {
            'alpha': alpha,
            'beta_market': betas[0],
            'beta_size': betas[1],
            'beta_value': betas[2],
            'r_squared': r_squared
        }
        
        logger.info(
            f"Fama-French analysis: "
            f"Alpha={alpha*100:.2f}% (annualized), "
            f"R²={r_squared:.4f}"
        )
        
        return results
    
    def plot_equity_curve(
        self,
        equity_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot equity curve over time.
        
        Args:
            equity_df: Equity curve DataFrame
            save_path: Path to save plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curve
        ax1.plot(equity_df.index, equity_df['equity'], linewidth=2)
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.legend()
        
        # Drawdown
        returns = equity_df['returns'].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_returns_distribution(
        self,
        returns: pd.Series,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of returns.
        
        Args:
            returns: Series of returns
            save_path: Path to save plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(x=returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax1.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Returns', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(
        self,
        results: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive backtest report.
        
        Args:
            results: Backtest results dictionary
            output_path: Path to save report (optional)
            
        Returns:
            Report as string
        """
        metrics = results['metrics']
        
        report = f"""
{'='*80}
STATISTICAL ARBITRAGE BACKTEST REPORT
{'='*80}

PERFORMANCE SUMMARY
{'─'*80}
Total Return:              {metrics.get('total_return', 0)*100:>10.2f}%
Annual Return:             {metrics.get('annual_return', 0)*100:>10.2f}%
Sharpe Ratio:              {metrics.get('sharpe_ratio', 0):>10.2f}
Maximum Drawdown:          {metrics.get('max_drawdown', 0)*100:>10.2f}%
Volatility (Annual):       {metrics.get('volatility', 0)*100:>10.2f}%

TRADE STATISTICS
{'─'*80}
Total Trades:              {metrics.get('total_trades', 0):>10,.0f}
Win Rate:                  {metrics.get('win_rate', 0)*100:>10.2f}%
Average Win:               ${metrics.get('avg_win', 0):>10,.2f}
Average Loss:              ${metrics.get('avg_loss', 0):>10,.2f}
Profit Factor:             {metrics.get('profit_factor', 0):>10.2f}
Total P&L:                 ${metrics.get('total_pnl', 0):>10,.2f}
Average Trade P&L:         ${metrics.get('avg_trade_pnl', 0):>10,.2f}

RISK METRICS
{'─'*80}
Initial Capital:           ${self.initial_capital:>10,.2f}
Final Equity:              ${results['equity_curve']['equity'].iloc[-1]:>10,.2f}
Transaction Cost (bps):    {self.transaction_cost_bps:>10.2f}

{'='*80}
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Create synthetic portfolios
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='3D')
    portfolios = []
    
    for date in dates:
        portfolio = pd.DataFrame({
            'ticker': [f'STOCK{i}' for i in range(20)],
            'side': ['long'] * 10 + ['short'] * 10,
            'shares': np.random.randint(10, 100, 20),
            'date': date
        })
        portfolios.append(portfolio)
    
    # Create synthetic price data
    all_tickers = [f'STOCK{i}' for i in range(20)]
    price_dates = pd.date_range('2022-12-15', '2024-01-15', freq='D')
    
    price_data = []
    for ticker in all_tickers:
        base_price = np.random.uniform(50, 150)
        prices = base_price * (1 + np.cumsum(np.random.randn(len(price_dates)) * 0.02))
        
        for date, price in zip(price_dates, prices):
            price_data.append({
                'Date': date,
                'ticker': ticker,
                'Close': price
            })
    
    price_df = pd.DataFrame(price_data)
    price_df = price_df.set_index(['Date', 'ticker'])
    
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest(portfolios, price_df)
    
    # Generate report
    report = backtester.generate_report(results)
    print(report)
    
    # Plot results
    # backtester.plot_equity_curve(results['equity_curve'])
