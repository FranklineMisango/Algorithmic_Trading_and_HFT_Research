"""
Performance Analysis for Put-Futures Arbitrage Backtest
Analyzes backtest results and generates reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

class BacktestAnalyzer:
    """Analyze Lean backtest results"""

    def __init__(self, results_path="backtest-results.json"):
        self.results_path = results_path
        self.results = None

    def load_results(self):
        """Load backtest results from Lean output"""
        if os.path.exists(self.results_path):
            with open(self.results_path, 'r') as f:
                self.results = json.load(f)
        else:
            print(f"Results file not found: {self.results_path}")
            return False
        return True

    def analyze_performance(self):
        """Analyze key performance metrics"""
        if not self.results:
            return

        # Extract equity curve
        equity = pd.DataFrame(self.results.get('equity', {}))
        if equity.empty:
            print("No equity data found")
            return

        # Calculate returns
        equity['returns'] = equity['value'].pct_change()

        # Performance metrics
        total_return = (equity['value'].iloc[-1] / equity['value'].iloc[0]) - 1
        sharpe_ratio = self.calculate_sharpe_ratio(equity['returns'])
        max_drawdown = self.calculate_max_drawdown(equity['value'])
        win_rate = (equity['returns'] > 0).mean()

        print("=== Backtest Performance Summary ===")
        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Win Rate: {win_rate:.2%}")

        # Plot equity curve
        self.plot_equity_curve(equity)

        # Analyze trades
        self.analyze_trades()

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.05):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_max_drawdown(self, equity):
        """Calculate maximum drawdown"""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return drawdown.min()

    def analyze_trades(self):
        """Analyze individual trades"""
        if not self.results:
            return

        orders = self.results.get('orders', [])
        if not orders:
            print("No trades found")
            return

        trades_df = pd.DataFrame(orders)

        # Calculate trade P&L
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        total_trades = len(trades_df)

        print("=== Trade Analysis ===")
        print(f"Total Trades: {total_trades}")
        print(f"Profitable Trades: {profitable_trades}")
        print(f"Trade Win Rate: {profitable_trades/total_trades:.2%}")

        if 'pnl' in trades_df.columns:
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
            print(f"Average Win: ${avg_win:.2f}")
            print(f"Average Loss: ${avg_loss:.2f}")

    def plot_equity_curve(self, equity):
        """Plot equity curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(equity.index, equity['value'])
        plt.title('Equity Curve - Put-Futures Arbitrage')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.savefig('equity_curve.png')
        plt.show()

    def generate_report(self):
        """Generate comprehensive report"""
        if not self.load_results():
            return

        print("Generating backtest report...")

        # Create report directory
        os.makedirs('reports', exist_ok=True)

        # Generate analysis
        self.analyze_performance()

        # Save metrics to file
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_return': self.calculate_total_return(),
            'sharpe_ratio': self.calculate_sharpe_ratio(pd.Series()),
            'max_drawdown': self.calculate_max_drawdown(pd.Series()),
            'total_trades': len(self.results.get('orders', []))
        }

        with open('reports/backtest_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print("Report saved to reports/ directory")

    def calculate_total_return(self):
        """Calculate total return"""
        if not self.results:
            return 0
        equity = pd.DataFrame(self.results.get('equity', {}))
        if equity.empty:
            return 0
        return (equity['value'].iloc[-1] / equity['value'].iloc[0]) - 1

if __name__ == "__main__":
    analyzer = BacktestAnalyzer()
    analyzer.generate_report()