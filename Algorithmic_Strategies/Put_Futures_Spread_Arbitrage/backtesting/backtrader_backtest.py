"""
Alternative Backtesting Implementation using Backtrader
Since Lean CLI has compatibility issues with Python 3.14, this uses backtrader for local backtesting
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import json

class PutCallParityStrategy(bt.Strategy):
    """Put-Call Parity Arbitrage Strategy"""

    params = (
        ('arbitrage_threshold', 0.01),
        ('max_position_size', 10),
        ('commission', 0.50),
    )

    def __init__(self):
        # Keep track of positions
        self.spx_price = self.datas[0].close
        self.es_price = self.datas[1].close

        # Risk-free rate (simplified)
        self.rf_rate = 0.05

        # Position tracking
        self.current_arbitrage_position = 0

    def next(self):
        # Get current prices
        spx = self.spx_price[0]
        es = self.es_price[0]

        # Calculate time to expiration (assume 30 days)
        T = 30/365

        # Find at-the-money strike (simplified - assume we have options data)
        # In practice, you'd need options data feed
        strike = spx  # ATM approximation

        # Simplified option prices (in practice, get from data)
        # This is a major limitation - we need real options data
        call_price = 5.0  # Placeholder
        put_price = 4.8   # Placeholder

        # Calculate theoretical futures price
        PV_K = strike * np.exp(-self.rf_rate * T)
        F_theoretical = call_price - put_price + PV_K

        # Check parity
        diff = es - F_theoretical
        parity_diff = abs(diff)

        # Execute arbitrage if threshold exceeded
        if parity_diff > self.params.arbitrage_threshold:
            if diff > 0:  # Futures overpriced
                self.execute_arbitrage_sell(es, F_theoretical)
            else:  # Futures underpriced
                self.execute_arbitrage_buy(es, F_theoretical)

    def execute_arbitrage_sell(self, es_price, f_theoretical):
        """Sell futures, buy synthetic long"""
        size = min(self.params.max_position_size, self.getposition(self.datas[1]).size)

        if size > 0:
            # Sell futures
            self.sell(data=self.datas[1], size=size)
            self.log(f'SELL ARBITRAGE: Sold {size} ES @ {es_price:.2f}')

    def execute_arbitrage_buy(self, es_price, f_theoretical):
        """Buy futures, sell synthetic long"""
        size = self.params.max_position_size

        # Buy futures
        self.buy(data=self.datas[1], size=size)
        self.log(f'BUY ARBITRAGE: Bought {size} ES @ {es_price:.2f}')

    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

class BacktraderBacktest:
    """Backtrader-based backtesting system"""

    def __init__(self, data_path="../../../data"):
        self.data_path = os.path.abspath(data_path)
        self.cerebro = bt.Cerebro()

    def load_data(self):
        """Load SPY and ES data by aggregating daily CSV files"""
        # Load SPY data
        spy_path = f"{self.data_path}/equity/usa/daily"
        spy_df = self.aggregate_daily_data(spy_path, 'SPY')
        if not spy_df.empty:
            spy_data = bt.feeds.PandasData(dataname=spy_df, name='SPY')
            self.cerebro.adddata(spy_data)

        # Load ES data
        es_path = f"{self.data_path}/future/cme/daily/es"
        es_df = self.aggregate_daily_data(es_path, 'ES')
        if not es_df.empty:
            es_data = bt.feeds.PandasData(dataname=es_df, name='ES')
            self.cerebro.adddata(es_data)

    def aggregate_daily_data(self, path, symbol):
        """Aggregate daily CSV files into a single DataFrame"""
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            return pd.DataFrame()

        all_data = []
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

        for csv_file in sorted(csv_files):
            try:
                # Extract date from filename (YYYYMMDD_symbol_daily.csv)
                date_str = csv_file[:8]
                date = datetime.strptime(date_str, '%Y%m%d')

                # Read the CSV
                df = pd.read_csv(f"{path}/{csv_file}", header=None,
                               names=['open', 'high', 'low', 'close', 'volume'])

                if df.empty:
                    continue

                # Convert prices back from Lean format (deci-cents to dollars)
                if symbol == 'SPY':
                    pass  # Prices are already in dollars
                    # df['open'] = df['open'] / 10000
                    # df['high'] = df['high'] / 10000
                    # df['low'] = df['low'] / 10000
                    # df['close'] = df['close'] / 10000

                # For futures, prices are already in dollars
                # Add datetime index
                df['datetime'] = date
                df.set_index('datetime', inplace=True)

                all_data.append(df)

            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue

        if all_data:
            combined_df = pd.concat(all_data).sort_index()
            # Ensure correct data types
            combined_df = combined_df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': int})
            print(f"Loaded {len(combined_df)} days of {symbol} data")
            return combined_df

        print(f"No data loaded for {symbol}")
        return pd.DataFrame()

    def load_csv_data(self, filepath):
        """Load CSV data into pandas DataFrame"""
        df = pd.read_csv(filepath, header=None,
                        names=['open', 'high', 'low', 'close', 'volume'])

        # Convert prices back from Lean format
        df['open'] = df['open'] / 10000
        df['high'] = df['high'] / 10000
        df['low'] = df['low'] / 10000
        df['close'] = df['close'] / 10000

        # Create date index
        date_str = os.path.basename(filepath)[:8]
        start_date = datetime.strptime(date_str, '%Y%m%d')
        df['datetime'] = pd.date_range(start=start_date, periods=len(df), freq='D')
        df.set_index('datetime', inplace=True)

        return df

    def setup_strategy(self):
        """Setup the arbitrage strategy"""
        self.cerebro.addstrategy(PutCallParityStrategy,
                                arbitrage_threshold=0.01,
                                max_position_size=10,
                                commission=0.50)

        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    def run_backtest(self):
        """Run the backtest"""
        print("Running backtrader backtest...")

        # Set initial cash
        self.cerebro.broker.setcash(1000000.0)

        # Set commission
        self.cerebro.broker.setcommission(commission=0.50)

        # Run backtest
        results = self.cerebro.run()

        # Get results
        strat = results[0]

        # Print results
        print("\n=== Backtest Results ===")
        print(f"Starting Portfolio Value: ${1000000:.2f}")
        print(f"Final Portfolio Value: ${self.cerebro.broker.getvalue():.2f}")
        print(f"Total Return: ${(self.cerebro.broker.getvalue() - 1000000):.2f}")

        # Sharpe Ratio
        sharpe = strat.analyzers.sharpe.get_analysis()
        if 'sharperatio' in sharpe:
            print(f"Sharpe Ratio: {sharpe['sharperatio']:.2f}")

        # Max Drawdown
        drawdown = strat.analyzers.drawdown.get_analysis()
        if 'max' in drawdown:
            print(f"Max Drawdown: {drawdown['max']['drawdown']:.2f}%")

        # Trade Analysis
        trade_analysis = strat.analyzers.trades.get_analysis()
        if trade_analysis:
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            won_trades = trade_analysis.get('won', {}).get('total', 0)
            lost_trades = trade_analysis.get('lost', {}).get('total', 0)

            print(f"Total Trades: {total_trades}")
            print(f"Won Trades: {won_trades}")
            print(f"Lost Trades: {lost_trades}")
            if total_trades > 0:
                win_rate = won_trades / total_trades
                print(f"Win Rate: {win_rate:.2%}")

        # Save results to JSON for analyzer
        results_data = {
            'final_value': self.cerebro.broker.getvalue(),
            'total_return': self.cerebro.broker.getvalue() - 1000000,
            'sharpe_ratio': sharpe.get('sharperatio', None) if 'sharperatio' in sharpe else None,
            'max_drawdown': drawdown['max']['drawdown'] if 'max' in drawdown else None,
            'total_trades': trade_analysis.get('total', {}).get('total', 0) if trade_analysis else 0,
            'won_trades': trade_analysis.get('won', {}).get('total', 0) if trade_analysis else 0,
            'lost_trades': trade_analysis.get('lost', {}).get('total', 0) if trade_analysis else 0,
        }
        
        with open('backtest-results.json', 'w') as f:
            json.dump(results_data, f, indent=2)

        # Plot results
        self.plot_results()

        return results

    def plot_results(self):
        """Plot equity curve"""
        plt.figure(figsize=(12, 8))

        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.cerebro.broker.getvalue())
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Portfolio Value ($)')

        # Plot drawdown
        plt.subplot(2, 1, 2)
        # Drawdown plotting would require more complex implementation

        plt.tight_layout()
        plt.savefig('backtrader_results.png')
        plt.show()

def main():
    """Main backtesting function"""
    backtest = BacktraderBacktest()

    try:
        backtest.load_data()
        backtest.setup_strategy()
        results = backtest.run_backtest()

        print("\nBacktest completed successfully!")
        print("Note: This is a simplified implementation.")
        print("Real options data integration would be needed for accurate results.")

    except Exception as e:
        print(f"Backtest failed: {e}")
        print("Make sure data files exist and backtrader is installed:")
        print("pip install backtrader")

if __name__ == "__main__":
    main()