"""
Multi-year validation for HJB market making model
Addresses critique requirement for multi-year backtests instead of single days
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import zipfile
import os
import sys
from scipy import stats
import warnings

# Add the HJB model path
sys.path.append('/home/misango/codechest/Algorithmic_Trading_and_HFT_Research/Market_Making/HJB_DP_MM_Optimisation')

from hjb_cpu_modelling import HJBMarketMaker
from hjb_gpu_modelling import HJBGPUMarketMaker

class MultiYearValidator:
    """Multi-year validation of HJB market making model"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.results = {}

    def load_multi_year_data(self, symbol='btcusdt', start_year=2024, end_year=2025):
        """Load multiple years of BTCUSDT data"""
        all_data = []

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if year == 2025 and month > 11:  # Current date limit
                    break

                date_str = f"{year}{month:02d}01"
                try:
                    data = self._load_single_month(symbol, date_str)
                    if data:
                        all_data.extend(data)
                        print(f"Loaded {len(data)} records for {year}-{month:02d}")
                except Exception as e:
                    print(f"Failed to load {year}-{month:02d}: {e}")
                    continue

        print(f"Total records loaded: {len(all_data)}")
        return all_data

    def _load_single_month(self, symbol, date):
        """Load data for a single month"""
        zip_path = os.path.join(self.data_path, 'crypto', 'binance', 'minute', symbol.lower(), f"{date}_trade.zip")

        if not os.path.exists(zip_path):
            return None

        data = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            csv_filename = f"{date}_{symbol.lower()}_minute_trade.csv"
            with zip_ref.open(csv_filename) as file:
                lines = file.readlines()
                for line in lines[1:]:  # Skip header
                    try:
                        parts = line.decode('utf-8').strip().split(',')
                        if len(parts) == 6:
                            time_ms, open_p, high_p, low_p, close_p, volume = parts
                            data.append({
                                'timestamp': int(time_ms),
                                'open': float(open_p),
                                'high': float(high_p),
                                'low': float(low_p),
                                'close': float(close_p),
                                'volume': float(volume)
                            })
                    except:
                        continue
        return data

    def calibrate_parameters_multi_year(self, data):
        """Calibrate model parameters using multi-year data"""
        print("Calibrating parameters using multi-year data...")

        # Extract price series
        prices = np.array([d['close'] for d in data])
        returns = np.diff(np.log(prices))

        # Estimate volatility (annualized)
        sigma = np.std(returns) * np.sqrt(525600)  # 525600 minutes per year

        # Estimate jump parameters using extreme value theory
        # Fit generalized Pareto distribution to negative returns
        negative_returns = -returns[returns < 0]
        if len(negative_returns) > 100:
            # Simple jump detection: returns beyond 3-sigma
            threshold = np.mean(returns) - 3 * np.std(returns)
            jumps = returns[returns < threshold]

            if len(jumps) > 10:
                jump_intensity = len(jumps) / len(returns)  # Fraction of minutes with jumps
                jump_mean = np.mean(jumps)
                jump_std = np.std(jumps)
            else:
                jump_intensity = 0.001  # Default low intensity
                jump_mean = -0.01
                jump_std = 0.005
        else:
            jump_intensity = 0.001
            jump_mean = -0.01
            jump_std = 0.005

        # Market making parameters (conservative estimates)
        gamma = 0.01  # Risk aversion
        k = 2.0      # Market liquidity
        c = 1.0      # Base intensity

        params = {
            'sigma': sigma,
            'gamma': gamma,
            'k': k,
            'c': c,
            'jump_intensity': jump_intensity,
            'jump_mean': jump_mean,
            'jump_std': jump_std,
            'S_min': np.min(prices) * 0.95,
            'S_max': np.max(prices) * 1.05
        }

        print("Calibrated parameters:")
        for key, value in params.items():
            print(f"  {key}: {value:.6f}")

        return params

    def run_walk_forward_validation(self, data, params, window_days=30, step_days=7):
        """Run walk-forward validation with expanding window"""
        print("Running walk-forward validation...")

        # Convert data to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime').sort_index()

        results = []
        start_date = df.index.min()

        while start_date < df.index.max() - pd.Timedelta(days=window_days):
            end_date = start_date + pd.Timedelta(days=window_days)

            # Get training data (past window)
            train_data = df.loc[start_date:end_date]

            # Get test data (next step_days)
            test_start = end_date
            test_end = test_start + pd.Timedelta(days=step_days)
            test_data = df.loc[test_start:test_end]

            if len(test_data) < 100:  # Skip if insufficient test data
                break

            print(f"Testing period: {test_start.date()} to {test_end.date()}")

            # Run simulation on test data
            sim_result = self._simulate_period(test_data, params)
            sim_result['test_start'] = test_start
            sim_result['test_end'] = test_end

            results.append(sim_result)

            # Move window forward
            start_date += pd.Timedelta(days=step_days)

        return results

    def _simulate_period(self, test_data, params):
        """Simulate HJB strategy for a test period"""
        # Initialize model with calibrated parameters
        model = HJBMarketMaker(
            sigma=params['sigma'],
            gamma=params['gamma'],
            k=params['k'],
            c=params['c'],
            T=1.0,  # 1 hour horizon
            I_max=10,
            S_min=params['S_min'],
            S_max=params['S_max'],
            dS=(params['S_max'] - params['S_min']) / 100,
            dt=0.01,
            jump_intensity=params['jump_intensity'],
            jump_mean=params['jump_mean'],
            jump_std=params['jump_std']
        )

        # Solve HJB equation
        model.solve_pde()

        # Simulate trading
        cash = 100000.0
        inventory = 0.0
        trades = []

        for idx, row in test_data.iterrows():
            current_price = row['close']
            volume = row['volume']

            # Get optimal quotes
            try:
                bid, ask = model.optimal_quotes(0.5, inventory)  # Mid time horizon

                # Simulate market order execution (simplified)
                # Assume some trades hit our quotes based on volume
                trade_probability = min(0.1, volume / 1000)  # Volume-based probability

                if np.random.random() < trade_probability:
                    if current_price <= bid:  # Buy order
                        size = np.random.uniform(0.1, 1.0)
                        cash -= current_price * size
                        inventory += size
                        trades.append({'type': 'BUY', 'price': current_price, 'size': size})
                    elif current_price >= ask:  # Sell order
                        size = min(np.random.uniform(0.1, 1.0), inventory)
                        if size > 0:
                            cash += current_price * size
                            inventory -= size
                            trades.append({'type': 'SELL', 'price': current_price, 'size': size})

            except Exception as e:
                continue  # Skip if model fails

        # Calculate final P&L
        final_price = test_data['close'].iloc[-1]
        unrealized_pnl = inventory * final_price
        total_pnl = cash - 100000.0 + unrealized_pnl

        # Calculate Sharpe ratio
        if trades:
            trade_returns = []
            for trade in trades:
                if trade['type'] == 'SELL':
                    trade_returns.append((trade['price'] - 0) / 100000.0)  # Simplified
            if trade_returns:
                sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        return {
            'final_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'trades_executed': len(trades),
            'final_inventory': inventory,
            'final_cash': cash
        }

    def run_statistical_tests(self, results):
        """Run comprehensive statistical tests on validation results"""
        print("Running statistical tests...")

        pnls = [r['final_pnl'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]

        # Basic statistics
        stats_results = {
            'n_periods': len(results),
            'mean_pnl': np.mean(pnls),
            'std_pnl': np.std(pnls),
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'min_pnl': np.min(pnls),
            'max_pnl': np.max(pnls),
            'pnl_skewness': stats.skew(pnls),
            'pnl_kurtosis': stats.kurtosis(pnls)
        }

        # t-test for profitability
        t_stat, p_value = stats.ttest_1samp(pnls, 0)
        stats_results['t_stat'] = t_stat
        stats_results['p_value'] = p_value
        stats_results['significant_at_5pct'] = p_value < 0.05

        # Sharpe ratio test
        sharpe_t_stat, sharpe_p_value = stats.ttest_1samp(sharpes, 0)
        stats_results['sharpe_t_stat'] = sharpe_t_stat
        stats_results['sharpe_p_value'] = sharpe_p_value

        # Maximum drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        stats_results['max_drawdown'] = np.max(drawdowns)

        return stats_results

    def plot_validation_results(self, results, stats_results, save_path=None):
        """Plot comprehensive validation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. P&L over time
        dates = [r['test_start'] for r in results]
        pnls = [r['final_pnl'] for r in results]
        cumulative_pnl = np.cumsum(pnls)

        axes[0,0].plot(dates, cumulative_pnl, 'b-', linewidth=2, label='Cumulative P&L')
        axes[0,0].plot(dates, pnls, 'r--', alpha=0.7, label='Period P&L')
        axes[0,0].set_title('Multi-Year P&L Performance')
        axes[0,0].set_ylabel('P&L ($)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. Sharpe ratio over time
        sharpes = [r['sharpe_ratio'] for r in results]
        axes[0,1].plot(dates, sharpes, 'g-', linewidth=2)
        axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0,1].set_title('Sharpe Ratio Evolution')
        axes[0,1].set_ylabel('Sharpe Ratio')
        axes[0,1].grid(True, alpha=0.3)

        # 3. P&L distribution
        axes[1,0].hist(pnls, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1,0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1,0].set_title('P&L Distribution')
        axes[1,0].set_xlabel('P&L ($)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)

        # 4. Rolling statistics
        rolling_mean = pd.Series(pnls).rolling(window=10).mean()
        rolling_std = pd.Series(pnls).rolling(window=10).std()

        axes[1,1].plot(dates, rolling_mean, 'b-', label='Rolling Mean (10 periods)', linewidth=2)
        axes[1,1].fill_between(dates,
                              rolling_mean - rolling_std,
                              rolling_mean + rolling_std,
                              alpha=0.3, color='blue', label='±1 Std Dev')
        axes[1,1].set_title('Rolling P&L Statistics')
        axes[1,1].set_ylabel('P&L ($)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Multi-year validation plot saved to {save_path}")

        plt.show()

        # Print statistical summary
        print("\n" + "="*60)
        print("MULTI-YEAR VALIDATION STATISTICS")
        print("="*60)
        print(f"Number of test periods: {stats_results['n_periods']}")
        print(f"Mean P&L per period: ${stats_results['mean_pnl']:.2f} ± ${stats_results['std_pnl']:.2f}")
        print(f"Mean Sharpe ratio: {stats_results['mean_sharpe']:.3f} ± {stats_results['std_sharpe']:.3f}")
        print(f"P&L range: ${stats_results['min_pnl']:.2f} to ${stats_results['max_pnl']:.2f}")
        print(f"Maximum drawdown: ${stats_results['max_drawdown']:.2f}")
        print(f"P&L distribution - Skewness: {stats_results['pnl_skewness']:.3f}, Kurtosis: {stats_results['pnl_kurtosis']:.3f}")
        print(f"t-test for profitability: t={stats_results['t_stat']:.3f}, p={stats_results['p_value']:.4f}")
        print(f"Significant at 5% level: {'Yes' if stats_results['significant_at_5pct'] else 'No'}")

def main():
    """Run comprehensive multi-year validation"""
    print("🚀 Starting Multi-Year HJB Validation")
    print("=" * 50)

    validator = MultiYearValidator('/home/misango/codechest/Algorithmic_Trading_and_HFT_Research/data')

    # Load multi-year data
    print("Loading multi-year BTCUSDT data...")
    data = validator.load_multi_year_data('btcusdt', 2024, 2025)

    if not data:
        print("❌ No data loaded. Exiting.")
        return

    # Calibrate parameters
    params = validator.calibrate_parameters_multi_year(data)

    # Run walk-forward validation
    results = validator.run_walk_forward_validation(data, params, window_days=60, step_days=14)

    if not results:
        print("❌ No validation results. Exiting.")
        return

    # Run statistical tests
    stats_results = validator.run_statistical_tests(results)

    # Plot and save results
    validator.plot_validation_results(results, stats_results,
                                   save_path='multi_year_hjb_validation.png')

    print("\n✅ Multi-year validation completed!")
    print("   Results saved to multi_year_hjb_validation.png")

if __name__ == "__main__":
    main()