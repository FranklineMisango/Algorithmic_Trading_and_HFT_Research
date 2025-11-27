"""
Empirical validation of HJB market making model using 2025 BTCUSDT data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import zipfile
import os
import sys
from scipy import stats
import random

# Add the HJB model path
sys.path.append('/home/misango/codechest/Algorithmic_Trading_and_HFT_Research/Market_Making/HJB_DP_MM_Optimisation')

from hjb_gpu_modelling import HJBGPUMarketMaker

class ExtendedEmpiricalValidator:
    """Validate HJB market making model using extended real market data"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.real_data = None
        self.hjb_model = None

    def load_real_data(self, symbol='btcusdt', date='20250101'):
        """Load real BTCUSDT minute data"""
        zip_path = os.path.join(self.data_path, 'crypto', 'binance', 'minute', symbol.lower(), f"{date}_trade.zip")

        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Data file not found: {zip_path}")

        # Extract and load CSV
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            csv_filename = f"{date}_{symbol.lower()}_minute_trade.csv"
            with zip_ref.open(csv_filename) as file:
                # Read CSV data
                lines = file.readlines()
                data = []
                for line in lines:
                    parts = line.decode('utf-8').strip().split(',')
                    if len(parts) == 6:
                        time_ms, open_p, high_p, low_p, close_p, volume = parts
                        data.append({
                            'time_ms': int(time_ms),
                            'open': float(open_p),
                            'high': float(high_p),
                            'low': float(low_p),
                            'close': float(close_p),
                            'volume': float(volume)
                        })

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(date) + pd.to_timedelta(df['time_ms'], unit='ms')
        df = df.set_index('timestamp')

        self.real_data = df
        print(f"Loaded {len(df)} minutes of {symbol.upper()} data for {date}")
        return df

    def calibrate_parameters(self, data):
        """Calibrate HJB model parameters from real data"""
        # Calculate volatility from returns
        returns = np.log(data['close'] / data['close'].shift(1)).dropna()
        sigma = returns.std() * np.sqrt(1440)  # Annualized volatility (assuming minute data)

        # Estimate jump parameters from extreme returns
        jump_threshold = 3 * returns.std()  # 3-sigma threshold
        jumps = returns[abs(returns) > jump_threshold]

        jump_intensity = len(jumps) / len(returns)  # Fraction of jumps
        jump_mean = jumps.mean() if len(jumps) > 0 else 0.0
        jump_std = jumps.std() if len(jumps) > 0 else 0.02

        # Market making parameters (conservative estimates)
        gamma = 0.01  # Risk aversion
        k = 2.0       # Order book liquidity
        c = 1.0       # Base arrival rate

        # Get price statistics for HJB grid bounds
        price_mean = data['close'].mean()
        price_std = data['close'].std()

        # Set price grid bounds based on real data (±3 std from mean)
        S_min = max(0.1, price_mean - 3 * price_std)
        S_max = price_mean + 3 * price_std

        # Scale prices to a reasonable range for HJB solver (around 100)
        price_scale = 100.0 / price_mean
        S_min_scaled = S_min * price_scale
        S_max_scaled = S_max * price_scale

        params = {
            'sigma': sigma,
            'gamma': gamma,
            'k': k,
            'c': c,
            'jump_intensity': jump_intensity,
            'jump_mean': jump_mean,
            'jump_std': jump_std,
            'S_min': S_min_scaled,
            'S_max': S_max_scaled,
            'price_scale': price_scale,
            'original_price_mean': price_mean
        }

        print("Calibrated parameters:")
        for key, value in params.items():
            print(f"  {key}: {value:.6f}")

        return params

    def simulate_hjb_on_real_data(self, data, params, transaction_cost_bps=5):
        """Simulate HJB strategy on real price path"""
        # Initialize HJB model with calibrated price bounds
        self.hjb_model = HJBGPUMarketMaker(
            sigma=params['sigma'],
            gamma=params['gamma'],
            k=params['k'],
            c=params['c'],
            T=1.0,  # 1 day horizon
            S_min=params['S_min'],
            S_max=params['S_max'],
            jump_intensity=params['jump_intensity'],
            jump_mean=params['jump_mean'],
            jump_std=params['jump_std']
        )

        # Solve the PDE
        self.hjb_model.solve_pde()

        # Simulation parameters
        n_steps = len(data) - 1
        dt = 1.0 / 1440  # 1 minute in days

        # Scale real prices to HJB model's price range
        price_scale = params['price_scale']
        scaled_prices = data['close'].values * price_scale

        # Initialize simulation arrays
        inventory = np.zeros(n_steps + 1)
        cash = np.zeros(n_steps + 1)
        pnl = np.zeros(n_steps + 1)

        # Transaction cost per trade (as fraction)
        transaction_cost = transaction_cost_bps / 10000.0

        trades_executed = 0

        for i in range(n_steps):
            t = i * dt  # Time in days
            S_scaled = scaled_prices[i]  # Scaled price for HJB model
            S_real = data['close'].values[i]  # Real price for PnL calculation
            I = inventory[i]  # Current inventory

            # Get optimal quotes from HJB model (in scaled price space)
            bid_quote_scaled, ask_quote_scaled = self.hjb_model.optimal_quotes(t, S_scaled, I)

            # Convert back to real price space
            bid_quote_real = bid_quote_scaled / price_scale
            ask_quote_real = ask_quote_scaled / price_scale

            # Simulate order arrivals and executions
            # Simplified: assume some probability of execution based on spread
            bid_spread = S_real - bid_quote_real
            ask_spread = ask_quote_real - S_real

            # Execution probabilities (simplified model)
            bid_exec_prob = min(0.1, max(0.01, 1.0 / (1.0 + bid_spread * 100)))  # Higher spread = lower prob
            ask_exec_prob = min(0.1, max(0.01, 1.0 / (1.0 + ask_spread * 100)))

            # Simulate executions
            bid_executed = np.random.random() < bid_exec_prob
            ask_executed = np.random.random() < ask_exec_prob

            # Update inventory and cash (using real prices)
            if bid_executed and I < self.hjb_model.I_max:
                inventory[i+1] = I + 1
                cash[i+1] = cash[i] - bid_quote_real * (1 + transaction_cost)
                trades_executed += 1
            elif ask_executed and I > -self.hjb_model.I_max:
                inventory[i+1] = I - 1
                cash[i+1] = cash[i] + ask_quote_real * (1 - transaction_cost)
                trades_executed += 1
            else:
                inventory[i+1] = I
                cash[i+1] = cash[i]

            # Update PnL (using real prices)
            pnl[i+1] = cash[i+1] + inventory[i+1] * S_real

        # Final liquidation
        final_price = data['close'].values[-1]
        final_pnl = cash[-1] + inventory[-1] * final_price * (1 - transaction_cost)

        results = {
            'prices': data['close'].values,
            'scaled_prices': scaled_prices,
            'inventory': inventory,
            'cash': cash,
            'pnl': pnl,
            'final_pnl': final_pnl,
            'trades_executed': trades_executed,
            'sharpe_ratio': self.calculate_sharpe_ratio(pnl),
            'max_drawdown': self.calculate_max_drawdown(pnl)
        }

        return results

    def calculate_sharpe_ratio(self, pnl_series, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if len(pnl_series) < 2:
            return 0.0

        # Calculate returns, avoiding division by zero
        returns = []
        for i in range(1, len(pnl_series)):
            prev_pnl = pnl_series[i-1]
            curr_pnl = pnl_series[i]
            if abs(prev_pnl) > 1e-8:  # Avoid division by very small numbers
                ret = (curr_pnl - prev_pnl) / abs(prev_pnl)
                # Cap extreme returns to avoid outliers
                ret = np.clip(ret, -0.5, 0.5)
                returns.append(ret)
            else:
                returns.append(0.0)  # If previous PnL was effectively zero, assume 0 return

        returns = np.array(returns)

        # Remove any remaining NaN or inf values
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe = excess_returns.mean() / excess_returns.std()

        # Cap Sharpe ratio to reasonable bounds
        return np.clip(sharpe, -5.0, 5.0) * np.sqrt(252)  # Annualized

    def calculate_max_drawdown(self, pnl_series):
        """Calculate maximum drawdown"""
        if len(pnl_series) == 0:
            return 0.0

        peak = pnl_series[0]
        max_drawdown = 0.0

        for value in pnl_series:
            if value > peak:
                peak = value
            if peak > 0:  # Only calculate drawdown if peak is positive
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def run_2025_validation(self, dates, n_simulations_per_date=5):
        """Run validation across 2025 dates"""
        all_results = []

        print(f"Running 2025 validation across {len(dates)} dates...")

        for date in dates:
            try:
                print(f"\nProcessing date: {date}")
                # Load data for this date
                data = self.load_real_data('btcusdt', date)

                # Calibrate parameters for this date
                params = self.calibrate_parameters(data)

                # Run simulations for this date
                for sim in range(n_simulations_per_date):
                    print(f"  Simulation {sim+1}/{n_simulations_per_date}")
                    results = self.simulate_hjb_on_real_data(data, params)
                    results['date'] = date
                    results['simulation'] = sim
                    all_results.append(results)

            except Exception as e:
                print(f"Error processing date {date}: {e}")
                continue

        return all_results

    def plot_2025_results(self, all_results, save_path=None):
        """Plot aggregated results across 2025 dates"""
        if not all_results:
            print("No results to plot")
            return

        # Aggregate results by date
        dates = []
        avg_pnl_by_date = []
        avg_sharpe_by_date = []
        avg_drawdown_by_date = []

        date_groups = {}
        for result in all_results:
            date = result['date']
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(result)

        for date in sorted(date_groups.keys()):
            results_for_date = date_groups[date]
            dates.append(date)
            avg_pnl_by_date.append(np.mean([r['final_pnl'] for r in results_for_date]))
            avg_sharpe_by_date.append(np.mean([r['sharpe_ratio'] for r in results_for_date]))
            avg_drawdown_by_date.append(np.mean([r['max_drawdown'] for r in results_for_date]))

        fig, axes = plt.subplots(3, 1, figsize=(15, 15))

        # Plot average PnL by date
        axes[0].bar(range(len(dates)), avg_pnl_by_date)
        axes[0].set_xticks(range(len(dates)))
        axes[0].set_xticklabels(dates, rotation=45)
        axes[0].set_ylabel('Average Final PnL ($)')
        axes[0].set_title('Average Daily PnL by Date (2025)')
        axes[0].grid(True, alpha=0.3)

        # Plot average Sharpe ratio by date
        axes[1].bar(range(len(dates)), avg_sharpe_by_date)
        axes[1].set_xticks(range(len(dates)))
        axes[1].set_xticklabels(dates, rotation=45)
        axes[1].set_ylabel('Average Sharpe Ratio')
        axes[1].set_title('Average Sharpe Ratio by Date (2025)')
        axes[1].grid(True, alpha=0.3)

        # Plot average max drawdown by date
        axes[2].bar(range(len(dates)), avg_drawdown_by_date)
        axes[2].set_xticks(range(len(dates)))
        axes[2].set_xticklabels(dates, rotation=45)
        axes[2].set_ylabel('Average Max Drawdown')
        axes[2].set_title('Average Max Drawdown by Date (2025)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"2025 results plot saved to {save_path}")
        else:
            plt.show()

def main():
    """Run 2025 empirical validation"""
    # Initialize validator
    validator = ExtendedEmpiricalValidator('/home/misango/codechest/Algorithmic_Trading_and_HFT_Research/data')

    # Select dates for 2025 simulation
    available_dates = [
        '20250101', '20250115', '20250201', '20250215', '20250301', '20250315',
        '20250401', '20250415', '20250501', '20250515', '20250601', '20250615',
        '20250701', '20250715', '20250801', '20250815', '20250901', '20250915',
        '20251001', '20251015', '20251101', '20251115'
    ]

    # Run 2025 validation
    all_results = validator.run_2025_validation(available_dates, n_simulations_per_date=3)

    if not all_results:
        print("No results obtained from 2025 validation")
        return

    # Aggregate all results
    final_pnls = [r['final_pnl'] for r in all_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in all_results]
    max_drawdowns = [r['max_drawdown'] for r in all_results]
    trades_executed = [r['trades_executed'] for r in all_results]

    print("\n=== 2025 EMPIRICAL VALIDATION RESULTS ===")
    print(f"Total simulations: {len(all_results)}")
    print(f"Date range: {available_dates[0]} to {available_dates[-1]}")
    print(f"Average final PnL: ${np.mean(final_pnls):.2f} ± ${np.std(final_pnls):.2f}")
    print(f"Average Sharpe ratio: {np.mean(sharpe_ratios):.3f} ± {np.std(sharpe_ratios):.3f}")
    print(f"Average max drawdown: {np.mean(max_drawdowns)*100:.2f}% ± {np.std(max_drawdowns)*100:.2f}%")
    print(f"Average trades executed: {np.mean(trades_executed):.1f} ± {np.std(trades_executed):.1f}")

    # Statistical significance test (t-test against zero)
    t_stat, p_value = stats.ttest_1samp(final_pnls, 0)
    print(f"\nStatistical significance test (PnL > 0):")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significant at 5% level: {'Yes' if p_value < 0.05 else 'No'}")

    # Calculate annualized metrics
    total_days = len(available_dates)
    annualized_return = np.mean(final_pnls) * (365 / total_days)
    annualized_volatility = np.std(final_pnls) * np.sqrt(365 / total_days)
    annualized_sharpe = np.mean(sharpe_ratios)

    print(f"\nAnnualized Metrics (estimated):")
    print(f"Annualized return: ${annualized_return:.2f}")
    print(f"Annualized volatility: ${annualized_volatility:.2f}")
    print(f"Annualized Sharpe ratio: {annualized_sharpe:.3f}")

    # Plot 2025 results
    validator.plot_2025_results(all_results, save_path='hjb_2025_validation.png')

    # Save detailed results
    results_df = pd.DataFrame({
        'date': [r['date'] for r in all_results],
        'simulation': [r['simulation'] for r in all_results],
        'final_pnl': final_pnls,
        'sharpe_ratio': sharpe_ratios,
        'max_drawdown': max_drawdowns,
        'trades_executed': trades_executed
    })
    results_df.to_csv('hjb_2025_results.csv', index=False)
    print("Detailed results saved to hjb_2025_results.csv")

if __name__ == "__main__":
    main()