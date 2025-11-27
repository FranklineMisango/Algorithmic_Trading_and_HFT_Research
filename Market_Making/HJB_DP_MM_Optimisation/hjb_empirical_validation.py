"""
Empirical validation of HJB market making model using real BTCUSDT data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import zipfile
import os
import sys
from scipy import stats

# Add the HJB model path
sys.path.append('/home/misango/codechest/Algorithmic_Trading_and_HFT_Research/Market_Making/HJB_DP_MM_Optimisation')

from hjb_gpu_modelling import HJBGPUMarketMaker

class EmpiricalValidator:
    """Validate HJB market making model using real market data"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.real_data = None
        self.hjb_model = None

    def load_real_data(self, symbol='btcusdt', date='20241001'):
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

    def run_statistical_tests(self, simulated_pnl, benchmark_returns=None):
        """Run statistical tests on the strategy performance"""
        # Calculate returns properly
        pnl_series = pd.Series(simulated_pnl)
        
        # Calculate percentage returns, handling edge cases
        returns = pnl_series.pct_change().fillna(0).replace([np.inf, -np.inf], 0).clip(-0.5, 0.5)
        sim_returns = returns.values
        sim_returns = sim_returns[np.isfinite(sim_returns)]  # Remove any remaining NaN/inf

        if len(sim_returns) < 10:
            print("Warning: Insufficient data for statistical tests")
            return {
                'mean_return': 0.0,
                'std_return': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'normality_p_value': 1.0
            }

        # Basic statistics
        stats_results = {
            'mean_return': sim_returns.mean(),
            'std_return': sim_returns.std(),
            'skewness': stats.skew(sim_returns) if len(sim_returns) > 2 else 0.0,
            'kurtosis': stats.kurtosis(sim_returns) if len(sim_returns) > 2 else 0.0,
            'sharpe_ratio': self.calculate_sharpe_ratio(simulated_pnl),
            'max_drawdown': self.calculate_max_drawdown(simulated_pnl),
            'total_return': (simulated_pnl[-1] - simulated_pnl[0]) / max(abs(simulated_pnl[0]), 1e-8) if simulated_pnl[0] != 0 else 0.0
        }

        # Normality test
        try:
            _, normality_p = stats.shapiro(sim_returns[:min(5000, len(sim_returns))])  # Shapiro-Wilk test
            stats_results['normality_p_value'] = normality_p
        except:
            stats_results['normality_p_value'] = 1.0  # Default to 1.0 if test fails

        # If benchmark provided, calculate alpha/beta
        if benchmark_returns is not None:
            # Simple linear regression for beta
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(benchmark_returns, sim_returns)
                stats_results['beta'] = slope
                stats_results['alpha'] = intercept
                stats_results['r_squared'] = r_value**2
            except:
                stats_results['beta'] = np.nan
                stats_results['alpha'] = np.nan
                stats_results['r_squared'] = np.nan

        return stats_results

    def plot_results(self, data, simulation_results, save_path=None):
        """Plot simulation results"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

        timestamps = data.index

        # Plot 1: Price and inventory
        ax1 = axes[0]
        ax1.plot(timestamps, data['close'], label='BTC Price', color='black', alpha=0.7)
        ax1.set_ylabel('BTC Price (USD)')
        ax1.legend(loc='upper left')

        ax1_twin = ax1.twinx()
        ax1_twin.plot(timestamps, simulation_results['inventory'], label='Inventory', color='blue', alpha=0.7)
        ax1_twin.set_ylabel('Inventory')
        ax1_twin.legend(loc='upper right')

        ax1.set_title('BTC Price and HJB Inventory')

        # Plot 2: PnL
        axes[1].plot(timestamps, simulation_results['pnl'], label='HJB PnL', color='green')
        axes[1].set_ylabel('PnL (USD)')
        axes[1].set_title('HJB Strategy PnL')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Cash position
        axes[2].plot(timestamps, simulation_results['cash'], label='Cash', color='red')
        axes[2].set_ylabel('Cash (USD)')
        axes[2].set_title('Cash Position')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Rolling Sharpe ratio (if enough data)
        if len(simulation_results['pnl']) > 60:  # Need at least 1 hour of data
            pnl_series = pd.Series(simulation_results['pnl'])
            returns = pnl_series.pct_change().fillna(0)
            rolling_sharpe = returns.rolling(60, min_periods=1).apply(
                lambda x: x.mean() / x.std() * np.sqrt(1440) if x.std() > 0 else 0
            ).fillna(0)
            # Ensure rolling_sharpe has same length as timestamps
            if len(rolling_sharpe) < len(timestamps):
                # Pad with zeros at the beginning
                padding = len(timestamps) - len(rolling_sharpe)
                rolling_sharpe = pd.concat([pd.Series([0] * padding), rolling_sharpe]).reset_index(drop=True)
            elif len(rolling_sharpe) > len(timestamps):
                # Truncate if too long
                rolling_sharpe = rolling_sharpe[:len(timestamps)]

            axes[3].plot(timestamps, rolling_sharpe.values, label='Rolling Sharpe (1h)', color='purple')
            axes[3].set_ylabel('Sharpe Ratio')
            axes[3].set_title('Rolling Sharpe Ratio')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Results plot saved to {save_path}")
        else:
            plt.show()

def main():
    """Run empirical validation"""
    # Initialize validator
    validator = EmpiricalValidator('/home/misango/codechest/Algorithmic_Trading_and_HFT_Research/data')

    # Load real data
    data = validator.load_real_data('btcusdt', '20250101')

    # Calibrate parameters
    params = validator.calibrate_parameters(data)

    # Run multiple simulations for statistical significance
    n_simulations = 10
    results_list = []

    print(f"\nRunning {n_simulations} simulations...")

    for i in range(n_simulations):
        print(f"Simulation {i+1}/{n_simulations}")
        results = validator.simulate_hjb_on_real_data(data, params)
        results_list.append(results)

    # Aggregate results
    final_pnls = [r['final_pnl'] for r in results_list]
    sharpe_ratios = [r['sharpe_ratio'] for r in results_list]
    max_drawdowns = [r['max_drawdown'] for r in results_list]
    trades_executed = [r['trades_executed'] for r in results_list]

    print("\n=== EMPIRICAL VALIDATION RESULTS ===")
    print(f"Simulations run: {n_simulations}")
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

    # Plot results from first simulation
    validator.plot_results(data, results_list[0], save_path='hjb_empirical_validation.png')

    # Run statistical tests
    stats_results = validator.run_statistical_tests(results_list[0]['pnl'])
    print("\nDetailed statistics for first simulation:")
    for key, value in stats_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()