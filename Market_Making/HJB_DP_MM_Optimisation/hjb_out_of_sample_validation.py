"""
Out-of-sample validation for HJB market making model
Addresses critique requirements for multi-year backtesting, statistical significance,
and comparison against realistic benchmarks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import zipfile
import os
import sys
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
import seaborn as sns

# Add the HJB model path
sys.path.append('/Users/misango/codechest/Algorithmic_Trading_and_HFT_Research/Market_Making/HJB_DP_MM_Optimisation')

from hjb_gpu_modelling import HJBGPUMarketMaker

class OutOfSampleValidator:
    """Rigorous out-of-sample validation with statistical significance testing"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.validation_results = {}
        self.benchmark_results = {}

    def load_multi_year_data(self, symbol='btcusdt', start_year=2021, end_year=2024):
        """Load multi-year dataset for robust out-of-sample validation"""
        all_data = []

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                date_str = "02d"

                try:
                    data = self._load_single_month(symbol, date_str)
                    if data:
                        all_data.extend(data)
                        print(f"Loaded {len(data)} records for {date_str}")
                except Exception as e:
                    print(f"Failed to load {date_str}: {e}")
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

    def time_series_cross_validation(self, data, n_splits=5, train_years=2, test_months=3):
        """Implement proper time-series cross-validation"""
        print("Running time-series cross-validation...")

        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('datetime').reset_index(drop=True)

        # Group by year-month for splitting
        df['year_month'] = df['datetime'].dt.to_period('M')

        unique_months = sorted(df['year_month'].unique())
        total_months = len(unique_months)

        if total_months < (train_years * 12 + test_months) * (n_splits + 1):
            raise ValueError("Insufficient data for the specified validation parameters")

        results = []

        for fold in range(n_splits):
            # Rolling window: train on past years, test on future months
            test_start_idx = fold * test_months
            test_end_idx = test_start_idx + test_months
            train_end_month = unique_months[test_start_idx]

            # Training data: previous train_years
            train_start_month = train_end_month - train_years * 12
            train_mask = (df['year_month'] >= train_start_month) & (df['year_month'] < train_end_month)
            test_mask = (df['year_month'] >= unique_months[test_start_idx]) & (df['year_month'] < unique_months[test_end_idx])

            train_data = df[train_mask]
            test_data = df[test_mask]

            print(f"Fold {fold+1}: Train {train_start_month} to {train_end_month-1}, Test {unique_months[test_start_idx]} to {unique_months[test_end_idx-1]}")
            print(f"  Train size: {len(train_data)}, Test size: {len(test_data)}")

            if len(train_data) == 0 or len(test_data) == 0:
                continue

            # Run validation for this fold
            fold_result = self._validate_single_fold(train_data, test_data)
            results.append(fold_result)

        return results

    def _validate_single_fold(self, train_data, test_data):
        """Validate on a single train/test fold"""
        # Calibrate parameters on training data
        params = self._calibrate_parameters(train_data)

        # Initialize HJB model
        hjb_model = HJBGPUMarketMaker(
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

        # Solve PDE
        hjb_model.solve_pde()

        # Simulate on test data
        test_results = self._simulate_strategy(hjb_model, test_data, params)

        # Compare against benchmarks
        benchmark_results = self._run_benchmarks(test_data, params)

        return {
            'hjb_results': test_results,
            'benchmarks': benchmark_results,
            'parameters': params,
            'train_size': len(train_data),
            'test_size': len(test_data)
        }

    def _calibrate_parameters(self, data):
        """Calibrate model parameters from training data"""
        returns = np.log(data['close'] / data['close'].shift(1)).dropna()

        # Volatility estimation
        sigma = returns.std() * np.sqrt(1440)  # Annualized

        # Jump detection (3-sigma events)
        jump_threshold = 3 * returns.std()
        jumps = returns[abs(returns) > jump_threshold]

        jump_intensity = len(jumps) / len(returns)
        jump_mean = jumps.mean() if len(jumps) > 0 else 0.0
        jump_std = jumps.std() if len(jumps) > 0 else 0.02

        # Conservative market making parameters
        gamma = 0.001  # Very low risk aversion for crypto
        k = 1.0        # Moderate liquidity
        c = 0.5        # Base arrival rate

        # Price grid bounds
        price_mean = data['close'].mean()
        price_std = data['close'].std()
        price_scale = 100.0 / price_mean

        S_min = max(0.1, price_mean - 4 * price_std) * price_scale
        S_max = (price_mean + 4 * price_std) * price_scale

        return {
            'sigma': sigma,
            'gamma': gamma,
            'k': k,
            'c': c,
            'jump_intensity': jump_intensity,
            'jump_mean': jump_mean,
            'jump_std': jump_std,
            'S_min': S_min,
            'S_max': S_max,
            'price_scale': price_scale,
            'original_price_mean': price_mean
        }

    def _simulate_strategy(self, hjb_model, test_data, params, transaction_cost_bps=10):
        """Simulate HJB strategy on test data with realistic costs"""
        n_steps = len(test_data) - 1
        dt = 1.0 / 1440  # 1 minute in days

        price_scale = params['price_scale']
        scaled_prices = test_data['close'].values * price_scale

        # Initialize tracking
        inventory = np.zeros(n_steps + 1)
        cash = np.zeros(n_steps + 1)
        pnl = np.zeros(n_steps + 1)
        trades = []

        transaction_cost = transaction_cost_bps / 10000.0  # Convert bps to fraction

        for i in range(n_steps):
            t = i * dt
            S_scaled = scaled_prices[i]
            S_real = test_data['close'].values[i]
            I = inventory[i]

            # Get optimal quotes
            bid_scaled, ask_scaled = hjb_model.optimal_quotes(t, S_scaled, I)
            bid_real = bid_scaled / price_scale
            ask_real = ask_scaled / price_scale

            # Realistic execution model based on spread and market conditions
            spread_bps = (ask_real - bid_real) / S_real * 10000
            market_volatility = np.log(test_data['high'].values[i] / test_data['low'].values[i])

            # Execution probability depends on spread and volatility
            base_exec_prob = 0.02  # Base 2% probability per minute
            spread_factor = max(0.1, 1.0 / (1.0 + spread_bps / 50))  # Wider spread = lower prob
            vol_factor = min(2.0, 1.0 + market_volatility * 10)  # Higher vol = higher prob

            exec_prob = base_exec_prob * spread_factor * vol_factor

            # Simulate executions
            bid_executed = np.random.random() < exec_prob and I < hjb_model.I_max
            ask_executed = np.random.random() < exec_prob and I > -hjb_model.I_max

            # Update positions
            if bid_executed:
                inventory[i+1] = I + 1
                cash[i+1] = cash[i] - bid_real * (1 + transaction_cost)
                trades.append({'time': i, 'side': 'buy', 'price': bid_real, 'quantity': 1})
            elif ask_executed:
                inventory[i+1] = I - 1
                cash[i+1] = cash[i] + ask_real * (1 - transaction_cost)
                trades.append({'time': i, 'side': 'sell', 'price': ask_real, 'quantity': 1})
            else:
                inventory[i+1] = I
                cash[i+1] = cash[i]

            # Mark-to-market PnL
            pnl[i+1] = cash[i+1] + inventory[i+1] * S_real

        # Final liquidation with costs
        final_price = test_data['close'].values[-1]
        liquidation_cost = abs(inventory[-1]) * final_price * transaction_cost
        final_pnl = cash[-1] + inventory[-1] * final_price - liquidation_cost

        return {
            'pnl': pnl,
            'final_pnl': final_pnl,
            'inventory': inventory,
            'cash': cash,
            'trades': trades,
            'sharpe_ratio': self._calculate_sharpe_ratio(pnl),
            'max_drawdown': self._calculate_max_drawdown(pnl),
            'total_return': (final_pnl - pnl[0]) / max(abs(pnl[0]), 1e-8),
            'num_trades': len(trades)
        }

    def _run_benchmarks(self, test_data, params):
        """Run benchmark strategies for comparison"""
        benchmarks = {}

        # 1. Buy and Hold
        initial_price = test_data['close'].values[0]
        final_price = test_data['close'].values[-1]
        bh_return = (final_price - initial_price) / initial_price

        # Calculate daily returns for Sharpe
        bh_pnl = test_data['close'].values / initial_price
        benchmarks['buy_hold'] = {
            'total_return': bh_return,
            'sharpe_ratio': self._calculate_sharpe_ratio(bh_pnl),
            'max_drawdown': self._calculate_max_drawdown(bh_pnl)
        }

        # 2. Avellaneda-Stoikov benchmark
        as_results = self._simulate_avellaneda_stoikov(test_data, params)
        benchmarks['avellaneda_stoikov'] = as_results

        # 3. Random trading (null hypothesis)
        random_results = self._simulate_random_strategy(test_data, params)
        benchmarks['random'] = random_results

        return benchmarks

    def _simulate_avellaneda_stoikov(self, test_data, params):
        """Simulate classical Avellaneda-Stoikov model"""
        n_steps = len(test_data) - 1
        dt = 1.0 / 1440

        price_scale = params['price_scale']
        scaled_prices = test_data['close'].values * price_scale

        inventory = np.zeros(n_steps + 1)
        cash = np.zeros(n_steps + 1)
        pnl = np.zeros(n_steps + 1)

        transaction_cost = 10 / 10000.0  # 10 bps

        for i in range(n_steps):
            t = i * dt
            S_scaled = scaled_prices[i]
            S_real = test_data['close'].values[i]
            I = inventory[i]

            # AS optimal spreads (simplified)
            remaining_time = 1.0 - t
            gamma = params['gamma']
            sigma = params['sigma']
            k = params['k']

            spread_factor = gamma * sigma**2 * remaining_time + (2/gamma) * np.log(1 + gamma/k)
            bid_spread = spread_factor / 2
            ask_spread = spread_factor / 2

            bid_scaled = S_scaled - bid_spread
            ask_scaled = S_scaled + ask_spread

            bid_real = bid_scaled / price_scale
            ask_real = ask_scaled / price_scale

            # Same execution logic as HJB
            spread_bps = (ask_real - bid_real) / S_real * 10000
            exec_prob = 0.02 * max(0.1, 1.0 / (1.0 + spread_bps / 50))

            bid_executed = np.random.random() < exec_prob and abs(I) < 5
            ask_executed = np.random.random() < exec_prob and abs(I) < 5

            if bid_executed:
                inventory[i+1] = I + 1
                cash[i+1] = cash[i] - bid_real * (1 + transaction_cost)
            elif ask_executed:
                inventory[i+1] = I - 1
                cash[i+1] = cash[i] + ask_real * (1 - transaction_cost)
            else:
                inventory[i+1] = I
                cash[i+1] = cash[i]

            pnl[i+1] = cash[i+1] + inventory[i+1] * S_real

        final_price = test_data['close'].values[-1]
        final_pnl = cash[-1] + inventory[-1] * final_price - abs(inventory[-1]) * final_price * transaction_cost

        return {
            'pnl': pnl,
            'final_pnl': final_pnl,
            'sharpe_ratio': self._calculate_sharpe_ratio(pnl),
            'max_drawdown': self._calculate_max_drawdown(pnl),
            'total_return': (final_pnl - pnl[0]) / max(abs(pnl[0]), 1e-8)
        }

    def _simulate_random_strategy(self, test_data, params):
        """Random trading strategy for null hypothesis testing"""
        n_steps = len(test_data) - 1

        inventory = np.zeros(n_steps + 1)
        cash = np.zeros(n_steps + 1)
        pnl = np.zeros(n_steps + 1)

        transaction_cost = 10 / 10000.0

        for i in range(n_steps):
            S_real = test_data['close'].values[i]
            I = inventory[i]

            # Random action with small probability
            action_prob = 0.01  # 1% chance per minute

            if np.random.random() < action_prob:
                if np.random.random() < 0.5 and abs(I) < 5:  # Buy
                    inventory[i+1] = I + 1
                    cash[i+1] = cash[i] - S_real * (1 + transaction_cost)
                elif abs(I) < 5:  # Sell
                    inventory[i+1] = I - 1
                    cash[i+1] = cash[i] + S_real * (1 - transaction_cost)
                else:
                    inventory[i+1] = I
                    cash[i+1] = cash[i]
            else:
                inventory[i+1] = I
                cash[i+1] = cash[i]

            pnl[i+1] = cash[i+1] + inventory[i+1] * S_real

        final_price = test_data['close'].values[-1]
        final_pnl = cash[-1] + inventory[-1] * final_price - abs(inventory[-1]) * final_price * transaction_cost

        return {
            'pnl': pnl,
            'final_pnl': final_pnl,
            'sharpe_ratio': self._calculate_sharpe_ratio(pnl),
            'max_drawdown': self._calculate_max_drawdown(pnl),
            'total_return': (final_pnl - pnl[0]) / max(abs(pnl[0]), 1e-8)
        }

    def _calculate_sharpe_ratio(self, pnl_series, risk_free_rate=0.02):
        """Calculate Sharpe ratio with proper error handling"""
        if len(pnl_series) < 2:
            return 0.0

        returns = []
        for i in range(1, len(pnl_series)):
            prev = pnl_series[i-1]
            curr = pnl_series[i]
            if abs(prev) > 1e-8:
                ret = (curr - prev) / abs(prev)
                ret = np.clip(ret, -0.5, 0.5)
                returns.append(ret)
            else:
                returns.append(0.0)

        returns = np.array(returns)
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        sharpe = excess_returns.mean() / excess_returns.std()
        return np.clip(sharpe, -5.0, 5.0) * np.sqrt(252)

    def _calculate_max_drawdown(self, pnl_series):
        """Calculate maximum drawdown"""
        if len(pnl_series) == 0:
            return 0.0

        peak = pnl_series[0]
        max_dd = 0.0

        for value in pnl_series:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)

        return max_dd

    def run_statistical_tests(self, validation_results):
        """Run comprehensive statistical tests on validation results"""
        print("\nRunning statistical significance tests...")

        # Extract HJB returns across all folds
        hjb_returns = []
        bh_returns = []
        as_returns = []

        for fold in validation_results:
            hjb_returns.append(fold['hjb_results']['total_return'])
            bh_returns.append(fold['benchmarks']['buy_hold']['total_return'])
            as_returns.append(fold['benchmarks']['avellaneda_stoikov']['total_return'])

        hjb_returns = np.array(hjb_returns)
        bh_returns = np.array(bh_returns)
        as_returns = np.array(as_returns)

        # Remove any NaN or inf values
        hjb_returns = hjb_returns[np.isfinite(hjb_returns)]
        bh_returns = bh_returns[np.isfinite(bh_returns)]
        as_returns = as_returns[np.isfinite(as_returns)]

        if len(hjb_returns) < 3:
            print("Insufficient data for statistical tests")
            return {}

        # Statistical tests
        tests = {}

        # T-test vs Buy & Hold
        try:
            t_stat_bh, p_value_bh = stats.ttest_1samp(hjb_returns - bh_returns, 0)
            tests['vs_buy_hold'] = {
                't_statistic': t_stat_bh,
                'p_value': p_value_bh,
                'significant': p_value_bh < 0.05,
                'mean_difference': np.mean(hjb_returns - bh_returns),
                'std_difference': np.std(hjb_returns - bh_returns)
            }
        except:
            tests['vs_buy_hold'] = {'error': 'Test failed'}

        # T-test vs Avellaneda-Stoikov
        try:
            t_stat_as, p_value_as = stats.ttest_1samp(hjb_returns - as_returns, 0)
            tests['vs_avellaneda_stoikov'] = {
                't_statistic': t_stat_as,
                'p_value': p_value_as,
                'significant': p_value_as < 0.05,
                'mean_difference': np.mean(hjb_returns - as_returns),
                'std_difference': np.std(hjb_returns - as_returns)
            }
        except:
            tests['vs_avellaneda_stoikov'] = {'error': 'Test failed'}

        # Sharpe ratio comparison
        hjb_sharpes = [fold['hjb_results']['sharpe_ratio'] for fold in validation_results]
        as_sharpes = [fold['benchmarks']['avellaneda_stoikov']['sharpe_ratio'] for fold in validation_results]

        try:
            t_stat_sharpe, p_value_sharpe = stats.ttest_ind(hjb_sharpes, as_sharpes)
            tests['sharpe_ratio_comparison'] = {
                't_statistic': t_stat_sharpe,
                'p_value': p_value_sharpe,
                'significant': p_value_sharpe < 0.05,
                'hjb_mean_sharpe': np.mean(hjb_sharpes),
                'as_mean_sharpe': np.mean(as_sharpes)
            }
        except:
            tests['sharpe_ratio_comparison'] = {'error': 'Test failed'}

        return tests

    def plot_validation_results(self, validation_results, statistical_tests, save_path=None):
        """Create comprehensive validation plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('HJB Market Making: Out-of-Sample Validation Results', fontsize=16)

        # Extract data for plotting
        hjb_returns = [fold['hjb_results']['total_return'] for fold in validation_results]
        bh_returns = [fold['benchmarks']['buy_hold']['total_return'] for fold in validation_results]
        as_returns = [fold['benchmarks']['avellaneda_stoikov']['total_return'] for fold in validation_results]

        hjb_sharpes = [fold['hjb_results']['sharpe_ratio'] for fold in validation_results]
        as_sharpes = [fold['benchmarks']['avellaneda_stoikov']['sharpe_ratio'] for fold in validation_results]

        # Returns comparison
        axes[0,0].boxplot([hjb_returns, bh_returns, as_returns],
                         labels=['HJB', 'Buy & Hold', 'A-S'])
        axes[0,0].set_title('Total Returns Distribution')
        axes[0,0].set_ylabel('Total Return')
        axes[0,0].grid(True, alpha=0.3)

        # Sharpe ratios comparison
        axes[0,1].boxplot([hjb_sharpes, as_sharpes],
                         labels=['HJB', 'A-S'])
        axes[0,1].set_title('Sharpe Ratio Distribution')
        axes[0,1].set_ylabel('Sharpe Ratio')
        axes[0,1].grid(True, alpha=0.3)

        # Returns over time (last fold)
        if validation_results:
            last_fold = validation_results[-1]
            pnl_data = last_fold['hjb_results']['pnl']
            time_steps = np.arange(len(pnl_data)) / 1440  # Convert to days

            axes[0,2].plot(time_steps, pnl_data, label='HJB Strategy', linewidth=2)
            axes[0,2].plot(time_steps, last_fold['benchmarks']['buy_hold']['pnl'],
                          label='Buy & Hold', alpha=0.7)
            axes[0,2].set_title('PnL Over Time (Last Fold)')
            axes[0,2].set_xlabel('Days')
            axes[0,2].set_ylabel('Normalized PnL')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)

        # Statistical significance
        if 'vs_buy_hold' in statistical_tests and 'significant' in statistical_tests['vs_buy_hold']:
            sig_bh = statistical_tests['vs_buy_hold']['significant']
            p_bh = statistical_tests['vs_buy_hold']['p_value']
            axes[1,0].bar(['HJB vs Buy&Hold'], [p_bh], color='red' if sig_bh else 'blue')
            axes[1,0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
            axes[1,0].set_title('Statistical Significance vs Buy&Hold')
            axes[1,0].set_ylabel('p-value')
            axes[1,0].set_ylim(0, 0.1)
            axes[1,0].legend()

        if 'vs_avellaneda_stoikov' in statistical_tests and 'significant' in statistical_tests['vs_avellaneda_stoikov']:
            sig_as = statistical_tests['vs_avellaneda_stoikov']['significant']
            p_as = statistical_tests['vs_avellaneda_stoikov']['p_value']
            axes[1,1].bar(['HJB vs A-S'], [p_as], color='red' if sig_as else 'blue')
            axes[1,1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
            axes[1,1].set_title('Statistical Significance vs A-S')
            axes[1,1].set_ylabel('p-value')
            axes[1,1].set_ylim(0, 0.1)
            axes[1,1].legend()

        # Risk metrics
        hjb_maxdd = [fold['hjb_results']['max_drawdown'] for fold in validation_results]
        as_maxdd = [fold['benchmarks']['avellaneda_stoikov']['max_drawdown'] for fold in validation_results]

        axes[1,2].boxplot([hjb_maxdd, as_maxdd], labels=['HJB', 'A-S'])
        axes[1,2].set_title('Maximum Drawdown Distribution')
        axes[1,2].set_ylabel('Max Drawdown')
        axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation plots saved to {save_path}")

        plt.show()

        # Print summary statistics
        print("\n" + "="*60)
        print("OUT-OF-SAMPLE VALIDATION SUMMARY")
        print("="*60)

        print(f"Number of validation folds: {len(validation_results)}")
        print(".4f")
        print(".4f")
        print(".4f")

        print(".2f")
        print(".2f")

        if statistical_tests.get('vs_buy_hold', {}).get('significant'):
            print("✓ Statistically significant vs Buy & Hold (p < 0.05)")
        else:
            print("✗ Not statistically significant vs Buy & Hold")

        if statistical_tests.get('vs_avellaneda_stoikov', {}).get('significant'):
            print("✓ Statistically significant vs Avellaneda-Stoikov (p < 0.05)")
        else:
            print("✗ Not statistically significant vs Avellaneda-Stoikov")

        print("="*60)

    def run_full_validation(self, data_path, save_plots=True):
        """Run complete out-of-sample validation pipeline"""
        print("Starting comprehensive out-of-sample validation...")

        # Load multi-year data
        data = self.load_multi_year_data()

        if not data:
            raise ValueError("No data loaded for validation")

        # Run time-series cross-validation
        validation_results = self.time_series_cross_validation(data)

        # Run statistical tests
        statistical_tests = self.run_statistical_tests(validation_results)

        # Generate plots and summary
        plot_path = "/Users/misango/codechest/Algorithmic_Trading_and_HFT_Research/Market_Making/HJB_DP_MM_Optimisation/hjb_validation_results.png" if save_plots else None
        self.plot_validation_results(validation_results, statistical_tests, plot_path)

        return {
            'validation_results': validation_results,
            'statistical_tests': statistical_tests
        }


if __name__ == "__main__":
    # Example usage
    data_path = "/Users/misango/codechest/Algorithmic_Trading_and_HFT_Research/data_pipeline/data"

    validator = OutOfSampleValidator(data_path)

    try:
        results = validator.run_full_validation(data_path)
        print("Validation completed successfully!")
    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
