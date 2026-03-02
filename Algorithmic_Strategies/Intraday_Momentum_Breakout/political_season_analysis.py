"""
Political Season Analysis - Parameter Optimization Across Market Regimes

This script analyzes the intraday momentum breakout strategy across 4 political seasons:
- 2012-2016: Obama second term
- 2016-2021: Trump administration + COVID
- 2021-2024: Biden administration
- 2024-2026: Post-2024 election period

For each season, we optimize parameters to achieve profitability and save comprehensive results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from itertools import product
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Import our components
from noise_area import NoiseAreaCalculator
from signal_generator import SignalGenerator
from position_sizer import PositionSizer
from enhanced_backtester import EnhancedBacktester, OrderSide, OrderType

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class PoliticalSeasonOptimizer:
    """Optimize strategy parameters across political seasons."""

    def __init__(self, base_config_path: str = 'config.yaml'):
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        # Define political seasons
        self.seasons = {
            '2012_2016': {'start': '2012-01-01', 'end': '2016-12-31', 'name': 'Obama Term 2'},
            '2016_2021': {'start': '2016-01-01', 'end': '2021-12-31', 'name': 'Trump + COVID'},
            '2021_2024': {'start': '2021-01-01', 'end': '2024-12-31', 'name': 'Biden Term'},
            '2024_2026': {'start': '2024-01-01', 'end': '2026-12-31', 'name': 'Post-2024'}
        }

        # Parameter ranges for optimization
        self.param_ranges = {
            'noise_area_lookback_days': [10, 20, 30, 60],
            'target_daily_volatility': [0.05, 0.10, 0.15, 0.20, 0.25],
            'atr_multiplier': [1.0, 1.5, 2.0, 2.5],
            'confirmation_bars': [1, 2, 3],
            'volume_threshold_percentile': [50, 60, 70, 80],
            'min_signal_strength': [30, 40, 50, 60],
            'max_hold_bars': [39, 78, 117],  # 1, 2, 3 trading days
            'trailing_stop_percent': [0.005, 0.01, 0.015, 0.02]
        }

        # Results storage
        self.results = {}
        self.best_params = {}

        # Create output directory
        self.output_dir = Path("results/political_season_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load ES and NQ market data."""
        print("Loading market data...")

        # Load ES data
        es_data = pd.read_csv(
            'Data/ES_5min_RTH.csv',
            index_col='ts_event',
            parse_dates=True
        )

        # Load NQ data
        nq_data = pd.read_csv(
            'Data/NQ_5min_RTH.csv',
            index_col='ts_event',
            parse_dates=True
        )

        print(f"ES: {len(es_data)} bars from {es_data.index[0]} to {es_data.index[-1]}")
        print(f"NQ: {len(nq_data)} bars from {nq_data.index[0]} to {nq_data.index[-1]}")

        return {'ES': es_data, 'NQ': nq_data}

    def filter_season_data(self, data: Dict[str, pd.DataFrame],
                          start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Filter data for specific season."""
        filtered_data = {}
        for symbol, df in data.items():
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_data[symbol] = df[mask].copy()

            if len(filtered_data[symbol]) == 0:
                print(f"Warning: No {symbol} data found for period {start_date} to {end_date}")
            else:
                print(f"{symbol}: {len(filtered_data[symbol])} bars for season")

        return filtered_data

    def create_config_variant(self, param_dict: Dict) -> dict:
        """Create config variant with parameter changes."""
        config = self.base_config.copy()

        # Update parameters
        config['strategy']['noise_area']['lookback_days'] = param_dict['noise_area_lookback_days']
        config['strategy']['position_sizing']['target_daily_volatility'] = param_dict['target_daily_volatility']
        config['strategy']['noise_area']['atr_multiplier'] = param_dict['atr_multiplier']
        config['strategy']['entry_exit']['confirmation_bars'] = param_dict['confirmation_bars']
        config['strategy']['entry_exit']['volume_threshold_percentile'] = param_dict['volume_threshold_percentile']
        config['strategy']['entry_exit']['min_signal_strength'] = param_dict['min_signal_strength']
        config['strategy']['exit']['max_hold_bars'] = param_dict['max_hold_bars']
        config['strategy']['exit']['trailing_stop_percent'] = param_dict['trailing_stop_percent']

        return config

    def generate_signals_for_config(self, config: dict, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate trading signals using current config."""
        try:
            # Initialize components
            calculator = NoiseAreaCalculator(config)
            signal_gen = SignalGenerator(config)
            sizer = PositionSizer(config)

            # Process ES data
            es_data = data['ES'].copy()
            es_data = calculator.calculate_noise_area(es_data)
            es_data = calculator.identify_breakouts(es_data)
            es_data = signal_gen.generate_signals(es_data)

            # Process NQ data
            nq_data = data['NQ'].copy()
            nq_data = calculator.calculate_noise_area(nq_data)
            nq_data = calculator.identify_breakouts(nq_data)
            nq_data = signal_gen.generate_signals(nq_data)

            # Position sizing
            portfolio = sizer.calculate_portfolio_positions(es_data, nq_data)

            return portfolio

        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return {}

    def run_single_backtest(self, config: dict, signals_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run a single backtest and return results."""
        try:
            backtester = EnhancedBacktester(config)
            backtester.load_market_data({'ES': signals_data.get('ES_momentum', pd.DataFrame()),
                                       'NQ': signals_data.get('NQ_momentum', pd.DataFrame())})

            equity_curve = backtester.run_backtest(signals_data)

            return {
                'equity_curve': equity_curve,
                'performance_metrics': backtester.performance_metrics,
                'trades': backtester.get_trades_dataframe(),
                'config': config
            }

        except Exception as e:
            print(f"Backtest failed: {str(e)}")
            return {}

    def optimize_season_parameters(self, season_name: str, season_data: Dict[str, pd.DataFrame],
                                 max_combinations: int = 50) -> Dict:
        """Optimize parameters for a specific season."""
        print(f"\n{'='*60}")
        print(f"OPTIMIZING PARAMETERS FOR {season_name.upper()}")
        print(f"{'='*60}")

        # Generate parameter combinations (sample to limit computation)
        param_keys = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())

        # Create all combinations
        all_combinations = list(product(*param_values))
        np.random.seed(42)  # For reproducibility

        # Sample combinations if too many
        if len(all_combinations) > max_combinations:
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            combinations = [all_combinations[i] for i in indices]
        else:
            combinations = all_combinations

        print(f"Testing {len(combinations)} parameter combinations...")

        best_sharpe = -np.inf
        best_params = None
        best_results = None

        for i, param_combo in enumerate(combinations):
            param_dict = dict(zip(param_keys, param_combo))

            # Create config variant
            config = self.create_config_variant(param_dict)

            # Generate signals
            signals_data = self.generate_signals_for_config(config, season_data)

            if not signals_data:
                continue

            # Run backtest
            results = self.run_single_backtest(config, signals_data)

            if not results:
                continue

            # Evaluate performance
            sharpe = results['performance_metrics'].get('sharpe_ratio', -np.inf)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = param_dict
                best_results = results

            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{len(combinations)} combinations. Best Sharpe: {best_sharpe:.2f}")

        print(f"Best parameters for {season_name}: Sharpe = {best_sharpe:.2f}")

        return {
            'best_params': best_params,
            'best_results': best_results,
            'sharpe_ratio': best_sharpe
        }

    def run_full_analysis(self):
        """Run complete analysis across all seasons."""
        print("STARTING POLITICAL SEASON ANALYSIS")
        print("="*60)

        # Load full dataset
        full_data = self.load_market_data()

        # Analyze each season
        for season_key, season_info in self.seasons.items():
            print(f"\nAnalyzing {season_info['name']} ({season_key})")

            # Filter data for season
            season_data = self.filter_season_data(
                full_data, season_info['start'], season_info['end']
            )

            if not season_data or all(len(df) == 0 for df in season_data.values()):
                print(f"Skipping {season_key} - no data available")
                continue

            # Optimize parameters
            optimization_results = self.optimize_season_parameters(season_key, season_data)

            if optimization_results['best_params'] is not None:
                self.best_params[season_key] = optimization_results['best_params']
                self.results[season_key] = optimization_results['best_results']

                # Save season results
                self.save_season_results(season_key, optimization_results, season_info)

        # Generate comparative analysis
        self.generate_comparative_analysis()

        print("\n" + "="*60)
        print("POLITICAL SEASON ANALYSIS COMPLETE")
        print("="*60)

    def save_season_results(self, season_key: str, optimization_results: Dict, season_info: Dict):
        """Save results for a specific season."""
        season_dir = self.output_dir / season_key
        season_dir.mkdir(exist_ok=True)

        # Save best parameters
        with open(season_dir / 'best_parameters.yaml', 'w') as f:
            yaml.dump(optimization_results['best_params'], f, default_flow_style=False)

        # Save performance metrics
        with open(season_dir / 'performance_metrics.json', 'w') as f:
            json.dump(optimization_results['best_results']['performance_metrics'], f, indent=2, default=str)

        # Save equity curve
        optimization_results['best_results']['equity_curve'].to_csv(season_dir / 'equity_curve.csv')

        # Save trades
        optimization_results['best_results']['trades'].to_csv(season_dir / 'trades.csv', index=False)

        # Save config
        with open(season_dir / 'config_used.yaml', 'w') as f:
            yaml.dump(optimization_results['best_results']['config'], f, default_flow_style=False)

        print(f"Results saved to {season_dir}")

    def generate_comparative_analysis(self):
        """Generate comparative analysis across seasons."""
        print("\nGenerating comparative analysis...")

        # Collect metrics across seasons
        season_metrics = []
        equity_curves = {}

        for season_key, results in self.results.items():
            metrics = results['performance_metrics'].copy()
            metrics['season'] = season_key
            metrics['season_name'] = self.seasons[season_key]['name']
            season_metrics.append(metrics)

            # Normalize equity curves for comparison
            equity = results['equity_curve'].copy()
            initial_value = equity['portfolio_value'].iloc[0]
            equity['normalized_value'] = equity['portfolio_value'] / initial_value
            equity_curves[season_key] = equity

        # Create metrics comparison DataFrame
        metrics_df = pd.DataFrame(season_metrics)
        metrics_df.to_csv(self.output_dir / 'season_comparison_metrics.csv', index=False)

        # Create visualizations
        self.create_comparison_plots(metrics_df, equity_curves)

        # Save best parameters summary
        params_summary = {}
        for season_key, params in self.best_params.items():
            params_summary[season_key] = {
                'season_name': self.seasons[season_key]['name'],
                'parameters': params
            }

        with open(self.output_dir / 'best_parameters_summary.yaml', 'w') as f:
            yaml.dump(params_summary, f, default_flow_style=False)

    def create_comparison_plots(self, metrics_df: pd.DataFrame, equity_curves: Dict[str, pd.DataFrame]):
        """Create comparative visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Political Season Strategy Performance Comparison', fontsize=16)

        # 1. Sharpe Ratio Comparison
        axes[0, 0].bar(metrics_df['season_name'], metrics_df['sharpe_ratio'])
        axes[0, 0].set_title('Sharpe Ratio by Season')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Total Return Comparison
        axes[0, 1].bar(metrics_df['season_name'], metrics_df['total_return'])
        axes[0, 1].set_title('Total Return by Season')
        axes[0, 1].set_ylabel('Total Return (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Win Rate Comparison
        axes[1, 0].bar(metrics_df['season_name'], metrics_df['win_rate'])
        axes[1, 0].set_title('Win Rate by Season')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Max Drawdown Comparison
        axes[1, 1].bar(metrics_df['season_name'], -metrics_df['max_drawdown'])  # Negative for visualization
        axes[1, 1].set_title('Max Drawdown by Season (Absolute)')
        axes[1, 1].set_ylabel('Max Drawdown (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'season_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Equity curves comparison
        plt.figure(figsize=(12, 8))

        for season_key, equity in equity_curves.items():
            season_name = self.seasons[season_key]['name']
            plt.plot(equity.index, equity['normalized_value'], label=season_name, linewidth=2)

        plt.title('Normalized Equity Curves by Political Season')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (Normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'season_equity_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Parameter heatmap
        self.create_parameter_heatmap()

    def create_parameter_heatmap(self):
        """Create heatmap showing parameter evolution across seasons."""
        param_data = []
        for season_key, params in self.best_params.items():
            row = {'season': self.seasons[season_key]['name']}
            row.update(params)
            param_data.append(row)

        if param_data:
            params_df = pd.DataFrame(param_data)
            params_df.set_index('season', inplace=True)

            plt.figure(figsize=(12, 8))
            sns.heatmap(params_df.T, annot=True, cmap='YlOrRd', fmt='.2f')
            plt.title('Optimal Parameters by Political Season')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'parameter_evolution_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

    def print_summary_report(self):
        """Print comprehensive summary report."""
        print("\n" + "="*80)
        print("POLITICAL SEASON ANALYSIS SUMMARY REPORT")
        print("="*80)

        for season_key, results in self.results.items():
            season_name = self.seasons[season_key]['name']
            metrics = results['performance_metrics']

            print(f"\n{season_name} ({season_key})")
            print("-" * 40)
            print(".2f")
            print(".2f")
            print(".2f")
            print(".1%")
            print(f"Total Trades: {metrics['total_trades']}")
            print(".2f")

            print(f"Best Parameters:")
            for param, value in self.best_params[season_key].items():
                print(f"  {param}: {value}")

        print(f"\nDetailed results saved to: {self.output_dir}")
        print("="*80)


def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = PoliticalSeasonOptimizer()

    # Run full analysis
    analyzer.run_full_analysis()

    # Print summary
    analyzer.print_summary_report()


if __name__ == "__main__":
    main()