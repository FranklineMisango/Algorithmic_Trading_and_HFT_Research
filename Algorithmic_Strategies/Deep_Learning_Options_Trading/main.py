"""
Main Pipeline for Deep Learning Options Trading Strategy

Orchestrates the complete workflow: data acquisition, feature engineering,
model training, and backtesting with benchmark comparisons.
"""

import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime

from data_acquisition import OptionsDataAcquisition
from feature_engineering import OptionsFeatureEngineer
from lstm_model import DeepLearningOptionsTrader
from backtester import OptionsBacktester

class DeepLearningOptionsPipeline:
    """
    Main pipeline class that coordinates all components of the
    deep learning options trading strategy.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Initialize components
        self.data_acquirer = OptionsDataAcquisition(config_path)
        self.feature_engineer = OptionsFeatureEngineer(config_path)
        self.model_trader = DeepLearningOptionsTrader(config_path)
        self.backtester = OptionsBacktester(config_path)

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler()
            ]
        )

    def run_data_pipeline(self, start_date: str = None, end_date: str = None):
        """Run data acquisition and feature engineering."""
        self.logger.info("Starting data pipeline")

        # Acquire data
        prices_df, options_df = self.data_acquirer.fetch_full_dataset(start_date, end_date)

        # Engineer features
        features_df = self.feature_engineer.create_feature_matrix(options_df, prices_df)

        # Create sequential data for LSTM
        X, y, metadata = self.feature_engineer.create_sequential_data(features_df)

        # Save processed data
        self._save_processed_data(X, y, metadata, features_df)

        self.logger.info("Data pipeline completed")
        return X, y, metadata

    def run_model_training(self, X: np.ndarray = None, y: np.ndarray = None,
                          metadata: list = None):
        """Run model training pipeline."""
        self.logger.info("Starting model training")

        if X is None or y is None:
            # Load from disk
            X, y, metadata = self._load_processed_data()

        # Perform walk-forward validation
        validation_results = self.model_trader.walk_forward_validation(X, y)

        # Train final model on full dataset
        self.model_trader.train_model(X, y, metadata_train=metadata)

        # Save model
        self.model_trader._save_model("models/final_model.pth")

        self.logger.info("Model training completed")
        return validation_results

    def run_backtesting(self, positions_df: pd.DataFrame = None,
                       prices_df: pd.DataFrame = None, options_df: pd.DataFrame = None):
        """Run backtesting pipeline."""
        self.logger.info("Starting backtesting")

        if positions_df is None:
            # Generate positions from trained model
            positions_df = self._generate_model_positions()

        if prices_df is None or options_df is None:
            # Load from disk
            prices_df, options_df = self._load_raw_data()

        # Run backtest
        results = self.backtester.run_backtest(positions_df, prices_df, options_df)

        # Save results
        self._save_results(results)

        # Plot results
        self.backtester.plot_results()

        self.logger.info("Backtesting completed")
        return results

    def run_full_pipeline(self, start_date: str = None, end_date: str = None):
        """Run the complete end-to-end pipeline."""
        self.logger.info("Starting full pipeline execution")

        # 1. Data Pipeline
        X, y, metadata = self.run_data_pipeline(start_date, end_date)

        # 2. Model Training
        validation_results = self.run_model_training(X, y, metadata)

        # 3. Backtesting
        results = self.run_backtesting()

        # 4. Generate Report
        self._generate_report(results, validation_results)

        self.logger.info("Full pipeline completed successfully")
        return results

    def _generate_model_positions(self) -> pd.DataFrame:
        """Generate position signals from trained model."""
        self.logger.info("Generating model positions")

        # Load trained model
        try:
            self.model_trader.load_model("models/final_model.pth")
        except FileNotFoundError:
            raise ValueError("No trained model found. Run training first.")

        # Load processed data
        X, y, metadata = self._load_processed_data()

        # Generate predictions
        position_signals = self.model_trader.predict_positions(X)

        # Create positions DataFrame
        positions_data = []
        for i, signal in enumerate(position_signals):
            meta = metadata[i]
            positions_data.append({
                'date': meta['date'],
                'ticker': meta['ticker'],
                'position_signal': signal,
                'straddle_price': meta['straddle_price']
            })

        positions_df = pd.DataFrame(positions_data)
        positions_df['date'] = pd.to_datetime(positions_df['date'])

        # Save positions
        positions_df.to_csv("data/positions/model_positions.csv", index=False)

        return positions_df

    def _load_raw_data(self):
        """Load raw data from disk."""
        try:
            prices_df = pd.read_csv("data/underlying_prices/underlying_prices.csv")
            prices_df['Date'] = pd.to_datetime(prices_df['Date'])
            prices_df = prices_df.set_index(['ticker', 'Date'])

            options_df = pd.read_csv("data/options_data/options_data.csv")
            options_df['date'] = pd.to_datetime(options_df['date'])
            options_df['expiry'] = pd.to_datetime(options_df['expiry'])

            return prices_df, options_df

        except FileNotFoundError:
            raise ValueError("Raw data not found. Run data pipeline first.")

    def _load_processed_data(self):
        """Load processed sequential data."""
        try:
            import numpy as np
            X = np.load("data/processed/X_sequences.npy")
            y = np.load("data/processed/y_targets.npy")

            # Load metadata
            metadata_df = pd.read_csv("data/processed/metadata.csv")
            metadata = metadata_df.to_dict('records')

            return X, y, metadata

        except FileNotFoundError:
            raise ValueError("Processed data not found. Run data pipeline first.")

    def _save_processed_data(self, X: np.ndarray, y: np.ndarray,
                           metadata: list, features_df: pd.DataFrame):
        """Save processed data to disk."""
        # Create directories
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.save("data/processed/X_sequences.npy", X)
        np.save("data/processed/y_targets.npy", y)

        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv("data/processed/metadata.csv", index=False)

        # Save features
        features_df.to_csv("data/processed/features.csv", index=False)

    def _save_results(self, results: dict):
        """Save backtest results to disk."""
        import json

        Path("results").mkdir(exist_ok=True)

        # Save as JSON
        with open("results/backtest_results.json", 'w') as f:
            # Convert numpy types to native Python types
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif isinstance(value, (np.float64, np.float32)):
                    json_results[key] = float(value)
                elif isinstance(value, (np.int64, np.int32)):
                    json_results[key] = int(value)
                else:
                    json_results[key] = value

            json.dump(json_results, f, indent=2, default=str)

    def _generate_report(self, backtest_results: dict, validation_results: dict):
        """Generate comprehensive performance report."""
        self.logger.info("Generating performance report")

        report = f"""
# Deep Learning Options Trading Strategy - Performance Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Strategy Overview
This report presents the results of a deep learning-based options trading strategy
that uses LSTM neural networks to predict optimal positions in delta-neutral straddles
on S&P 100 constituents.

## Model Validation Results
Walk-forward validation Sharpe ratios: {validation_results.get('sharpe_ratios', [])}

## Backtest Performance

### Key Metrics
- Total Return: {backtest_results.get('total_return', 'N/A'):.2%}
- Annual Return: {backtest_results.get('annual_return', 'N/A'):.2%}
- Annual Volatility: {backtest_results.get('annual_volatility', 'N/A'):.2%}
- Sharpe Ratio: {backtest_results.get('sharpe_ratio', 'N/A'):.2f}
- Maximum Drawdown: {backtest_results.get('max_drawdown', 'N/A'):.2%}
- Win Rate: {backtest_results.get('win_rate', 'N/A'):.2%}
- Total Trades: {backtest_results.get('total_trades', 'N/A')}

### Benchmark Comparison
"""

        # Add benchmark results
        benchmarks = backtest_results.get('benchmarks', {})
        for name, metrics in benchmarks.items():
            report += f"- {name.title()}: Sharpe = {metrics.get('sharpe_ratio', 'N/A'):.2f}\n"

        report += f"- LSTM Strategy: Sharpe = {backtest_results.get('sharpe_ratio', 'N/A'):.2f}\n"

        report += """
## Risk Management
- Maximum position size: {}% of portfolio
- Maximum single stock exposure: {}%
- Drawdown stop loss: {}%
- Transaction costs: ${} per contract + {}% bid-ask spread

## Implementation Notes
- Model trained with Sharpe ratio optimization and turnover regularization
- Walk-forward validation used to prevent overfitting
- Conservative slippage assumptions
- Survivorship bias controlled via point-in-time S&P 100 composition

## Next Steps
1. Paper trading validation
2. Live trading with small capital allocation
3. Continuous model retraining and monitoring
4. Capacity analysis for larger capital deployment
""".format(
            self.config['backtest']['max_position_size'] * 100,
            self.config['backtest']['max_single_stock_exposure'] * 100,
            self.config['backtest']['max_drawdown_stop'] * 100,
            self.config['backtest']['transaction_cost_per_contract'],
            self.config['backtest']['bid_ask_spread'] * 100
        )

        # Save report
        with open("results/performance_report.md", 'w') as f:
            f.write(report)

        self.logger.info("Performance report saved to results/performance_report.md")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description='Deep Learning Options Trading Strategy')
    parser.add_argument('--mode', choices=['data', 'train', 'backtest', 'full'],
                       default='full', help='Pipeline mode to run')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--start-date', help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for data (YYYY-MM-DD)')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = DeepLearningOptionsPipeline(args.config)

    try:
        if args.mode == 'data':
            pipeline.run_data_pipeline(args.start_date, args.end_date)
        elif args.mode == 'train':
            pipeline.run_model_training()
        elif args.mode == 'backtest':
            pipeline.run_backtesting()
        elif args.mode == 'full':
            pipeline.run_full_pipeline(args.start_date, args.end_date)

        print("Pipeline execution completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()