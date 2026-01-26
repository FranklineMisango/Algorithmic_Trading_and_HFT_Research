import yaml
import sys
from datetime import datetime
from loguru import logger
sys.path.append('src')

from data_acquisition import DataAcquisition
from signal_generator import SignalGenerator
from backtester import Backtester

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting Perpetual Futures Funding Arbitrage Strategy")
    
    # Initialize components
    data_acq = DataAcquisition(config)
    signal_gen = SignalGenerator(config)
    backtester = Backtester(config)
    
    # Date range
    start_date = datetime.strptime(config['backtest']['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(config['backtest']['end_date'], '%Y-%m-%d')
    train_end = datetime.strptime(config['backtest']['train_end'], '%Y-%m-%d')
    
    # Process each asset
    for asset in config['assets']:
        symbol = asset['perp_symbol']
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*60}")
        
        # Fetch data
        df = data_acq.prepare_dataset(symbol, start_date, end_date)
        
        # Generate signals
        signals = signal_gen.generate_signals(df)
        
        # Validate bounds
        signal_gen.validate_bounds(df, signals)
        
        # Split data
        train_df = df[df.index < train_end]
        test_df = df[df.index >= train_end]
        train_signals = signals[signals.index < train_end]
        test_signals = signals[signals.index >= train_end]
        
        # In-sample backtest
        logger.info("\n--- In-Sample Backtest ---")
        train_results, train_trades = backtester.run_backtest(train_df, train_signals)
        train_metrics = backtester.calculate_metrics(train_results, train_trades)
        
        for metric, value in train_metrics.items():
            logger.info(f"{metric}: {value:.2f}")
        
        # Out-of-sample backtest
        logger.info("\n--- Out-of-Sample Backtest ---")
        test_results, test_trades = backtester.run_backtest(test_df, test_signals)
        test_metrics = backtester.calculate_metrics(test_results, test_trades)
        
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.2f}")
        
        # Strategy selection criteria
        if (test_metrics['Sharpe Ratio'] > 1.5 and 
            test_metrics['Max Drawdown (%)'] > -10 and 
            test_metrics['Win Rate (%)'] > 60):
            logger.success(f"✓ {symbol} meets selection criteria")
        else:
            logger.warning(f"✗ {symbol} does not meet selection criteria")

if __name__ == "__main__":
    main()
