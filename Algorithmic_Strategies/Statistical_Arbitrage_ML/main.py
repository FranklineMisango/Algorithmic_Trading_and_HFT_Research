"""
Main execution script for Statistical Arbitrage Strategy

Orchestrates the entire workflow: data acquisition, feature engineering,
model training, portfolio construction, and backtesting.
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import sys
import pandas as pd
import numpy as np
from loguru import logger

# Import strategy modules
from src.data_acquisition import DataAcquisitionEngine
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.portfolio_builder import PortfolioBuilder
from src.backtester import Backtester


def setup_logging(config: dict) -> None:
    """
    Configure logging based on config settings.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get('logging', {})
    level = log_config.get('level', 'INFO')
    log_file = log_config.get('log_file', './logs/strategy.log')
    console_output = log_config.get('console_output', True)
    
    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger.remove()  # Remove default handler
    
    if console_output:
        logger.add(sys.stderr, level=level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    logger.add(log_file, level=level, rotation="10 MB", retention="30 days")
    
    logger.info(f"Logging configured: level={level}, file={log_file}")


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def run_backtest_workflow(config: dict, args: argparse.Namespace) -> None:
    """
    Execute complete backtesting workflow.
    
    Args:
        config: Configuration dictionary
        args: Command-line arguments
    """
    logger.info("=" * 80)
    logger.info("STATISTICAL ARBITRAGE BACKTEST - WORKFLOW START")
    logger.info("=" * 80)
    
    # Parse dates
    start_date = datetime.strptime(args.start_date or config['data']['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date or config['data']['end_date'], '%Y-%m-%d')
    
    logger.info(f"Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Step 1: Data Acquisition
    logger.info("\n" + "─" * 80)
    logger.info("STEP 1: DATA ACQUISITION")
    logger.info("─" * 80)
    
    data_engine = DataAcquisitionEngine(
        data_dir=config['data']['data_dir'],
        cache_enabled=config['data']['cache_enabled']
    )
    
    # Get universe
    universe = data_engine.get_russell_3000_universe()
    logger.info(f"Universe size: {len(universe)} stocks")
    
    # Download data (add buffer for feature calculation)
    data_start = start_date - timedelta(days=365)  # 1 year buffer
    df_raw = data_engine.get_training_data(
        tickers=universe,
        start_date=data_start,
        end_date=end_date,
        apply_filters=True
    )
    
    # Step 2: Feature Engineering
    logger.info("\n" + "─" * 80)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("─" * 80)
    
    feature_engineer = FeatureEngineer(
        momentum_periods=config['features']['momentum_periods'],
        ma_periods=config['features']['ma_periods'],
        volume_period=config['features']['volume_period']
    )
    
    df_features = feature_engineer.calculate_all_features(
        df_raw,
        target_horizons=config['features']['target_horizons']
    )
    
    # Prepare ML dataset
    target_col = f"forward_return_{config['features']['target_horizons'][0]}d"
    X, y = feature_engineer.prepare_ml_dataset(df_features, target_col=target_col)
    
    logger.info(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Step 3: Model Training
    logger.info("\n" + "─" * 80)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("─" * 80)
    
    trainer = ModelTrainer(
        model_type=config['model']['type'],
        model_dir=config['model']['model_dir'],
        rolling_window_years=config['model']['rolling_window_years'],
        retrain_frequency_days=config['model']['retrain_frequency_days']
    )
    
    # Get dates for features
    dates = df_features.index.get_level_values(0)
    
    # Train model (for demonstration, train on first 80% as single shot)
    # In production, use rolling_window_train
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    metrics = trainer.train(
        X_train, y_train, X_test, y_test,
        **config['model']['hyperparameters']
    )
    
    logger.info(f"Model trained: {config['model']['type']}")
    logger.info(f"Validation R²: {metrics.get('val_r2', 0):.4f}")
    
    # Feature importance
    importance = trainer.get_feature_importance(X.columns.tolist())
    logger.info(f"Top 5 features:\n{importance.head(5)}")
    
    # Step 4: Generate Predictions & Build Portfolios
    logger.info("\n" + "─" * 80)
    logger.info("STEP 4: PORTFOLIO CONSTRUCTION")
    logger.info("─" * 80)
    
    portfolio_builder = PortfolioBuilder(
        n_long=config['portfolio']['n_long'],
        n_short=config['portfolio']['n_short'],
        max_position_size=config['portfolio']['max_position_size'],
        total_capital=config['portfolio']['total_capital'],
        n_portfolios=config['portfolio']['n_portfolios']
    )
    
    # Generate predictions for backtest period
    # Filter to backtest period only
    backtest_mask = (dates >= start_date) & (dates <= end_date)
    X_backtest = X[backtest_mask]
    dates_backtest = dates[backtest_mask]
    
    predictions = trainer.predict(X_backtest)
    
    # Reconstruct ticker information
    df_features_backtest = df_features[backtest_mask]
    tickers_backtest = df_features_backtest.index.get_level_values('ticker')
    
    # Create predictions series
    pred_series = pd.Series(predictions, index=pd.MultiIndex.from_arrays([dates_backtest, tickers_backtest]))
    
    # Build portfolios for each unique date
    unique_dates = dates_backtest.unique()
    portfolios = []
    
    for date in unique_dates[::config['portfolio']['holding_period']]:  # Every N days
        # Get predictions and prices for this date
        try:
            date_mask = dates_backtest == date
            date_predictions = pred_series[date_mask]
            date_predictions = date_predictions.droplevel(0)  # Remove date level
            
            date_prices = df_features_backtest.loc[date, 'Close']
            
            if isinstance(date_prices, pd.DataFrame):
                date_prices = date_prices.iloc[:, 0]
            
            portfolio = portfolio_builder.build_portfolio(
                predictions=date_predictions,
                prices=date_prices,
                date=pd.Timestamp(date)
            )
            
            portfolios.append(portfolio)
            
        except Exception as e:
            logger.warning(f"Failed to build portfolio for {date}: {str(e)}")
            continue
    
    logger.info(f"Built {len(portfolios)} portfolios")
    
    # Step 5: Backtesting
    logger.info("\n" + "─" * 80)
    logger.info("STEP 5: BACKTESTING")
    logger.info("─" * 80)
    
    backtester = Backtester(
        initial_capital=config['portfolio']['total_capital'],
        transaction_cost_bps=config['backtest']['transaction_cost_bps'],
        holding_period=config['portfolio']['holding_period']
    )
    
    results = backtester.run_backtest(portfolios, df_raw)
    
    # Step 6: Results & Reporting
    logger.info("\n" + "─" * 80)
    logger.info("STEP 6: RESULTS & REPORTING")
    logger.info("─" * 80)
    
    # Generate report
    output_dir = Path(config['backtest']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = backtester.generate_report(
        results,
        output_path=output_dir / 'backtest_report.txt'
    )
    print(report)
    
    # Save results
    if config['backtest']['save_trades'] and not results['trades'].empty:
        results['trades'].to_csv(output_dir / 'trades.csv', index=False)
        logger.info(f"Trades saved to {output_dir / 'trades.csv'}")
    
    results['equity_curve'].to_csv(output_dir / 'equity_curve.csv')
    logger.info(f"Equity curve saved to {output_dir / 'equity_curve.csv'}")
    
    # Generate plots
    if config['backtest']['generate_plots']:
        backtester.plot_equity_curve(
            results['equity_curve'],
            save_path=output_dir / 'equity_curve.png'
        )
        
        if not results['equity_curve']['returns'].dropna().empty:
            backtester.plot_returns_distribution(
                results['equity_curve']['returns'].dropna(),
                save_path=output_dir / 'returns_distribution.png'
            )
    
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST WORKFLOW COMPLETE")
    logger.info("=" * 80)


def run_train_workflow(config: dict, args: argparse.Namespace) -> None:
    """
    Execute model training workflow only.
    
    Args:
        config: Configuration dictionary
        args: Command-line arguments
    """
    logger.info("Starting model training workflow...")
    
    # Implementation similar to backtest but only training
    # Left as exercise for production implementation
    logger.info("Training workflow not fully implemented. Use backtest mode.")


def run_predict_workflow(config: dict, args: argparse.Namespace) -> None:
    """
    Execute prediction workflow for live trading.
    
    Args:
        config: Configuration dictionary
        args: Command-line arguments
    """
    logger.info("Starting prediction workflow...")
    
    # Implementation for live predictions
    # Left as exercise for production implementation
    logger.info("Prediction workflow not fully implemented. Use backtest mode.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Statistical Arbitrage ML Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'train', 'predict'],
        default='backtest',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD) - overrides config'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD) - overrides config'
    )
    
    parser.add_argument(
        '--window-years',
        type=int,
        help='Training window in years (for train mode)'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Prediction date (YYYY-MM-DD) for predict mode'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Execute workflow based on mode
    if args.mode == 'backtest':
        run_backtest_workflow(config, args)
    elif args.mode == 'train':
        run_train_workflow(config, args)
    elif args.mode == 'predict':
        run_predict_workflow(config, args)


if __name__ == "__main__":
    main()
