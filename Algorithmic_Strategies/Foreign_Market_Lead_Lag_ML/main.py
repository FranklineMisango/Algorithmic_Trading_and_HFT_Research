"""
Main execution script for Foreign Market Lead-Lag ML Strategy.
Orchestrates data acquisition, feature engineering, model training, and backtesting.
"""

import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_acquisition import DataAcquisition
from feature_engineering import FeatureEngineering
from ml_models import MultiStockPredictor
from portfolio_constructor import PortfolioSimulator
from backtester import Backtester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    
    logger.info("="*60)
    logger.info("FOREIGN MARKET LEAD-LAG ML STRATEGY")
    logger.info("="*60)
    
    # Load configuration
    logger.info("Loading configuration...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    Path(config['output']['results_dir']).mkdir(exist_ok=True)
    Path(config['output']['models_dir']).mkdir(exist_ok=True)
    Path(config['output']['data_dir']).mkdir(exist_ok=True)
    
    # Step 1: Data Acquisition
    logger.info("\n" + "="*60)
    logger.info("STEP 1: DATA ACQUISITION")
    logger.info("="*60)
    
    data_acq = DataAcquisition(config)
    
    # Check if data already exists
    data_dir = config['output']['data_dir']
    if (Path(f"{data_dir}/sp500_daily_prices.csv").exists() and
        Path(f"{data_dir}/sp500_daily_returns.csv").exists() and
        Path(f"{data_dir}/foreign_weekly_returns.csv").exists()):
        
        logger.info("Loading existing data...")
        sp500_prices = pd.read_csv(f'{data_dir}/sp500_daily_prices.csv', 
                                   index_col=0, parse_dates=True)
        sp500_returns = pd.read_csv(f'{data_dir}/sp500_daily_returns.csv', 
                                    index_col=0, parse_dates=True)
        foreign_returns = pd.read_csv(f'{data_dir}/foreign_weekly_returns.csv', 
                                      index_col=0, parse_dates=True)
    else:
        logger.info("Downloading new data...")
        sp500_prices, sp500_returns, foreign_returns = data_acq.get_all_data()
        data_acq.save_data(sp500_prices, sp500_returns, foreign_returns, data_dir)
    
    logger.info(f"S&P 500 stocks: {sp500_returns.shape[1]}")
    logger.info(f"Foreign markets: {foreign_returns.shape[1]}")
    logger.info(f"Date range: {sp500_returns.index[0]} to {sp500_returns.index[-1]}")
    
    # Step 2: Feature Engineering
    logger.info("\n" + "="*60)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("="*60)
    
    feature_eng = FeatureEngineering(config)
    
    # Prepare data for all stocks
    logger.info("Preparing features for all stocks...")
    stock_data = feature_eng.prepare_all_stocks(foreign_returns, sp500_returns)
    
    logger.info(f"Prepared data for {len(stock_data)} stocks")
    
    # Step 3: Model Training & Validation
    logger.info("\n" + "="*60)
    logger.info("STEP 3: MODEL TRAINING & VALIDATION")
    logger.info("="*60)
    
    predictor = MultiStockPredictor(config)
    
    # Train models with validation
    logger.info("Training models with walk-forward validation...")
    validation_results = predictor.train_all_stocks(stock_data, validate=True)
    
    # Analyze validation results
    r2_scores = {stock: results['r2_oos'] 
                 for stock, results in validation_results.items()}
    r2_df = pd.Series(r2_scores).sort_values(ascending=False)
    
    logger.info(f"\nValidation Results Summary:")
    logger.info(f"  Stocks with positive R²_OOS: {(r2_df > 0).sum()} ({(r2_df > 0).mean():.1%})")
    logger.info(f"  Stocks with R²_OOS > 0.01: {(r2_df > 0.01).sum()} ({(r2_df > 0.01).mean():.1%})")
    logger.info(f"  Mean R²_OOS: {r2_df.mean():.4f}")
    logger.info(f"  Median R²_OOS: {r2_df.median():.4f}")
    logger.info(f"  Top 10 stocks by R²_OOS:")
    for stock, r2 in r2_df.head(10).items():
        logger.info(f"    {stock}: {r2:.4f}")
    
    # Save validation results
    r2_df.to_csv(f"{config['output']['results_dir']}/validation_r2_scores.csv")
    
    # Step 4: Generate Predictions
    logger.info("\n" + "="*60)
    logger.info("STEP 4: GENERATING PREDICTIONS")
    logger.info("="*60)
    
    # Create feature matrix for all dates
    lagged_features = feature_eng.create_lagged_features(foreign_returns)
    lagged_features = feature_eng.winsorize_features(lagged_features)
    lagged_features = feature_eng.standardize_features(lagged_features)
    
    # Align with daily frequency
    features_daily = lagged_features.reindex(sp500_returns.index, method='ffill')
    
    # Generate predictions for each day
    logger.info("Generating daily predictions...")
    predictions_list = []
    
    for date in features_daily.index:
        if date not in features_daily.dropna().index:
            continue
        
        features_row = features_daily.loc[date:date]
        
        try:
            predictions = predictor.predict_all_stocks(features_row)
            predictions.name = date
            predictions_list.append(predictions)
        except Exception as e:
            logger.debug(f"Error predicting for {date}: {e}")
            continue
    
    predictions_df = pd.DataFrame(predictions_list)
    logger.info(f"Generated predictions for {len(predictions_df)} days")
    
    # Save predictions
    predictions_df.to_csv(f"{config['output']['results_dir']}/predictions.csv")
    
    # Step 5: Portfolio Construction & Simulation
    logger.info("\n" + "="*60)
    logger.info("STEP 5: PORTFOLIO SIMULATION")
    logger.info("="*60)
    
    simulator = PortfolioSimulator(config)
    results_df = simulator.simulate(predictions_df, sp500_returns)
    
    logger.info(f"Simulated {len(results_df)} trading days")
    
    # Save results
    results_df.to_csv(f"{config['output']['results_dir']}/portfolio_results.csv")
    
    # Step 6: Backtesting & Performance Analysis
    logger.info("\n" + "="*60)
    logger.info("STEP 6: PERFORMANCE ANALYSIS")
    logger.info("="*60)
    
    # Download benchmark data
    import yfinance as yf
    benchmark = yf.download(config['backtesting']['benchmark'], 
                           start=config['data']['start_date'],
                           end=config['data']['end_date'],
                           progress=False)['Close']
    benchmark_returns = benchmark.pct_change()
    
    # Run backtest
    backtester = Backtester(config)
    metrics = backtester.run_backtest(results_df, benchmark_returns)
    
    # Save metrics
    metrics_df = pd.Series(metrics)
    metrics_df.to_csv(f"{config['output']['results_dir']}/performance_metrics.csv")
    
    logger.info("\n" + "="*60)
    logger.info("STRATEGY EXECUTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to {config['output']['results_dir']}/")
    
    return metrics


if __name__ == "__main__":
    metrics = main()
