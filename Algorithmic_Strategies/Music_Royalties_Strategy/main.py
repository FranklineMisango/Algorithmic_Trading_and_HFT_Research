"""
Main Entry Point for Music Royalties Strategy
End-to-end pipeline: data → features → model → portfolio → backtest → evaluation
"""

import argparse
import yaml
from pathlib import Path
import logging
import pandas as pd

from data_loader import load_and_prepare_data
from feature_engineering import engineer_all_features
from model_trainer import train_and_validate_model
from portfolio_constructor import PortfolioConstructor
from backtester import RoyaltyBacktester, prepare_universe_by_date
from performance_evaluator import evaluate_strategy_performance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """
    Main execution pipeline
    
    Args:
        args: Command line arguments
    """
    logger.info("="*80)
    logger.info("MUSIC ROYALTIES SYSTEMATIC TRADING STRATEGY")
    logger.info("="*80)
    
    # Load configuration
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"\nLoaded configuration from {config_path}")
    
    # Override config with command line args if provided
    if args.data_file:
        logger.info(f"Using data file: {args.data_file}")
    
    # =========================================================================
    # STEP 1: DATA LOADING & PREPROCESSING
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA LOADING & PREPROCESSING")
    logger.info("="*80)
    
    data_splits = load_and_prepare_data(config, filepath=args.data_file)
    
    logger.info(f"\nData splits:")
    logger.info(f"  Train:      {len(data_splits['train'])} transactions")
    logger.info(f"  Validation: {len(data_splits['validation'])} transactions")
    logger.info(f"  Test:       {len(data_splits['test'])} transactions")
    
    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("="*80)
    
    train_df = engineer_all_features(data_splits['train'], config, 
                                     include_interactions=args.include_interactions)
    val_df = engineer_all_features(data_splits['validation'], config,
                                   include_interactions=args.include_interactions)
    test_df = engineer_all_features(data_splits['test'], config,
                                    include_interactions=args.include_interactions)
    
    logger.info(f"\nEngineered {len(train_df.columns)} features")
    
    # =========================================================================
    # STEP 3: MODEL TRAINING & VALIDATION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: MODEL TRAINING & VALIDATION")
    logger.info("="*80)
    
    model, val_metrics = train_and_validate_model(train_df, val_df, config)
    
    logger.info(f"\nValidation Metrics:")
    logger.info(f"  MSE:  {val_metrics['mse']:.4f}")
    logger.info(f"  RMSE: {val_metrics['rmse']:.4f}")
    logger.info(f"  R²:   {val_metrics['r2']:.4f}")
    
    # Save model if requested
    if args.save_model:
        model_path = Path(args.output_dir) / "trained_model.pkl"
        model.save(str(model_path))
        logger.info(f"\nSaved model to {model_path}")
    
    # =========================================================================
    # STEP 4: PORTFOLIO CONSTRUCTION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: PORTFOLIO CONSTRUCTION")
    logger.info("="*80)
    
    # Calculate mispricing on test set
    test_df = model.calculate_mispricing(test_df)
    
    logger.info(f"\nMispricing Statistics:")
    logger.info(f"  Mean:   {test_df['mispricing'].mean():.3f}")
    logger.info(f"  Median: {test_df['mispricing'].median():.3f}")
    logger.info(f"  Undervalued assets: {(test_df['mispricing'] > 0).sum()}")
    
    # =========================================================================
    # STEP 5: BACKTESTING
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: BACKTESTING")
    logger.info("="*80)
    
    # Prepare universe by rebalancing dates
    universe_by_date = prepare_universe_by_date(
        test_df, 
        config['portfolio']['rebalancing_frequency']
    )
    
    logger.info(f"\nRebalancing {len(universe_by_date)} times ({config['portfolio']['rebalancing_frequency']})")
    
    # Run backtest
    backtester = RoyaltyBacktester(config)
    constructor = PortfolioConstructor(config)
    equity_curve = backtester.run_backtest(universe_by_date, model, constructor)
    
    # =========================================================================
    # STEP 6: PERFORMANCE EVALUATION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: PERFORMANCE EVALUATION")
    logger.info("="*80)
    
    trades_df = backtester.get_trades_df()
    performance = evaluate_strategy_performance(equity_curve, trades_df, config)
    
    # =========================================================================
    # STEP 7: SAVE RESULTS
    # =========================================================================
    if args.save_results:
        logger.info("\n" + "="*80)
        logger.info("STEP 7: SAVING RESULTS")
        logger.info("="*80)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save equity curve
        equity_path = output_dir / "equity_curve.csv"
        equity_curve.to_csv(equity_path, index=False)
        logger.info(f"Saved equity curve to {equity_path}")
        
        # Save trades
        if len(trades_df) > 0:
            trades_path = output_dir / "trades.csv"
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Saved trades to {trades_path}")
        
        # Save performance report
        report_path = output_dir / "performance_report.txt"
        with open(report_path, 'w') as f:
            f.write(performance['report'])
        logger.info(f"Saved performance report to {report_path}")
        
        # Save metrics as JSON
        import json
        metrics_path = output_dir / "performance_metrics.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
                return str(obj)
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        metrics_serializable = convert_to_serializable(performance)
        # Remove report from JSON (it's saved separately)
        metrics_serializable.pop('report', None)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    
    return performance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Music Royalties Systematic Trading Strategy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Path to data file (CSV). If not provided, uses synthetic data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results (equity curve, trades, reports)'
    )
    
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save trained model to disk'
    )
    
    parser.add_argument(
        '--include-interactions',
        action='store_true',
        help='Include interaction features in model'
    )
    
    args = parser.parse_args()
    
    # Run main pipeline
    performance = main(args)
