"""
Main Training Pipeline
=====================

Orchestrates the entire ML pipeline with proper data handling and evaluation.
"""

import argparse
import logging
from pathlib import Path

from src.utils import load_config, setup_logging, set_random_seeds, create_directory
from src.data_processor import DataProcessor
from src.feature_engine import FeatureEngine
from src.model_trainer import ModelTrainer
from src.backtester import Backtester


def main(config_path: str = "config.yaml", data_path: str = None):
    """
    Main training pipeline.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to data file (overrides config)
    """
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)
    set_random_seeds(config)
    
    logger.info("=" * 80)
    logger.info("FUTURES PRICE PREDICTION - ML TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Initialize components
    data_processor = DataProcessor(config)
    feature_engine = FeatureEngine(config)
    model_trainer = ModelTrainer(config)
    backtester = Backtester(config)
    
    # Step 1: Load and validate data
    logger.info("\n[Step 1/7] Loading and validating data...")
    df = data_processor.load_data(data_path)
    df = data_processor.validate_data(df)
    
    # Step 2: Engineer features
    logger.info("\n[Step 2/7] Engineering features...")
    df = feature_engine.engineer_features(df)
    
    # Step 3: Create targets (NO DATA LEAKAGE)
    logger.info("\n[Step 3/7] Creating targets...")
    df_clean, y_reg, y_clf = data_processor.create_targets(df, look_ahead=1)
    
    # Step 4: Time-aware train/test split
    logger.info("\n[Step 4/7] Splitting data (time-aware)...")
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, split_idx = \
        data_processor.time_aware_split(df_clean, y_reg, y_clf)
    
    # Step 5: Feature selection and preprocessing
    logger.info("\n[Step 5/7] Feature selection...")
    X_train_selected, X_test_selected = feature_engine.select_features(
        X_train, y_train_reg, X_test
    )
    
    # Handle class imbalance for classification
    X_train_clf, y_train_clf_resampled = feature_engine.handle_class_imbalance(
        X_train_selected, y_train_clf
    )
    
    # Step 6: Train models
    logger.info("\n[Step 6/7] Training models...")
    
    # XGBoost Classifier
    logger.info("\n--- Training XGBoost Classifier ---")
    xgb_clf = model_trainer.train_xgboost_classifier(X_train_clf, y_train_clf_resampled)
    clf_metrics = model_trainer.evaluate_classification(xgb_clf, X_test_selected, y_test_clf, 
                                                        "XGBoost_Classifier")
    
    # XGBoost Regressor
    logger.info("\n--- Training XGBoost Regressor ---")
    xgb_reg = model_trainer.train_xgboost_regressor(X_train_selected, y_train_reg)
    reg_metrics = model_trainer.evaluate_regression(xgb_reg, X_test_selected, y_test_reg,
                                                    "XGBoost_Regressor")
    
    # Random Forest
    logger.info("\n--- Training Random Forest ---")
    rf_clf = model_trainer.train_random_forest(X_train_clf, y_train_clf_resampled)
    rf_metrics = model_trainer.evaluate_classification(rf_clf, X_test_selected, y_test_clf,
                                                       "Random_Forest")
    
    # Ensemble
    logger.info("\n--- Training Ensemble ---")
    ensemble = model_trainer.train_ensemble(X_train_clf, y_train_clf_resampled)
    if ensemble:
        ensemble_metrics = model_trainer.evaluate_classification(ensemble, X_test_selected, y_test_clf,
                                                                 "Ensemble")
    
    # SHAP explanations
    if config.get("evaluation", {}).get("use_shap", False):
        logger.info("\n--- Generating SHAP Explanations ---")
        model_trainer.explain_model_shap(xgb_reg, X_test_selected, "XGBoost_Regressor")
    
    # Step 7: Backtesting
    logger.info("\n[Step 7/7] Running backtest...")
    predictions = xgb_reg.predict(X_test_selected)
    backtest_results = backtester.backtest_strategy(predictions, y_test_reg.values)
    
    # Save models
    logger.info("\nSaving models...")
    model_trainer.save_model(xgb_clf, "xgb_classifier")
    model_trainer.save_model(xgb_reg, "xgb_regressor")
    model_trainer.save_model(rf_clf, "random_forest")
    if ensemble:
        model_trainer.save_model(ensemble, "ensemble")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nBest Classification Model: XGBoost (AUC: {clf_metrics.get('roc_auc', 0):.4f})")
    logger.info(f"Best Regression Model: XGBoost (Directional Accuracy: {reg_metrics.get('directional_accuracy', 0):.4f})")
    logger.info(f"\nBacktest Performance:")
    logger.info(f"  Total Return: {backtest_results['metrics']['total_return_pct']:.2f}%")
    logger.info(f"  Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.4f}")
    logger.info(f"  Max Drawdown: {backtest_results['metrics']['max_drawdown_pct']:.2f}%")
    
    return {
        'models': model_trainer.models,
        'metrics': model_trainer.results,
        'backtest': backtest_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train futures price prediction models")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to data file (overrides config)")
    
    args = parser.parse_args()
    
    main(config_path=args.config, data_path=args.data)
