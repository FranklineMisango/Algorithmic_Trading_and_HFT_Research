"""
Main Pipeline for Crypto Macro-Fundamental Strategy

Orchestrates data acquisition, feature engineering, model training, and backtesting.
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_acquisition import DataAcquisition
from feature_engineering import FeatureEngineer
from ml_model import CryptoMLModel
from backtester import CryptoBacktester


class CryptoMacroFundamentalPipeline:
    """Main pipeline for crypto macro-fundamental strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline."""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_acq = DataAcquisition(config_path)
        self.engineer = FeatureEngineer(config_path)
        self.model = CryptoMLModel(config_path)
        self.backtester = CryptoBacktester(config_path)
        
        # Create output directory
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def run_data_pipeline(self) -> dict:
        """
        Run data acquisition pipeline.
        
        Returns:
            Dataset dict
        """
        print("\n" + "="*60)
        print("DATA ACQUISITION PIPELINE")
        print("="*60)
        
        dataset = self.data_acq.fetch_full_dataset()
        
        print(f"\nData loaded from {dataset['prices'].index[0]} to {dataset['prices'].index[-1]}")
        print(f"Total days: {len(dataset['prices'])}")
        print(f"Columns: {list(dataset['prices'].columns)}")
        print(f"\nInstitutional events: {len(dataset['events'])}")
        
        # Save to CSV
        dataset['prices'].to_csv(self.output_dir / "prices.csv")
        dataset['events'].to_csv(self.output_dir / "events.csv")
        print(f"\nData saved to {self.output_dir}/")
        
        return dataset
    
    def run_feature_pipeline(self, dataset: dict) -> tuple:
        """
        Run feature engineering pipeline.
        
        Args:
            dataset: Dataset dict from data pipeline
        
        Returns:
            (features, target) tuple
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        features = self.engineer.engineer_all_features(dataset['prices'], dataset['events'])
        target = self.engineer.create_target_variable(dataset['prices'])
        
        print(f"\nFeatures generated: {len(features.columns)}")
        print(f"Feature names: {list(features.columns)}")
        print(f"Target variable: {target.name}")
        print(f"Valid samples: {len(features)}")
        
        # Feature correlations with target
        correlations = {}
        for col in features.columns:
            aligned_features = features[col].dropna()
            aligned_target = target.reindex(aligned_features.index).dropna()
            common_idx = aligned_features.index.intersection(aligned_target.index)
            
            if len(common_idx) > 0:
                corr = np.corrcoef(
                    aligned_features[common_idx],
                    aligned_target[common_idx]
                )[0, 1]
                correlations[col] = corr
        
        print("\nTop 5 features by correlation with target:")
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feature, corr in sorted_corr:
            print(f"  {feature}: {corr:.4f}")
        
        # Save to CSV
        features.to_csv(self.output_dir / "features.csv")
        target.to_csv(self.output_dir / "target.csv")
        print(f"\nFeatures saved to {self.output_dir}/")
        
        return features, target
    
    def run_training_pipeline(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        optimize_hyperparams: bool = False
    ) -> dict:
        """
        Run model training pipeline.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            optimize_hyperparams: Whether to optimize hyperparameters
        
        Returns:
            ML results dict
        """
        print("\n" + "="*60)
        print("MODEL TRAINING PIPELINE")
        print("="*60)
        
        if optimize_hyperparams:
            print("\nRunning hyperparameter optimization (this may take a while)...")
        
        ml_results = self.model.walk_forward_validation(
            features,
            target,
            optimize_hyperparams=optimize_hyperparams
        )
        
        print(f"\nOverall Sharpe Ratio: {ml_results['overall_sharpe']:.4f}")
        print(f"Overall MSE: {ml_results['overall_mse']:.6f}")
        
        print("\nPer-fold results:")
        for fold in ml_results['fold_results']:
            print(f"  Fold {fold['fold']}: Sharpe={fold['sharpe']:.4f}, "
                  f"MSE={fold['mse']:.6f}, Period={fold['test_period']}")
        
        # Save results
        ml_results['results_df'].to_csv(self.output_dir / "predictions.csv")
        
        # Save fold results
        fold_df = pd.DataFrame(ml_results['fold_results'])
        fold_df.to_csv(self.output_dir / "fold_results.csv", index=False)
        
        print(f"\nResults saved to {self.output_dir}/")
        
        return ml_results
    
    def run_backtest_pipeline(
        self,
        ml_results: dict,
        dataset: dict,
        features: pd.DataFrame
    ) -> dict:
        """
        Run backtest pipeline.
        
        Args:
            ml_results: ML results from training
            dataset: Dataset dict
            features: Feature DataFrame
        
        Returns:
            Backtest results dict
        """
        print("\n" + "="*60)
        print("BACKTEST PIPELINE")
        print("="*60)
        
        # Strategy backtest
        print("\nRunning strategy backtest...")
        strategy_results = self.backtester.run_backtest(
            ml_results['results_df'],
            dataset['prices'],
            features
        )
        
        print("\nStrategy Performance:")
        for metric, value in strategy_results['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Buy-and-hold benchmark
        print("\nRunning buy-and-hold benchmark...")
        start_date = ml_results['results_df'].index[0]
        end_date = ml_results['results_df'].index[-1]
        
        benchmark_results = self.backtester.backtest_buy_and_hold(
            dataset['prices'],
            start_date,
            end_date
        )
        
        print("\nBuy-and-Hold Performance:")
        for metric, value in benchmark_results['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Save results
        strategy_results['backtest_df'].to_csv(self.output_dir / "strategy_backtest.csv")
        benchmark_results['backtest_df'].to_csv(self.output_dir / "benchmark_backtest.csv")
        
        # Comparison
        comparison = pd.DataFrame({
            'Strategy': strategy_results['metrics'],
            'Buy_and_Hold': benchmark_results['metrics']
        })
        comparison.to_csv(self.output_dir / "performance_comparison.csv")
        
        print(f"\nBacktest results saved to {self.output_dir}/")
        
        return {
            'strategy': strategy_results,
            'benchmark': benchmark_results
        }
    
    def run_hyperopt_pipeline(self, features: pd.DataFrame, target: pd.Series) -> dict:
        """
        Run standalone hyperparameter optimization.
        
        Args:
            features: Feature DataFrame
            target: Target Series
        
        Returns:
            Best parameters
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION PIPELINE")
        print("="*60)
        
        # Align data
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx].values
        y = target.loc[common_idx].values
        
        # Split for validation (last 20%)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"\nTrain samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Optimize
        print("\nRunning Bayesian optimization...")
        best_params = self.model.hyperparameter_optimization(
            X_train, y_train, X_val, y_val,
            n_trials=self.config['model']['hyperparameter_optimization']['n_trials']
        )
        
        print("\nBest hyperparameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Save
        import json
        with open(self.output_dir / "best_params.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\nBest parameters saved to {self.output_dir}/best_params.json")
        
        return best_params
    
    def run_comparison_pipeline(self, backtest_results: dict) -> None:
        """
        Run comparison and statistical tests.
        
        Args:
            backtest_results: Backtest results dict
        """
        print("\n" + "="*60)
        print("COMPARISON PIPELINE")
        print("="*60)
        
        strategy_returns = backtest_results['strategy']['backtest_df']['net_return'].values
        benchmark_returns = backtest_results['benchmark']['backtest_df']['net_return'].values
        
        # Statistical test
        print("\nRunning statistical tests...")
        test_results = self.backtester.statistical_tests(
            strategy_returns,
            benchmark_returns
        )
        
        print(f"\nOne-sided t-test (Strategy > Benchmark):")
        print(f"  t-statistic: {test_results['t_statistic']:.4f}")
        print(f"  p-value: {test_results['p_value']:.4f}")
        print(f"  Significant (α=0.05): {test_results['is_significant']}")
        print(f"  Mean difference: {test_results['mean_diff']:.6f}")
        
        # Capacity analysis
        print("\nRunning capacity analysis...")
        capacity_df = self.backtester.capacity_analysis(
            backtest_results['strategy']['backtest_df']
        )
        
        print("\nCapacity Analysis:")
        print(capacity_df.to_string(index=False))
        
        capacity_df.to_csv(self.output_dir / "capacity_analysis.csv", index=False)
        
        print(f"\nComparison results saved to {self.output_dir}/")
    
    def run_full_pipeline(self, optimize_hyperparams: bool = False) -> None:
        """
        Run full end-to-end pipeline.
        
        Args:
            optimize_hyperparams: Whether to optimize hyperparameters
        """
        print("\n" + "="*80)
        print("CRYPTO MACRO-FUNDAMENTAL STRATEGY - FULL PIPELINE")
        print("="*80)
        
        # 1. Data
        dataset = self.run_data_pipeline()
        
        # 2. Features
        features, target = self.run_feature_pipeline(dataset)
        
        # 3. Training
        ml_results = self.run_training_pipeline(features, target, optimize_hyperparams)
        
        # 4. Backtest
        backtest_results = self.run_backtest_pipeline(ml_results, dataset, features)
        
        # 5. Comparison
        self.run_comparison_pipeline(backtest_results)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"\nAll results saved to {self.output_dir}/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crypto Macro-Fundamental Strategy Pipeline"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['data', 'features', 'train', 'backtest', 'hyperopt', 'compare', 'full'],
        help='Pipeline mode to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--optimize-hyperparams',
        action='store_true',
        help='Optimize hyperparameters during training'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CryptoMacroFundamentalPipeline(args.config)
    
    if args.mode == 'data':
        pipeline.run_data_pipeline()
    
    elif args.mode == 'features':
        dataset = pipeline.run_data_pipeline()
        pipeline.run_feature_pipeline(dataset)
    
    elif args.mode == 'train':
        dataset = pipeline.run_data_pipeline()
        features, target = pipeline.run_feature_pipeline(dataset)
        pipeline.run_training_pipeline(features, target, args.optimize_hyperparams)
    
    elif args.mode == 'backtest':
        dataset = pipeline.run_data_pipeline()
        features, target = pipeline.run_feature_pipeline(dataset)
        ml_results = pipeline.run_training_pipeline(features, target, args.optimize_hyperparams)
        pipeline.run_backtest_pipeline(ml_results, dataset, features)
    
    elif args.mode == 'hyperopt':
        dataset = pipeline.run_data_pipeline()
        features, target = pipeline.run_feature_pipeline(dataset)
        pipeline.run_hyperopt_pipeline(features, target)
    
    elif args.mode == 'compare':
        dataset = pipeline.run_data_pipeline()
        features, target = pipeline.run_feature_pipeline(dataset)
        ml_results = pipeline.run_training_pipeline(features, target, args.optimize_hyperparams)
        backtest_results = pipeline.run_backtest_pipeline(ml_results, dataset, features)
        pipeline.run_comparison_pipeline(backtest_results)
    
    elif args.mode == 'full':
        pipeline.run_full_pipeline(args.optimize_hyperparams)


if __name__ == "__main__":
    main()
