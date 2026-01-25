"""
Main Orchestration Script for AI-Enhanced 60/40 Portfolio

This script coordinates all modules to run the complete AI-driven
portfolio allocation strategy.
"""

import yaml
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_acquisition import DataAcquisition
from feature_engineering import FeatureEngineer
from ml_model import PortfolioMLModel
from backtester import PortfolioBacktester


class AIPortfolioStrategy:
    """Main class to orchestrate the AI-enhanced portfolio strategy."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the strategy.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_acq = DataAcquisition(self.config)
        self.feature_eng = FeatureEngineer(self.config)
        self.ml_model = PortfolioMLModel(self.config)
        self.backtester = PortfolioBacktester(self.config)
        
        # Data storage
        self.prices = None
        self.returns = None
        self.indicators = None
        self.features = None
        self.allocations = None
        self.backtest_results = None
        
        # Create output directories
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Create output directories for results."""
        for dir_name in ['results', 'figures', 'models']:
            dir_path = self.config['output'][f'{dir_name}_dir']
            os.makedirs(dir_path, exist_ok=True)
    
    def load_data(self):
        """Load and prepare all data."""
        print("\n" + "="*60)
        print("STEP 1: LOADING DATA")
        print("="*60)
        
        self.prices, self.returns, self.indicators = self.data_acq.get_full_dataset()
        
        print(f"\nData loaded successfully")
        print(f"  - Price data: {self.prices.shape}")
        print(f"  - Returns data: {self.returns.shape}")
        print(f"  - Indicators data: {self.indicators.shape}")
        print(f"  - Date range: {self.prices.index[0]} to {self.prices.index[-1]}")
    
    def engineer_features(self):
        """Engineer features from indicators."""
        print("\n" + "="*60)
        print("STEP 2: ENGINEERING FEATURES")
        print("="*60)
        
        # Create all features
        features_raw = self.feature_eng.engineer_all_features(self.indicators)
        
        # Prepare for training
        self.features = self.feature_eng.prepare_features_for_training(features_raw)
        
        print(f"\nFeatures engineered successfully")
        print(f"  - Raw features: {features_raw.shape}")
        print(f"  - Prepared features: {self.features.shape}")
        print(f"  - Number of feature columns: {len(self.features.columns)}")
    
    def train_models(self):
        """Train ML models for each asset."""
        print("\n" + "="*60)
        print("STEP 3: TRAINING ML MODELS")
        print("="*60)
        
        # Create target variables (next period returns)
        targets = self.ml_model.create_target_variables(self.returns, lookback=1)
        
        # Prepare train/test split
        X_train, X_test, y_train, y_test = self.ml_model.prepare_train_test_data(
            self.features, 
            targets,
            test_size=self.config['model']['validation']['test_size']
        )
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train models
        self.ml_model.train_all_models(X_train, y_train)
        
        # Evaluate models
        evaluation_results = self.ml_model.evaluate_all_models(X_test, y_test)
        
        # Save evaluation results
        eval_path = os.path.join(self.config['output']['results_dir'], 'model_evaluation.csv')
        evaluation_results.to_csv(eval_path)
        print(f"\nEvaluation results saved to {eval_path}")
        
        # Save models
        self.ml_model.save_models(self.config['output']['models_dir'])
        
        # Store for later use
        self.X_test = X_test
        self.y_test = y_test
    
    def generate_allocations(self):
        """Generate portfolio allocations using trained models."""
        print("\n" + "="*60)
        print("STEP 4: GENERATING PORTFOLIO ALLOCATIONS")
        print("="*60)
        
        # Predict returns for entire dataset
        predicted_returns = self.ml_model.predict_returns(self.features)
        
        # Calculate optimal allocations
        self.allocations = self.ml_model.calculate_optimal_allocations(predicted_returns)
        
        print(f"\nAllocations generated successfully")
        print(f"  - Allocations shape: {self.allocations.shape}")
        
        # Save allocations
        if self.config['output']['save_allocations']:
            alloc_path = os.path.join(self.config['output']['results_dir'], 'allocations.csv')
            self.allocations.to_csv(alloc_path)
            print(f"  - Saved to: {alloc_path}")
        
        # Print allocation summary
        print(f"\nAverage Allocations:")
        avg_alloc = self.allocations.mean()
        for asset, alloc in avg_alloc.items():
            print(f"  {asset}: {alloc:.2%}")
    
    def run_backtest(self):
        """Run backtest with AI allocations."""
        print("\n" + "="*60)
        print("STEP 5: RUNNING BACKTEST")
        print("="*60)
        
        # Backtest AI strategy
        ai_results = self.backtester.backtest_strategy(
            self.allocations, 
            self.returns, 
            self.prices
        )
        
        # Create benchmark strategies
        benchmark_results = self.backtester.create_benchmark_strategy(
            self.returns, 
            self.prices, 
            self.config['backtest']['benchmark']
        )
        
        traditional_6040_results = self.backtester.create_traditional_6040(
            self.returns, 
            self.prices
        )
        
        # Store results
        self.backtest_results = {
            'AI Portfolio': ai_results,
            'Buy & Hold SPY': benchmark_results,
            'Traditional 60/40': traditional_6040_results
        }
        
        # Calculate and display metrics
        comparison = self.backtester.compare_strategies(self.backtest_results)
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(comparison.round(4))
        
        # Save comparison
        comp_path = os.path.join(self.config['output']['results_dir'], 'performance_comparison.csv')
        comparison.to_csv(comp_path)
        print(f"\nPerformance comparison saved to {comp_path}")
        
        return comparison
    
    def generate_visualizations(self):
        """Generate all visualization plots."""
        print("\n" + "="*60)
        print("STEP 6: GENERATING VISUALIZATIONS")
        print("="*60)
        
        figures_dir = self.config['output']['figures_dir']
        
        # 1. Portfolio value comparison
        fig1 = self.backtester.plot_portfolio_value(self.backtest_results)
        fig1.savefig(os.path.join(figures_dir, 'portfolio_value.png'), dpi=300, bbox_inches='tight')
        print("Portfolio value plot saved")
        plt.close()
        
        # 2. Drawdown comparison
        fig2 = self.backtester.plot_drawdown(self.backtest_results)
        fig2.savefig(os.path.join(figures_dir, 'drawdown.png'), dpi=300, bbox_inches='tight')
        print("Drawdown plot saved")
        plt.close()
        
        # 3. Monthly returns for AI portfolio
        fig3 = self.backtester.plot_monthly_returns(self.backtest_results['AI Portfolio'])
        fig3.savefig(os.path.join(figures_dir, 'monthly_returns.png'), dpi=300, bbox_inches='tight')
        print("Monthly returns plot saved")
        plt.close()
        
        # 4. Portfolio allocations over time
        fig4 = self.backtester.plot_allocations(self.allocations)
        fig4.savefig(os.path.join(figures_dir, 'allocations.png'), dpi=300, bbox_inches='tight')
        print("Allocations plot saved")
        plt.close()
        
        # 5. Feature importance (for first asset as example)
        fig5, ax = plt.subplots(figsize=(12, 8))
        first_asset = list(self.ml_model.feature_importance.keys())[0]
        importance = self.ml_model.feature_importance[first_asset].head(20)
        importance.plot(kind='barh', ax=ax)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top 20 Feature Importances - {first_asset}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig5.savefig(os.path.join(figures_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        print("Feature importance plot saved")
        plt.close()
        
        print(f"\nAll visualizations saved to {figures_dir}/")
    
    def run_full_strategy(self):
        """Run the complete strategy pipeline."""
        print("\n" + "="*60)
        print("AI-ENHANCED 60/40 PORTFOLIO STRATEGY")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute all steps
        self.load_data()
        self.engineer_features()
        self.train_models()
        self.generate_allocations()
        comparison = self.run_backtest()
        self.generate_visualizations()
        
        # Final summary
        print("\n" + "="*60)
        print("STRATEGY EXECUTION COMPLETE")
        print("="*60)
        
        # Highlight key metrics
        ai_sharpe = comparison.loc['AI Portfolio', 'Sharpe Ratio']
        benchmark_sharpe = comparison.loc['Buy & Hold SPY', 'Sharpe Ratio']
        improvement = ((ai_sharpe - benchmark_sharpe) / benchmark_sharpe) * 100
        
        print(f"\nKey Results:")
        print(f"  AI Portfolio Sharpe Ratio: {ai_sharpe:.4f}")
        print(f"  Benchmark Sharpe Ratio: {benchmark_sharpe:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
        
        print(f"\nAll results saved to:")
        print(f"  - Results: {self.config['output']['results_dir']}/")
        print(f"  - Figures: {self.config['output']['figures_dir']}/")
        print(f"  - Models: {self.config['output']['models_dir']}/")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        
        return comparison


def main():
    """Main function to run the strategy."""
    # Initialize and run strategy
    strategy = AIPortfolioStrategy()
    results = strategy.run_full_strategy()
    
    return strategy, results


if __name__ == "__main__":
    strategy, results = main()
