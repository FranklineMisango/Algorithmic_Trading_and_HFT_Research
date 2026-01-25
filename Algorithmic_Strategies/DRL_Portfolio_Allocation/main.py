"""
Main Pipeline for DRL Portfolio Allocation

Orchestrates training, hyperparameter optimization, backtesting, and evaluation.
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from data_acquisition import DataAcquisition
from portfolio_env import PortfolioEnv
from rl_agent import DRLAgent
from benchmark import MarkowitzOptimizer, Classic6040, EqualWeight, BenchmarkBacktester
from backtester import Backtester


class DRLPortfolioPipeline:
    """Main pipeline for DRL portfolio allocation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline."""
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directories
        os.makedirs(self.config['output']['results_dir'], exist_ok=True)
        os.makedirs(self.config['output']['models_dir'], exist_ok=True)
        os.makedirs(self.config['output']['logs_dir'], exist_ok=True)
    
    def run_data_acquisition(self) -> Dict:
        """Fetch and prepare data."""
        print("=" * 60)
        print("STEP 1: Data Acquisition")
        print("=" * 60)
        
        data_acq = DataAcquisition(self.config_path)
        dataset = data_acq.fetch_full_dataset()
        
        return dataset
    
    def run_training(self, dataset: Dict) -> DRLAgent:
        """Train DRL agent."""
        print("\n" + "=" * 60)
        print("STEP 2: Agent Training")
        print("=" * 60)
        
        # Create environments
        train_env = PortfolioEnv(
            prices=dataset['train']['prices'],
            returns=dataset['train']['returns'],
            config_path=self.config_path
        )
        
        val_env = PortfolioEnv(
            prices=dataset['val']['prices'],
            returns=dataset['val']['returns'],
            config_path=self.config_path
        )
        
        # Create agent
        agent = DRLAgent(train_env, config_path=self.config_path)
        
        # Train
        agent.train(
            eval_env=val_env,
            save_path=f"{self.config['output']['models_dir']}/best_model"
        )
        
        # Save final model
        agent.save(f"{self.config['output']['models_dir']}/final_model")
        
        return agent
    
    def run_backtest(self, agent: DRLAgent, dataset: Dict) -> Dict:
        """Backtest agent and benchmarks."""
        print("\n" + "=" * 60)
        print("STEP 3: Backtesting")
        print("=" * 60)
        
        test_prices = dataset['test']['prices']
        test_returns = dataset['test']['returns']
        
        # Backtest DRL agent
        print("\n1. DRL Agent...")
        backtester = Backtester(self.config_path)
        drl_results = backtester.run_backtest(agent, test_prices, test_returns)
        
        # Backtest benchmarks
        print("\n2. Benchmarks...")
        benchmark_backtester = BenchmarkBacktester(self.config_path)
        
        # Markowitz
        print("   - Markowitz MVO...")
        mvo_results = benchmark_backtester.backtest_markowitz(test_prices, test_returns)
        mvo_df = pd.DataFrame({
            'portfolio_value': mvo_results['portfolio_history'][1:],
            'returns': test_returns.values @ np.array(mvo_results['weights_history'][:-1]).T
        }, index=test_returns.index)
        mvo_metrics = backtester.calculate_metrics(mvo_df)
        
        # 60/40
        print("   - 60/40...")
        classic = Classic6040(self.config_path)
        classic_results = benchmark_backtester.backtest_static(test_prices, test_returns, classic)
        classic_df = pd.DataFrame({
            'portfolio_value': classic_results['portfolio_history'][1:],
            'returns': test_returns.values @ classic.get_weights()
        }, index=test_returns.index)
        classic_metrics = backtester.calculate_metrics(classic_df)
        
        # Equal weight
        print("   - Equal Weight...")
        equal = EqualWeight(n_assets=len(test_prices.columns))
        equal_results = benchmark_backtester.backtest_static(test_prices, test_returns, equal)
        equal_df = pd.DataFrame({
            'portfolio_value': equal_results['portfolio_history'][1:],
            'returns': test_returns.values @ equal.get_weights()
        }, index=test_returns.index)
        equal_metrics = backtester.calculate_metrics(equal_df)
        
        return {
            'DRL': {'results_df': drl_results['results_df'], 'metrics': drl_results['metrics']},
            'Markowitz': {'results_df': mvo_df, 'metrics': mvo_metrics},
            '60/40': {'results_df': classic_df, 'metrics': classic_metrics},
            'Equal Weight': {'results_df': equal_df, 'metrics': equal_metrics}
        }
    
    def compare_results(self, all_results: Dict):
        """Compare and visualize results."""
        print("\n" + "=" * 60)
        print("STEP 4: Performance Comparison")
        print("=" * 60)
        
        backtester = Backtester(self.config_path)
        
        # Create comparison table
        comparison_df = backtester.compare_strategies(all_results)
        print("\n" + str(comparison_df))
        
        # Save results
        comparison_df.to_csv(f"{self.config['output']['results_dir']}/comparison.csv")
        
        # Statistical tests
        print("\n" + "=" * 60)
        print("Statistical Tests (DRL vs Benchmarks)")
        print("=" * 60)
        
        drl_returns = all_results['DRL']['results_df']['returns'].values
        
        for benchmark in ['Markowitz', '60/40', 'Equal Weight']:
            benchmark_returns = all_results[benchmark]['results_df']['returns'].values
            
            test_results = backtester.statistical_tests(drl_returns, benchmark_returns)
            
            print(f"\nDRL vs {benchmark}:")
            print(f"  Mean difference: {test_results['mean_diff']:.6f}")
            print(f"  T-statistic: {test_results['t_statistic']:.3f}")
            print(f"  P-value: {test_results['p_value']:.4f}")
            print(f"  Significant: {'Yes' if test_results['is_significant'] else 'No'}")
        
        return comparison_df
    
    def run_full_pipeline(self):
        """Execute full pipeline."""
        print("\n" + "=" * 60)
        print("DRL PORTFOLIO ALLOCATION - FULL PIPELINE")
        print("=" * 60)
        
        # Step 1: Data
        dataset = self.run_data_acquisition()
        
        # Step 2: Training
        agent = self.run_training(dataset)
        
        # Step 3: Backtest
        all_results = self.run_backtest(agent, dataset)
        
        # Step 4: Compare
        comparison_df = self.compare_results(all_results)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nResults saved to: {self.config['output']['results_dir']}/")
        print(f"Models saved to: {self.config['output']['models_dir']}/")
        
        return {
            'dataset': dataset,
            'agent': agent,
            'results': all_results,
            'comparison': comparison_df
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DRL Portfolio Allocation")
    
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'data', 'train', 'backtest', 'compare'],
        help='Pipeline mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DRLPortfolioPipeline(args.config)
    
    if args.mode == 'full':
        pipeline.run_full_pipeline()
    
    elif args.mode == 'data':
        dataset = pipeline.run_data_acquisition()
        print("Data acquisition completed")
    
    elif args.mode == 'train':
        dataset = pipeline.run_data_acquisition()
        agent = pipeline.run_training(dataset)
        print("Training completed")
    
    elif args.mode == 'backtest':
        # Load existing model
        dataset = pipeline.run_data_acquisition()
        
        from portfolio_env import PortfolioEnv
        env = PortfolioEnv(
            dataset['test']['prices'],
            dataset['test']['returns']
        )
        agent = DRLAgent(env, config_path=args.config)
        agent.load(f"{pipeline.config['output']['models_dir']}/best_model")
        
        results = pipeline.run_backtest(agent, dataset)
        pipeline.compare_results(results)
        print("Backtesting completed")
    
    else:
        print(f"Mode {args.mode} not fully implemented")


if __name__ == "__main__":
    main()
