"""
Main Pipeline for Jump Risk Crypto Strategy
Orchestrates the complete workflow from data loading to evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import argparse
import logging
from datetime import datetime

from data_loader import load_and_prepare_data, CryptoDataLoader
from jump_detector import detect_and_analyze_jumps, JumpDetector
from copula_analyzer import analyze_contagion, CopulaAnalyzer
from portfolio_optimizer import optimize_and_compare, PortfolioOptimizer
from backtester import run_backtest, JumpRiskBacktester
from performance_evaluator import evaluate_performance, PerformanceEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JumpRiskPipeline:
    """
    Main pipeline for jump risk analysis and portfolio optimization
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results = {}
        
    def run(self, data_filepath: str = None, save_results: bool = True):
        """
        Run complete pipeline
        
        Args:
            data_filepath: Optional path to data file
            save_results: Whether to save results to disk
        """
        logger.info("="*60)
        logger.info("JUMP RISK CRYPTO PORTFOLIO OPTIMIZATION PIPELINE")
        logger.info("="*60)
        
        # Phase 0: Load Data
        logger.info("\n[PHASE 0] Loading data...")
        data_splits = load_and_prepare_data(self.config, data_filepath)
        self.results['data_splits'] = data_splits
        
        # Phase 1: Jump Detection
        logger.info("\n[PHASE 1] Detecting jumps...")
        train_df = data_splits['train']
        test_df = data_splits['test']
        
        train_jumps, train_metrics, train_cojumps = detect_and_analyze_jumps(
            train_df, self.config
        )
        test_jumps, test_metrics, test_cojumps = detect_and_analyze_jumps(
            test_df, self.config
        )
        
        self.results['train_jumps'] = train_jumps
        self.results['test_jumps'] = test_jumps
        self.results['jump_metrics'] = {'train': train_metrics, 'test': test_metrics}
        self.results['cojumps'] = {'train': train_cojumps, 'test': test_cojumps}
        
        # Phase 2: Contagion Analysis
        logger.info("\n[PHASE 2] Analyzing jump contagion...")
        contagion_results = analyze_contagion(train_jumps, self.config)
        
        self.results['contagion'] = contagion_results
        
        # Log key contagion findings
        high_risk_pairs = contagion_results['jump_ratios'][
            contagion_results['jump_ratios']['risk_level'].isin(['high', 'critical'])
        ]
        logger.info(f"\nKey Findings:")
        logger.info(f"  High-risk pairs: {len(high_risk_pairs)}")
        logger.info(f"  Contagion clusters: {len(contagion_results['clusters'])}")
        
        # Phase 3: Portfolio Optimization
        logger.info("\n[PHASE 3] Optimizing portfolio...")
        
        # Prepare training data
        train_returns = train_jumps.pivot(
            index='date', columns='asset', values='returns'
        ).dropna()
        train_jump_returns = train_jumps.pivot(
            index='date', columns='asset', values='jump_size'
        ).fillna(0)
        
        # Align dates
        common_dates = train_returns.index.intersection(train_jump_returns.index)
        train_returns = train_returns.loc[common_dates]
        train_jump_returns = train_jump_returns.loc[common_dates]
        
        # Optimize
        opt_result = optimize_and_compare(train_returns, train_jump_returns, self.config)
        self.results['optimization'] = opt_result
        
        # Log optimal weights
        logger.info("\nOptimal Portfolio Weights (Jump-Adjusted):")
        sorted_weights = sorted(
            opt_result['optimal_weights'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for asset, weight in sorted_weights:
            if weight > 0.01:
                logger.info(f"  {asset}: {weight*100:.1f}%")
        
        # Phase 4: Backtesting
        logger.info("\n[PHASE 4] Running backtest...")
        backtest_results = run_backtest(test_jumps, self.config)
        self.results['backtest'] = backtest_results
        
        # Phase 5: Performance Evaluation
        logger.info("\n[PHASE 5] Evaluating performance...")
        performance_report = evaluate_performance(
            backtest_results['combined_results'], 
            self.config
        )
        self.results['performance'] = performance_report
        
        # Summary
        self._print_summary()
        
        # Save results
        if save_results:
            self._save_results()
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        
        return self.results
    
    def _print_summary(self):
        """
        Print executive summary of results
        """
        logger.info("\n" + "="*60)
        logger.info("EXECUTIVE SUMMARY")
        logger.info("="*60)
        
        # Jump detection summary
        jump_metrics = self.results['jump_metrics']['test']
        avg_intensity = np.mean([m['jump_intensity'] for m in jump_metrics.values()])
        logger.info(f"\nJump Detection:")
        logger.info(f"  Average jump intensity: {avg_intensity*100:.2f}%")
        logger.info(f"  Systemic co-jump events: {self.results['cojumps']['test']['is_systemic'].sum()}")
        
        # Contagion summary
        contagion = self.results['contagion']
        logger.info(f"\nContagion Analysis:")
        logger.info(f"  High-risk pairs: {(contagion['jump_ratios']['risk_level'] != 'low').sum()}")
        logger.info(f"  Contagion clusters: {len(contagion['clusters'])}")
        
        # Tail dependence
        tail_summary = contagion['tail_summary']
        avg_upper = tail_summary['lambda_upper'].mean()
        avg_lower = tail_summary['lambda_lower'].mean()
        logger.info(f"  Avg upper tail λ: {avg_upper:.3f}")
        logger.info(f"  Avg lower tail λ: {avg_lower:.3f}")
        if avg_upper > avg_lower:
            logger.info(f"  → Surge contagion stronger than crash contagion")
        
        # Portfolio performance
        performance = self.results['performance']
        comparison = performance['comparison']
        
        logger.info(f"\nPortfolio Performance:")
        for strategy in comparison.index:
            sharpe = comparison.loc[strategy, 'sharpe_ratio']
            annual_ret = comparison.loc[strategy, 'annualized_return'] * 100
            max_dd = comparison.loc[strategy, 'max_drawdown'] * 100
            logger.info(f"  {strategy:20s}: Sharpe={sharpe:.3f}, Return={annual_ret:+.1f}%, MaxDD={max_dd:.1f}%")
        
        # Statistical significance
        significant = performance['significant_improvements']
        if significant:
            logger.info(f"\nStatistical Significance:")
            logger.info(f"  Jump-adjusted strategy shows {len(significant)} significant improvements")
            logger.info(f"  Confidence level: {self.config['metrics']['statistical_tests']['confidence_level']*100}%")
        else:
            logger.info(f"\nNo statistically significant improvements detected")
    
    def _save_results(self):
        """
        Save results to disk
        """
        # Create results directory
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save jump detection results
        self.results['train_jumps'].to_csv(
            results_dir / f'train_jumps_{timestamp}.csv', 
            index=False
        )
        self.results['test_jumps'].to_csv(
            results_dir / f'test_jumps_{timestamp}.csv', 
            index=False
        )
        
        # Save contagion analysis
        self.results['contagion']['jump_ratios'].to_csv(
            results_dir / f'jump_ratios_{timestamp}.csv', 
            index=False
        )
        self.results['contagion']['tail_summary'].to_csv(
            results_dir / f'tail_dependence_{timestamp}.csv', 
            index=False
        )
        
        # Save portfolio weights
        weights_df = pd.DataFrame([
            {'asset': asset, 'weight': weight}
            for asset, weight in self.results['optimization']['optimal_weights'].items()
        ])
        weights_df.to_csv(results_dir / f'optimal_weights_{timestamp}.csv', index=False)
        
        # Save backtest results
        self.results['backtest']['combined_results'].to_csv(
            results_dir / f'backtest_results_{timestamp}.csv', 
            index=False
        )
        
        # Save performance comparison
        self.results['performance']['comparison'].to_csv(
            results_dir / f'performance_comparison_{timestamp}.csv'
        )
        
        logger.info(f"\nResults saved to: {results_dir}")


def main():
    """
    Main entry point for CLI
    """
    parser = argparse.ArgumentParser(
        description='Jump Risk Crypto Portfolio Optimization'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default=None,
        help='Path to data file (optional, will generate synthetic if not provided)'
    )
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='Do not save results to disk'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = JumpRiskPipeline(args.config)
    results = pipeline.run(
        data_filepath=args.data, 
        save_results=not args.no_save
    )
    
    return results


if __name__ == '__main__':
    main()
