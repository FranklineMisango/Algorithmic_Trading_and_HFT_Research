"""
Main Pipeline for AI Economy Score Predictor Strategy
"""

import yaml
from data_acquisition import DataAcquisition
from llm_scorer import LLMScorer
from feature_engineering import FeatureEngineer
from prediction_model import PredictionModel
from signal_generator import SignalGenerator
from backtester import Backtester


class AIEconomyPipeline:
    """Complete pipeline orchestration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_acq = DataAcquisition(config_path)
        self.llm_scorer = LLMScorer(config_path)
        self.engineer = FeatureEngineer(config_path)
        self.pred_model = PredictionModel(config_path)
        self.signal_gen = SignalGenerator(config_path)
        self.backtester = Backtester(config_path)
    
    def run_full_pipeline(self):
        """Execute complete pipeline."""
        print("="*70)
        print("AI ECONOMY SCORE PREDICTOR - FULL PIPELINE")
        print("="*70)
        
        # Step 1: Data acquisition
        print("\nSTEP 1: DATA ACQUISITION")
        sp500 = self.data_acq.fetch_sp500_constituents()
        macro = self.data_acq.fetch_macro_data('2015-01-01', '2025-12-31')
        controls = self.data_acq.fetch_control_variables('2015-01-01', '2025-12-31')
        spf = self.data_acq.fetch_spf_forecasts('2015-01-01', '2025-12-31')
        
        print(f"Loaded {len(sp500)} companies, {len(macro)} indicators")
        
        # Step 2: LLM scoring (placeholder - would score actual transcripts)
        print("\nSTEP 2: LLM SCORING")
        print("Note: Using placeholder scores (integrate with transcript API in production)")
        
        # Step 3: Feature engineering
        print("\nSTEP 3: FEATURE ENGINEERING")
        # Would normalize scores, create deltas, etc.
        
        # Step 4: Train prediction models
        print("\nSTEP 4: TRAIN PREDICTION MODELS")
        # Would train GDP/IP/employment models
        
        # Step 5: Generate signals
        print("\nSTEP 5: GENERATE TRADING SIGNALS")
        # Would generate long/short signals
        
        # Step 6: Backtest
        print("\nSTEP 6: BACKTEST STRATEGY")
        # Would run backtest with costs
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Economy Score Predictor')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'data', 'score', 'backtest'])
    parser.add_argument('--config', type=str, default='config.yaml')
    
    args = parser.parse_args()
    
    pipeline = AIEconomyPipeline(args.config)
    
    if args.mode == 'full':
        pipeline.run_full_pipeline()
