"""
Main Pipeline for Sentiment-Based LLM Equity Strategy

Orchestrates full pipeline: data → model → portfolio → backtest
"""

import argparse
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from data_acquisition import SentimentDataAcquisition
from sentiment_model import SmartyBERT
from portfolio_construction import SentimentPortfolioConstructor
from backtester import SentimentBacktester


class SentimentLLMPipeline:
    """Full pipeline for sentiment-based equity strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        
        # Create output directories
        Path("data/market").mkdir(parents=True, exist_ok=True)
        Path("data/text").mkdir(parents=True, exist_ok=True)
        Path("data/models").mkdir(parents=True, exist_ok=True)
        Path("data/portfolios").mkdir(parents=True, exist_ok=True)
        Path("data/results").mkdir(parents=True, exist_ok=True)
    
    def run_data_pipeline(self, start_date: str, end_date: str):
        """
        Fetch and save market and text data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        print("\n" + "="*60)
        print("STEP 1: DATA ACQUISITION")
        print("="*60)
        
        data_acq = SentimentDataAcquisition(self.config_path)
        dataset = data_acq.fetch_full_dataset(start_date, end_date)
        
        print(f"\nMarket data: {len(dataset['market_data'])} rows")
        print(f"Text data: {len(dataset['text_data'])} rows")
        print("\nData saved to data/market/ and data/text/")
        
        return dataset
    
    def run_model_training(self, dataset: dict):
        """
        Train sentiment model on market-labeled data.
        
        Args:
            dataset: Dict with 'market_data' and 'text_data'
        """
        print("\n" + "="*60)
        print("STEP 2: MODEL TRAINING")
        print("="*60)
        
        # Initialize model
        smarty_bert = SmartyBERT(self.config_path)
        
        # Merge text and market data
        merged = dataset['text_data'].merge(
            dataset['market_data'][['ticker', 'date', 'return']],
            on=['ticker', 'date'],
            how='inner'
        )
        
        # Construct market-labeled targets
        # Simplified: use returns as labels (placeholder for abnormal returns)
        labels = smarty_bert.construct_labels(
            merged['return'],
            merged['return'] * 0.8,  # Placeholder market returns
            beta=1.0
        )
        
        # Train on subset for demo
        train_size = min(1000, len(merged))
        texts = merged['text'].iloc[:train_size].tolist()
        labels = labels[:train_size]
        
        print(f"\nTraining on {train_size} samples...")
        history = smarty_bert.train_model(texts, labels, validation_split=0.2)
        
        # Save model
        model_path = "data/models/smarty_bert"
        smarty_bert.save_model(model_path)
        print(f"\nModel saved to {model_path}")
        
        return smarty_bert, history
    
    def run_inference(self, dataset: dict, model: SmartyBERT = None):
        """
        Generate sentiment predictions.
        
        Args:
            dataset: Dataset dict
            model: Trained model (or load from checkpoint)
        
        Returns:
            Sentiment scores DataFrame
        """
        print("\n" + "="*60)
        print("STEP 3: INFERENCE")
        print("="*60)
        
        if model is None:
            # Load model
            model = SmartyBERT(self.config_path)
            model.load_model("data/models/smarty_bert")
        
        # Predict sentiment
        texts = dataset['text_data']['text'].tolist()
        
        print(f"\nPredicting on {len(texts)} samples...")
        predictions = model.predict(texts, batch_size=32)
        
        # Create sentiment DataFrame
        sentiment_scores = dataset['text_data'][['ticker', 'date']].copy()
        sentiment_scores['sentiment'] = predictions
        
        # Save
        sentiment_scores.to_csv("data/portfolios/sentiment_scores.csv", index=False)
        print("\nSentiment scores saved to data/portfolios/sentiment_scores.csv")
        
        return sentiment_scores
    
    def run_portfolio_construction(
        self,
        sentiment_scores: pd.DataFrame,
        market_data: pd.DataFrame
    ):
        """
        Construct long-short portfolios.
        
        Args:
            sentiment_scores: Sentiment scores
            market_data: Market data
        
        Returns:
            Portfolio weights
        """
        print("\n" + "="*60)
        print("STEP 4: PORTFOLIO CONSTRUCTION")
        print("="*60)
        
        constructor = SentimentPortfolioConstructor(self.config_path)
        portfolio = constructor.construct_portfolio(sentiment_scores, market_data)
        
        # Save
        portfolio.to_csv("data/portfolios/portfolio_weights.csv", index=False)
        print(f"\nPortfolio constructed: {len(portfolio)} positions")
        print("\nPortfolio saved to data/portfolios/portfolio_weights.csv")
        
        return portfolio
    
    def run_backtest(self, portfolio: pd.DataFrame, market_data: pd.DataFrame):
        """
        Backtest strategy.
        
        Args:
            portfolio: Portfolio weights
            market_data: Market data
        
        Returns:
            Backtest results
        """
        print("\n" + "="*60)
        print("STEP 5: BACKTEST")
        print("="*60)
        
        backtester = SentimentBacktester(self.config_path)
        results = backtester.run_backtest(portfolio, market_data)
        
        # Save results
        results['daily_returns'].to_csv("data/results/daily_returns.csv", index=False)
        
        # Print metrics
        print("\n" + "-"*60)
        print("PERFORMANCE METRICS")
        print("-"*60)
        for metric, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"{metric:30s}: {value:10.4f}")
            else:
                print(f"{metric:30s}: {value:10}")
        
        # Compare to benchmarks
        benchmarks = self.config['evaluation']['benchmarks']
        print("\n" + "-"*60)
        print("BENCHMARK COMPARISON")
        print("-"*60)
        print(f"Strategy Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"Target Sharpe (from research): {benchmarks['sharpe_ratio']:.2f}")
        print(f"Target Annual Return: {benchmarks['annualized_return']:.2%}")
        
        return results
    
    def run_full_pipeline(self, start_date: str, end_date: str):
        """
        Run complete pipeline.
        
        Args:
            start_date: Start date
            end_date: End date
        """
        print("\n" + "="*80)
        print(" SENTIMENT-BASED LLM EQUITY STRATEGY - FULL PIPELINE")
        print("="*80)
        
        # 1. Data
        dataset = self.run_data_pipeline(start_date, end_date)
        
        # 2. Model Training
        model, history = self.run_model_training(dataset)
        
        # 3. Inference
        sentiment_scores = self.run_inference(dataset, model)
        
        # 4. Portfolio Construction
        portfolio = self.run_portfolio_construction(sentiment_scores, dataset['market_data'])
        
        # 5. Backtest
        results = self.run_backtest(portfolio, dataset['market_data'])
        
        print("\n" + "="*80)
        print(" PIPELINE COMPLETE")
        print("="*80)
        print("\nResults saved to data/results/")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sentiment-Based LLM Equity Strategy Pipeline"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['data', 'train', 'inference', 'portfolio', 'backtest', 'full'],
        default='full',
        help="Pipeline mode"
    )
    parser.add_argument('--start-date', type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end-date', type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument('--config', type=str, default="config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SentimentLLMPipeline(args.config)
    
    if args.mode == 'full':
        # Run complete pipeline
        pipeline.run_full_pipeline(args.start_date, args.end_date)
    
    elif args.mode == 'data':
        # Data acquisition only
        pipeline.run_data_pipeline(args.start_date, args.end_date)
    
    elif args.mode == 'train':
        # Train model
        data_acq = SentimentDataAcquisition(args.config)
        dataset = data_acq.fetch_full_dataset(args.start_date, args.end_date)
        pipeline.run_model_training(dataset)
    
    elif args.mode == 'inference':
        # Inference only
        data_acq = SentimentDataAcquisition(args.config)
        dataset = data_acq.fetch_full_dataset(args.start_date, args.end_date)
        pipeline.run_inference(dataset)
    
    elif args.mode == 'portfolio':
        # Portfolio construction only
        sentiment_scores = pd.read_csv("data/portfolios/sentiment_scores.csv")
        market_data = pd.read_csv("data/market/prices.csv")
        pipeline.run_portfolio_construction(sentiment_scores, market_data)
    
    elif args.mode == 'backtest':
        # Backtest only
        portfolio = pd.read_csv("data/portfolios/portfolio_weights.csv")
        market_data = pd.read_csv("data/market/prices.csv")
        pipeline.run_backtest(portfolio, market_data)


if __name__ == "__main__":
    main()
