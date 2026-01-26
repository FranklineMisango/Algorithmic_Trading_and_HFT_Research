#!/usr/bin/env python3
import yaml
import pandas as pd
from src import SentimentSignal, FeatureEngineer, SentimentModel, PortfolioConstructor, Backtester

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    print("=" * 60)
    print("Market-Neutral News Sentiment Strategy")
    print("=" * 60)
    
    # Initialize components
    signal_calc = SentimentSignal(
        epsilon=config['signal']['epsilon'],
        relevance_threshold=config['signal']['relevance_threshold']
    )
    
    feature_eng = FeatureEngineer()
    
    model = SentimentModel(
        alpha=config['model']['alpha'],
        training_years=config['model']['training_years']
    )
    
    portfolio_constructor = PortfolioConstructor(
        long_pct=config['portfolio']['long_percentile'],
        short_pct=config['portfolio']['short_percentile'],
        net_exposure=config['portfolio']['net_exposure'],
        sp1500_only_shorts=config['portfolio']['sp1500_shorts_only']
    )
    
    backtester = Backtester(portfolio_constructor)
    
    print("\n[1/5] Loading data...")
    # Load your data here
    # df = pd.read_csv('data/processed_data.csv')
    
    print("[2/5] Calculating sentiment signals...")
    # sentiment_scores = signal_calc.aggregate_stock_sentiment(news_df)
    
    print("[3/5] Training model with walk-forward validation...")
    # predictions = model.walk_forward_validation(df, start_year, end_year)
    
    print("[4/5] Running backtest...")
    # results = backtester.run(predictions, returns_df, sp1500_constituents)
    
    print("[5/5] Calculating performance metrics...")
    # metrics = backtester.calculate_metrics()
    
    print("\n" + "=" * 60)
    print("Backtest Complete")
    print("=" * 60)
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
