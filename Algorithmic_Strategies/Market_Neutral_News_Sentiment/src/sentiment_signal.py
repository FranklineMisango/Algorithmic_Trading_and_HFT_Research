import numpy as np
import pandas as pd
from typing import Dict, List

class SentimentSignal:
    def __init__(self, epsilon: float = 1e-9, relevance_threshold: float = 0.7):
        self.epsilon = epsilon
        self.relevance_threshold = relevance_threshold
    
    def calculate_article_score(self, p_positive: float, p_negative: float) -> float:
        """Log-odds ratio sentiment score per article"""
        return np.log((p_positive + self.epsilon) / (p_negative + self.epsilon))
    
    def aggregate_stock_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news to stock-day level"""
        news_df = news_df[news_df['relevance_score'] >= self.relevance_threshold].copy()
        news_df['article_score'] = news_df.apply(
            lambda x: self.calculate_article_score(x['p_positive'], x['p_negative']), axis=1
        )
        return news_df.groupby(['ticker', 'date'])['article_score'].mean().reset_index()
    
    def cross_sectional_rank(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Rank stocks by sentiment within each date"""
        sentiment_df['sentiment_rank'] = sentiment_df.groupby('date')['article_score'].rank(pct=True)
        sentiment_df['sentiment_zscore'] = sentiment_df.groupby('date')['article_score'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        return sentiment_df
