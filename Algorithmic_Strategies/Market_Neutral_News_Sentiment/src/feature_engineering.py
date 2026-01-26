import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class FeatureEngineer:
    def __init__(self):
        self.topic_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.fitted = False
    
    def calculate_sector_relative_returns(self, returns_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns relative to GICS sector average"""
        merged = returns_df.merge(sector_df, on='ticker', how='left')
        merged['sector_avg_return'] = merged.groupby(['date', 'gics_sector'])['return'].transform('mean')
        merged['sector_relative_return'] = merged['return'] - merged['sector_avg_return']
        return merged
    
    def encode_topics(self, news_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """One-hot encode news topics"""
        if fit:
            topic_encoded = self.topic_encoder.fit_transform(news_df[['topic']])
            self.fitted = True
        else:
            topic_encoded = self.topic_encoder.transform(news_df[['topic']])
        
        topic_cols = [f'topic_{i}' for i in range(topic_encoded.shape[1])]
        topic_df = pd.DataFrame(topic_encoded, columns=topic_cols, index=news_df.index)
        return pd.concat([news_df, topic_df], axis=1)
