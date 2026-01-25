"""
Feature Engineering Module for AI Economy Score Predictor

Implements n-gram analysis, score normalization, and delta features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import yaml
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print("Warning: scikit-learn not installed")


class FeatureEngineer:
    """Handles feature engineering for the strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.features_config = self.config['features']
    
    def normalize_scores(
        self,
        scores: pd.DataFrame,
        score_column: str = 'agg_score',
        method: str = 'zscore',
        window: int = 20
    ) -> pd.DataFrame:
        """
        Normalize scores using rolling window.
        
        Args:
            scores: DataFrame with scores
            score_column: Column to normalize
            method: 'zscore' or 'minmax'
            window: Rolling window size (quarters)
            
        Returns:
            DataFrame with normalized scores
        """
        df = scores.copy()
        
        if method == 'zscore':
            # Rolling mean and std
            rolling_mean = df[score_column].rolling(window=window, min_periods=4).mean()
            rolling_std = df[score_column].rolling(window=window, min_periods=4).std()
            
            # Z-score
            df[f'{score_column}_norm'] = (
                (df[score_column] - rolling_mean) / rolling_std
            )
        
        elif method == 'minmax':
            # Rolling min and max
            rolling_min = df[score_column].rolling(window=window, min_periods=4).min()
            rolling_max = df[score_column].rolling(window=window, min_periods=4).max()
            
            # Min-max normalization
            df[f'{score_column}_norm'] = (
                (df[score_column] - rolling_min) / (rolling_max - rolling_min)
            )
        
        return df
    
    def create_delta_features(
        self,
        scores: pd.DataFrame,
        score_column: str = 'agg_score'
    ) -> pd.DataFrame:
        """
        Create delta (momentum) features.
        
        Args:
            scores: DataFrame with scores
            score_column: Column to compute deltas from
            
        Returns:
            DataFrame with delta features
        """
        df = scores.copy()
        
        # YoY change (t vs t-4)
        df['yoy_change'] = df[score_column] - df[score_column].shift(4)
        
        # QoQ change (t vs t-1)
        df['qoq_change'] = df[score_column] - df[score_column].shift(1)
        
        # Momentum (acceleration of YoY change)
        df['momentum'] = (
            (df[score_column] - df[score_column].shift(4)) -
            (df[score_column].shift(4) - df[score_column].shift(8))
        )
        
        return df
    
    def extract_ngrams(
        self,
        texts: List[str],
        min_n: int = 2,
        max_n: int = 4,
        top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract top n-grams using TF-IDF.
        
        Args:
            texts: List of text strings
            min_n: Minimum n-gram size
            max_n: Maximum n-gram size
            top_k: Number of top n-grams to return
            
        Returns:
            Dict of n-grams by size
        """
        ngrams_by_size = {}
        
        for n in range(min_n, max_n + 1):
            try:
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    ngram_range=(n, n),
                    max_features=top_k,
                    stop_words='english',
                    lowercase=True
                )
                
                # Fit and transform
                tfidf_matrix = vectorizer.fit_transform(texts)
                
                # Get feature names and scores
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.sum(axis=0).A1
                
                # Sort by score
                top_indices = scores.argsort()[-top_k:][::-1]
                top_ngrams = [(feature_names[i], scores[i]) for i in top_indices]
                
                ngrams_by_size[f'{n}-gram'] = top_ngrams
                
            except Exception as e:
                print(f"Error extracting {n}-grams: {e}")
        
        return ngrams_by_size
    
    def create_sentiment_fingerprints(
        self,
        transcripts: pd.DataFrame,
        text_column: str = 'text',
        score_column: str = 'firm_score'
    ) -> Dict[str, List[str]]:
        """
        Identify n-gram "fingerprints" for high/low sentiment.
        
        Args:
            transcripts: DataFrame with texts and scores
            text_column: Column with transcript text
            score_column: Column with scores
            
        Returns:
            Dict with 'positive' and 'negative' fingerprint phrases
        """
        # Split into high and low score groups
        median_score = transcripts[score_column].median()
        
        high_texts = transcripts[transcripts[score_column] > median_score][text_column].tolist()
        low_texts = transcripts[transcripts[score_column] <= median_score][text_column].tolist()
        
        # Extract n-grams for each group
        high_ngrams = self.extract_ngrams(high_texts, top_k=50)
        low_ngrams = self.extract_ngrams(low_texts, top_k=50)
        
        # Find distinctive phrases (high TF-IDF in one group vs another)
        fingerprints = {
            'positive': [],
            'negative': []
        }
        
        for n_size in high_ngrams:
            high_set = set([phrase for phrase, _ in high_ngrams[n_size][:20]])
            low_set = set([phrase for phrase, _ in low_ngrams[n_size][:20]])
            
            # Phrases unique to high sentiment
            fingerprints['positive'].extend(list(high_set - low_set))
            
            # Phrases unique to low sentiment
            fingerprints['negative'].extend(list(low_set - high_set))
        
        return fingerprints
    
    def create_interaction_features(
        self,
        agg_scores: pd.DataFrame,
        controls: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create interaction features between AGG score and controls.
        
        Args:
            agg_scores: DataFrame with AGG scores
            controls: DataFrame with control variables
            
        Returns:
            DataFrame with interaction features
        """
        df = agg_scores.merge(controls, on='date', how='left')
        
        # AGG * Yield curve slope
        if 'yield_curve_slope' in df.columns:
            df['agg_x_yieldcurve'] = df['agg_score'] * df['yield_curve_slope']
        
        # AGG * Consumer sentiment
        if 'consumer_sentiment' in df.columns:
            df['agg_x_sentiment'] = df['agg_score'] * df['consumer_sentiment']
        
        # AGG * VIX (risk environment)
        if 'vix' in df.columns:
            df['agg_x_vix'] = df['agg_score'] * df['vix']
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 4]
    ) -> pd.DataFrame:
        """
        Create lagged features.
        
        Args:
            df: DataFrame
            columns: Columns to lag
            lags: Lag periods (quarters)
            
        Returns:
            DataFrame with lagged features
        """
        df_out = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df_out[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        return df_out
    
    def create_full_feature_set(
        self,
        agg_scores: pd.DataFrame,
        controls: pd.DataFrame,
        score_column: str = 'agg_score'
    ) -> pd.DataFrame:
        """
        Create complete feature set for modeling.
        
        Args:
            agg_scores: AGG scores
            controls: Control variables
            score_column: Score column name
            
        Returns:
            DataFrame with all features
        """
        # Normalize scores
        df = self.normalize_scores(agg_scores, score_column)
        
        # Delta features
        df = self.create_delta_features(df, score_column)
        
        # Merge controls
        df = df.merge(controls, on='date', how='left')
        
        # Interaction features
        df = self.create_interaction_features(df, controls)
        
        # Lag features
        lag_columns = [score_column, f'{score_column}_norm', 'yoy_change', 'qoq_change']
        df = self.create_lag_features(df, lag_columns)
        
        return df


# Test code
if __name__ == "__main__":
    engineer = FeatureEngineer('config.yaml')
    
    # Test data
    dates = pd.date_range('2010-01-01', periods=40, freq='Q')
    scores = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'quarter': dates.quarter,
        'agg_score': np.random.normal(3.0, 0.5, 40)
    })
    
    # Test normalization
    normalized = engineer.normalize_scores(scores)
    print("Normalized scores:")
    print(normalized[['date', 'agg_score', 'agg_score_norm']].head())
    
    # Test delta features
    with_deltas = engineer.create_delta_features(normalized)
    print("\nDelta features:")
    print(with_deltas[['date', 'agg_score', 'yoy_change', 'qoq_change', 'momentum']].head(10))
    
    # Test n-grams
    sample_texts = [
        "strong financial performance and robust demand",
        "challenging economic environment with headwinds",
        "positive outlook for growth and expansion"
    ]
    
    ngrams = engineer.extract_ngrams(sample_texts, min_n=2, max_n=3, top_k=10)
    print("\nTop n-grams:")
    for size, phrases in ngrams.items():
        print(f"\n{size}:")
        for phrase, score in phrases[:5]:
            print(f"  {phrase}: {score:.3f}")
