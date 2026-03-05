"""
Geopolitical risk scoring based on news sentiment and conflict events.
"""

import pandas as pd
import numpy as np
from typing import Dict
import yaml


class GeopoliticalScorer:
    """Score geopolitical risk from multiple sources."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.thresholds = self.config['signals']['risk_score_thresholds']
    
    def calculate_composite_score(self, geopolitical_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite geopolitical risk score."""
        df = geopolitical_data.copy()
        
        # Normalize components to 0-100 scale
        df['risk_score_norm'] = df['risk_score']  # Already 0-100
        
        # Sentiment: -1 to 1 -> 0 to 100 (negative sentiment = higher risk)
        df['sentiment_score'] = (1 - df['news_sentiment']) * 50
        
        # Conflict events: normalize by 95th percentile
        event_95 = df['conflict_events'].quantile(0.95)
        df['conflict_score'] = (df['conflict_events'] / event_95 * 100).clip(upper=100)
        
        # Military activity: already 0-100
        df['military_score'] = df['military_activity']
        
        # Weighted composite score
        weights = {
            'risk_score_norm': 0.40,
            'sentiment_score': 0.25,
            'conflict_score': 0.20,
            'military_score': 0.15
        }
        
        df['composite_risk_score'] = sum(
            df[col] * weight for col, weight in weights.items()
        )
        
        # Smooth with exponential moving average
        df['composite_risk_score_smooth'] = df['composite_risk_score'].ewm(span=7).mean()
        
        return df
    
    def classify_risk_level(self, geopolitical_data: pd.DataFrame) -> pd.DataFrame:
        """Classify risk into levels."""
        df = self.calculate_composite_score(geopolitical_data)
        
        # Classify based on thresholds
        df['risk_level'] = 'low'
        df.loc[df['composite_risk_score_smooth'] >= self.thresholds['medium'], 'risk_level'] = 'medium'
        df.loc[df['composite_risk_score_smooth'] >= self.thresholds['high'], 'risk_level'] = 'high'
        df.loc[df['composite_risk_score_smooth'] >= self.thresholds['critical'], 'risk_level'] = 'critical'
        
        return df
    
    def detect_risk_spikes(self, geopolitical_data: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """Detect sudden spikes in geopolitical risk."""
        df = self.calculate_composite_score(geopolitical_data)
        
        # Calculate daily change
        df['risk_change'] = df['composite_risk_score_smooth'].diff()
        
        # Z-score of changes
        df['risk_change_zscore'] = (
            df['risk_change'] - df['risk_change'].rolling(30).mean()
        ) / (df['risk_change'].rolling(30).std() + 1e-6)
        
        # Flag spikes
        df['risk_spike'] = df['risk_change_zscore'] > threshold
        
        return df
    
    def get_current_assessment(self, geopolitical_data: pd.DataFrame) -> Dict:
        """Get current geopolitical risk assessment."""
        df = self.classify_risk_level(geopolitical_data)
        latest = df.iloc[-1]
        
        assessment = {
            'date': df.index[-1],
            'composite_risk_score': latest['composite_risk_score_smooth'],
            'risk_level': latest['risk_level'],
            'news_sentiment': latest['news_sentiment'],
            'conflict_events': latest['conflict_events'],
            'military_activity': latest['military_activity'],
            'risk_change_7d': df['composite_risk_score_smooth'].iloc[-7:].diff().mean()
        }
        
        return assessment


if __name__ == "__main__":
    # Test geopolitical scorer
    from data_acquisition import DataAcquisition
    
    print("Testing Geopolitical Scorer...")
    
    acq = DataAcquisition()
    data = acq.fetch_all_data()
    
    scorer = GeopoliticalScorer()
    
    # Calculate risk scores
    scored = scorer.classify_risk_level(data['geopolitical'])
    
    print("\nGeopolitical Risk Analysis:")
    print(f"Total days: {len(scored)}")
    print(f"Low risk: {(scored['risk_level'] == 'low').sum()}")
    print(f"Medium risk: {(scored['risk_level'] == 'medium').sum()}")
    print(f"High risk: {(scored['risk_level'] == 'high').sum()}")
    print(f"Critical risk: {(scored['risk_level'] == 'critical').sum()}")
    
    # Current assessment
    assessment = scorer.get_current_assessment(data['geopolitical'])
    print("\nCurrent Assessment:")
    for key, value in assessment.items():
        print(f"  {key}: {value}")
    
    # Risk spikes
    spike_analysis = scorer.detect_risk_spikes(data['geopolitical'])
    spike_count = spike_analysis['risk_spike'].sum()
    print(f"\nDetected {spike_count} risk spike events")
