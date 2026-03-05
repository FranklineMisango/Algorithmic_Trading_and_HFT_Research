"""
Shipping traffic monitoring and anomaly detection for Strait of Hormuz.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import yaml


class ShippingMonitor:
    """Monitor shipping traffic and detect anomalies."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.normal_transits = self.config['signals']['shipping_threshold']['normal_daily_transits']
        self.alert_threshold = self.config['signals']['shipping_threshold']['alert_reduction']
        self.crisis_threshold = self.config['signals']['shipping_threshold']['crisis_reduction']
    
    def calculate_baseline(self, shipping_data: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Calculate rolling baseline for normal traffic."""
        df = shipping_data.copy()
        
        # Rolling mean and std
        df['baseline_mean'] = df['tanker_transits'].rolling(window=window, min_periods=1).mean()
        df['baseline_std'] = df['tanker_transits'].rolling(window=window, min_periods=1).std()
        
        # Z-score
        df['z_score'] = (df['tanker_transits'] - df['baseline_mean']) / (df['baseline_std'] + 1e-6)
        
        return df
    
    def detect_anomalies(self, shipping_data: pd.DataFrame) -> pd.DataFrame:
        """Detect traffic anomalies and classify severity."""
        df = self.calculate_baseline(shipping_data)
        
        # Calculate reduction from baseline
        df['reduction_pct'] = (df['baseline_mean'] - df['tanker_transits']) / df['baseline_mean']
        df['reduction_pct'] = df['reduction_pct'].clip(lower=0)  # Only care about reductions
        
        # Classify severity
        df['alert_level'] = 'normal'
        df.loc[df['reduction_pct'] >= self.alert_threshold, 'alert_level'] = 'alert'
        df.loc[df['reduction_pct'] >= self.crisis_threshold, 'alert_level'] = 'crisis'
        
        # Traffic signal (0-1, where 1 = maximum crisis)
        df['traffic_signal'] = df['reduction_pct'].clip(upper=1.0)
        
        return df
    
    def calculate_traffic_score(self, shipping_data: pd.DataFrame) -> pd.Series:
        """Calculate normalized traffic disruption score (0-100)."""
        df = self.detect_anomalies(shipping_data)
        
        # Convert to 0-100 scale
        traffic_score = df['traffic_signal'] * 100
        
        return traffic_score
    
    def get_current_status(self, shipping_data: pd.DataFrame) -> Dict:
        """Get current shipping status summary."""
        df = self.detect_anomalies(shipping_data)
        latest = df.iloc[-1]
        
        status = {
            'date': df.index[-1],
            'current_transits': latest['tanker_transits'],
            'baseline_transits': latest['baseline_mean'],
            'reduction_pct': latest['reduction_pct'] * 100,
            'alert_level': latest['alert_level'],
            'traffic_score': latest['traffic_signal'] * 100,
            'z_score': latest['z_score']
        }
        
        return status
    
    def analyze_crisis_periods(self, shipping_data: pd.DataFrame) -> pd.DataFrame:
        """Identify and analyze crisis periods."""
        df = self.detect_anomalies(shipping_data)
        
        # Find crisis periods
        crisis_mask = df['alert_level'] == 'crisis'
        
        # Group consecutive crisis days
        df['crisis_group'] = (crisis_mask != crisis_mask.shift()).cumsum()
        df.loc[~crisis_mask, 'crisis_group'] = np.nan
        
        # Analyze each crisis period
        crisis_periods = []
        for group_id in df['crisis_group'].dropna().unique():
            group_data = df[df['crisis_group'] == group_id]
            
            crisis_periods.append({
                'start_date': group_data.index[0],
                'end_date': group_data.index[-1],
                'duration_days': len(group_data),
                'avg_reduction': group_data['reduction_pct'].mean() * 100,
                'max_reduction': group_data['reduction_pct'].max() * 100,
                'avg_transits': group_data['tanker_transits'].mean()
            })
        
        return pd.DataFrame(crisis_periods)


if __name__ == "__main__":
    # Test shipping monitor
    from data_acquisition import DataAcquisition
    
    print("Testing Shipping Monitor...")
    
    acq = DataAcquisition()
    data = acq.fetch_all_data()
    
    monitor = ShippingMonitor()
    
    # Analyze shipping data
    analyzed = monitor.detect_anomalies(data['shipping'])
    
    print("\nShipping Traffic Analysis:")
    print(f"Total days: {len(analyzed)}")
    print(f"Normal days: {(analyzed['alert_level'] == 'normal').sum()}")
    print(f"Alert days: {(analyzed['alert_level'] == 'alert').sum()}")
    print(f"Crisis days: {(analyzed['alert_level'] == 'crisis').sum()}")
    
    # Current status
    status = monitor.get_current_status(data['shipping'])
    print("\nCurrent Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Crisis periods
    crisis_periods = monitor.analyze_crisis_periods(data['shipping'])
    if len(crisis_periods) > 0:
        print(f"\nIdentified {len(crisis_periods)} crisis periods:")
        print(crisis_periods.to_string())
