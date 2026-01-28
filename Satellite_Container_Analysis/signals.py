"""
Improved Trading Signal Generator
Fixes: Temporal smoothing, statistical validation, confidence scoring
"""
import pandas as pd
import numpy as np
from scipy import stats

class ImprovedSignalGenerator:
    """Generate robust trading signals with validation"""
    
    def __init__(self, lookback_days=30, signal_threshold=0.15):
        self.lookback_days = lookback_days
        self.signal_threshold = signal_threshold
        
    def calculate_signals(self, container_data):
        """Generate signals with temporal smoothing"""
        signals = []
        
        for port in container_data['port'].unique():
            port_data = container_data[container_data['port'] == port].copy()
            port_data = port_data.sort_values('date')
            
            # Temporal smoothing
            port_data['ma_7'] = port_data['container_ship_count'].rolling(7, min_periods=1).mean()
            port_data['ma_30'] = port_data['container_ship_count'].rolling(30, min_periods=1).mean()
            port_data['std_30'] = port_data['container_ship_count'].rolling(30, min_periods=1).std()
            
            # Calculate z-score for anomaly detection
            port_data['z_score'] = (port_data['container_ship_count'] - port_data['ma_30']) / port_data['std_30']
            
            # Percentage change from baseline
            port_data['pct_change'] = (port_data['ma_7'] - port_data['ma_30']) / port_data['ma_30']
            
            # Generate signals
            port_data['signal'] = 0
            port_data.loc[port_data['pct_change'] > self.signal_threshold, 'signal'] = 1  # Long
            port_data.loc[port_data['pct_change'] < -self.signal_threshold, 'signal'] = -1  # Short
            
            # Confidence based on z-score
            port_data['confidence'] = np.clip(abs(port_data['z_score']) / 3, 0, 1)
            
            # Statistical significance
            port_data['p_value'] = port_data['z_score'].apply(
                lambda z: 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else 1
            )
            
            # Filter by significance
            port_data.loc[port_data['p_value'] > 0.05, 'signal'] = 0
            
            signals.append(port_data)
        
        return pd.concat(signals, ignore_index=True)
    
    def generate_global_signal(self, port_signals):
        """Aggregate signals across all ports"""
        global_signals = port_signals.groupby('date').agg({
            'container_ship_count': 'sum',
            'signal': 'mean',
            'confidence': 'mean',
            'p_value': 'min'
        }).reset_index()
        
        # Global signal based on weighted average
        global_signals['global_signal'] = 0
        global_signals.loc[global_signals['signal'] > 0.3, 'global_signal'] = 1
        global_signals.loc[global_signals['signal'] < -0.3, 'global_signal'] = -1
        
        # Only keep statistically significant signals
        global_signals.loc[global_signals['p_value'] > 0.05, 'global_signal'] = 0
        
        return global_signals
    
    def backtest_signals(self, signals, returns_data):
        """Backtest trading signals"""
        signals = signals.merge(returns_data, on='date', how='left')
        
        # Calculate strategy returns
        signals['strategy_return'] = signals['global_signal'].shift(1) * signals['market_return']
        
        # Performance metrics
        total_return = (1 + signals['strategy_return']).prod() - 1
        sharpe = signals['strategy_return'].mean() / signals['strategy_return'].std() * np.sqrt(252)
        max_dd = (signals['strategy_return'].cumsum().cummax() - signals['strategy_return'].cumsum()).max()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': (signals['strategy_return'] > 0).mean()
        }
