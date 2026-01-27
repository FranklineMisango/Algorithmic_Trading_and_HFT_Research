import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class TradingSignalGenerator:
    def __init__(self):
        self.scaler = StandardScaler()
        self.signals_history = pd.DataFrame()
        
    def calculate_port_signals(self, container_data):
        """Generate trading signals for individual ports"""
        signals = []
        
        for port in container_data['port'].unique():
            port_data = container_data[container_data['port'] == port].copy()
            port_data = port_data.sort_values('datetime')
            
            if len(port_data) < 7:
                continue
                
            # Technical indicators
            port_data['sma_7'] = port_data['container_count'].rolling(7).mean()
            port_data['sma_21'] = port_data['container_count'].rolling(21).mean()
            port_data['rsi'] = self._calculate_rsi(port_data['container_count'])
            port_data['volatility'] = port_data['container_count'].rolling(7).std()
            
            # Generate signals
            port_data['signal'] = 0
            
            # Bullish: Short MA > Long MA and RSI < 70
            bullish = (port_data['sma_7'] > port_data['sma_21']) & (port_data['rsi'] < 70)
            port_data.loc[bullish, 'signal'] = 1
            
            # Bearish: Short MA < Long MA and RSI > 30
            bearish = (port_data['sma_7'] < port_data['sma_21']) & (port_data['rsi'] > 30)
            port_data.loc[bearish, 'signal'] = -1
            
            # Warning for extreme volumes (potential bottleneck)
            high_threshold = port_data['container_count'].quantile(0.95)
            port_data['warning'] = port_data['container_count'] > high_threshold
            
            # Signal strength based on volume change and volatility
            port_data['signal_strength'] = abs(port_data['pct_change']) / (port_data['volatility'] + 1e-6)
            port_data['signal_strength'] = np.clip(port_data['signal_strength'], 0, 1)
            
            signals.append(port_data)
        
        return pd.concat(signals, ignore_index=True) if signals else pd.DataFrame()
    
    def generate_global_signal(self, port_signals):
        """Generate global trading signal from all ports"""
        if port_signals.empty:
            return pd.DataFrame()
        
        # Aggregate by date
        daily_signals = port_signals.groupby(port_signals['datetime'].dt.date).agg({
            'container_count': 'sum',
            'signal': 'mean',
            'signal_strength': 'mean',
            'warning': 'any',
            'pct_change': 'mean'
        }).reset_index()
        
        daily_signals['datetime'] = pd.to_datetime(daily_signals['datetime'])
        
        # Global signal logic
        daily_signals['global_signal'] = 0
        
        # Strong bullish: Average signal > 0.3 and high strength
        strong_bull = (daily_signals['signal'] > 0.3) & (daily_signals['signal_strength'] > 0.5)
        daily_signals.loc[strong_bull, 'global_signal'] = 2
        
        # Bullish: Average signal > 0.1
        bull = (daily_signals['signal'] > 0.1) & ~strong_bull
        daily_signals.loc[bull, 'global_signal'] = 1
        
        # Bearish: Average signal < -0.1
        bear = (daily_signals['signal'] < -0.1) & (daily_signals['global_signal'] == 0)
        daily_signals.loc[bear, 'global_signal'] = -1
        
        # Strong bearish: Average signal < -0.3 and high strength
        strong_bear = (daily_signals['signal'] < -0.3) & (daily_signals['signal_strength'] > 0.5)
        daily_signals.loc[strong_bear, 'global_signal'] = -2
        
        # Risk adjustment for warnings
        if daily_signals['warning'].any():
            daily_signals.loc[daily_signals['warning'], 'global_signal'] = daily_signals.loc[daily_signals['warning'], 'global_signal'].astype(float) * 0.5
        
        return daily_signals
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def backtest_signals(self, signals, returns_data=None):
        """Simple backtesting of generated signals"""
        if returns_data is None:
            # Generate synthetic market returns for demonstration
            np.random.seed(42)
            returns_data = pd.DataFrame({
                'datetime': signals['datetime'],
                'market_return': np.random.normal(0.001, 0.02, len(signals))
            })
        
        # Merge signals with returns
        backtest_data = signals.merge(returns_data, on='datetime', how='inner')
        
        # Calculate strategy returns
        backtest_data['strategy_return'] = backtest_data['global_signal'] * backtest_data['market_return']
        backtest_data['cumulative_return'] = (1 + backtest_data['strategy_return']).cumprod()
        backtest_data['market_cumulative'] = (1 + backtest_data['market_return']).cumprod()
        
        # Performance metrics
        total_return = backtest_data['cumulative_return'].iloc[-1] - 1
        market_return = backtest_data['market_cumulative'].iloc[-1] - 1
        sharpe_ratio = backtest_data['strategy_return'].mean() / backtest_data['strategy_return'].std() * np.sqrt(252)
        
        return {
            'backtest_data': backtest_data,
            'total_return': total_return,
            'market_return': market_return,
            'excess_return': total_return - market_return,
            'sharpe_ratio': sharpe_ratio
        }

def generate_trading_report(signals, backtest_results):
    """Generate comprehensive trading report"""
    report = {
        'summary': {
            'total_signals': len(signals),
            'bullish_signals': len(signals[signals['global_signal'] > 0]),
            'bearish_signals': len(signals[signals['global_signal'] < 0]),
            'warning_days': len(signals[signals['warning']]),
            'date_range': f"{signals['datetime'].min()} to {signals['datetime'].max()}"
        },
        'performance': backtest_results
    }
    
    return report