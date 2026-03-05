"""
Multi-asset signal generation for Strait of Hormuz geopolitical strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict
import yaml
from shipping_monitor import ShippingMonitor
from geopolitical_scorer import GeopoliticalScorer


class SignalGenerator:
    """Generate trading signals across asset classes and regions."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.shipping_monitor = ShippingMonitor(config_path)
        self.geo_scorer = GeopoliticalScorer(config_path)
        self.risk_scaling = self.config['portfolio']['risk_scaling']
    
    def generate_master_signal(self, data: Dict) -> pd.DataFrame:
        """Generate master risk signal combining all sources."""
        # Analyze shipping traffic
        shipping_analyzed = self.shipping_monitor.detect_anomalies(data['shipping'])
        traffic_score = shipping_analyzed['traffic_signal'] * 100
        
        # Analyze geopolitical risk
        geo_analyzed = self.geo_scorer.classify_risk_level(data['geopolitical'])
        geo_score = geo_analyzed['composite_risk_score_smooth']
        
        # Market stress indicators
        macro = data['macro']
        vix_score = (macro['vix'] / 50 * 100).clip(upper=100)  # Normalize VIX
        
        # Oil price momentum (higher = more stress)
        oil_returns = macro['brent_oil'].pct_change(5)  # 5-day return
        oil_score = (oil_returns * 500).clip(0, 100)  # Positive moves only
        
        # Combine signals with weights
        master_signal = pd.DataFrame({
            'traffic_score': traffic_score,
            'geo_score': geo_score,
            'vix_score': vix_score,
            'oil_score': oil_score
        })
        
        # Weighted composite (0-100)
        weights = {
            'traffic_score': 0.35,
            'geo_score': 0.35,
            'vix_score': 0.15,
            'oil_score': 0.15
        }
        
        master_signal['composite_signal'] = sum(
            master_signal[col] * weight for col, weight in weights.items()
        )
        
        # Smooth signal
        master_signal['signal_smooth'] = master_signal['composite_signal'].ewm(span=5).mean()
        
        # Risk level classification
        master_signal['risk_level'] = self._classify_risk_level(master_signal['signal_smooth'])
        
        # Position sizing multiplier (0-1)
        master_signal['position_multiplier'] = master_signal['risk_level'].map(self.risk_scaling)
        
        return master_signal
    
    def _classify_risk_level(self, signal: pd.Series) -> pd.Series:
        """Classify signal into risk levels."""
        thresholds = self.config['signals']['risk_score_thresholds']
        
        risk_level = pd.Series('low', index=signal.index)
        risk_level[signal >= thresholds['medium']] = 'medium'
        risk_level[signal >= thresholds['high']] = 'high'
        risk_level[signal >= thresholds['critical']] = 'critical'
        
        return risk_level
    
    def generate_asset_signals(self, master_signal: pd.DataFrame, data: Dict) -> Dict[str, pd.DataFrame]:
        """Generate specific signals for each asset class."""
        signals = {}
        
        # Long Energy Equities (XLE, oil majors)
        signals['long_energy'] = master_signal[['signal_smooth', 'position_multiplier']].copy()
        signals['long_energy']['signal'] = signals['long_energy']['position_multiplier']
        
        # Long Defense Stocks
        signals['long_defense'] = master_signal[['signal_smooth', 'position_multiplier']].copy()
        signals['long_defense']['signal'] = signals['long_defense']['position_multiplier']
        
        # Long US Treasuries (flight to quality)
        signals['long_treasuries'] = master_signal[['signal_smooth', 'position_multiplier']].copy()
        signals['long_treasuries']['signal'] = signals['long_treasuries']['position_multiplier']
        
        # Short Transportation (airlines, shipping)
        signals['short_transport'] = master_signal[['signal_smooth', 'position_multiplier']].copy()
        signals['short_transport']['signal'] = -signals['short_transport']['position_multiplier']
        
        # Short Asian Equities (high oil import dependency)
        signals['short_asia'] = master_signal[['signal_smooth', 'position_multiplier']].copy()
        signals['short_asia']['signal'] = -signals['short_asia']['position_multiplier']
        
        # Long Oil Exporter Currencies (NOK, CAD)
        signals['long_fx_exporters'] = master_signal[['signal_smooth', 'position_multiplier']].copy()
        signals['long_fx_exporters']['signal'] = signals['long_fx_exporters']['position_multiplier']
        
        # Short Oil Importer Currencies (JPY, INR)
        signals['short_fx_importers'] = master_signal[['signal_smooth', 'position_multiplier']].copy()
        signals['short_fx_importers']['signal'] = -signals['short_fx_importers']['position_multiplier']
        
        # Short Emerging Market Bonds
        signals['short_em_bonds'] = master_signal[['signal_smooth', 'position_multiplier']].copy()
        signals['short_em_bonds']['signal'] = -signals['short_em_bonds']['position_multiplier']
        
        return signals
    
    def generate_regional_signals(self, master_signal: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate region-specific signals."""
        regional_signals = {}
        
        # US: Moderate impact (energy producer)
        regional_signals['us'] = master_signal.copy()
        regional_signals['us']['impact_multiplier'] = 0.7
        
        # Europe: High impact (energy importer)
        regional_signals['europe'] = master_signal.copy()
        regional_signals['europe']['impact_multiplier'] = 1.0
        
        # Asia: Maximum impact (major oil importers)
        regional_signals['asia'] = master_signal.copy()
        regional_signals['asia']['impact_multiplier'] = 1.2
        
        return regional_signals
    
    def get_current_signals(self, data: Dict) -> Dict:
        """Get current signal values across all assets."""
        master_signal = self.generate_master_signal(data)
        asset_signals = self.generate_asset_signals(master_signal, data)
        
        latest = master_signal.iloc[-1]
        
        current_signals = {
            'date': master_signal.index[-1],
            'composite_signal': latest['signal_smooth'],
            'risk_level': latest['risk_level'],
            'position_multiplier': latest['position_multiplier'],
            'asset_signals': {
                name: sig.iloc[-1]['signal'] 
                for name, sig in asset_signals.items()
            }
        }
        
        return current_signals


if __name__ == "__main__":
    # Test signal generator
    from data_acquisition import DataAcquisition
    
    print("Testing Signal Generator...")
    
    acq = DataAcquisition()
    data = acq.fetch_all_data()
    
    generator = SignalGenerator()
    
    # Generate master signal
    master_signal = generator.generate_master_signal(data)
    
    print("\nMaster Signal Analysis:")
    print(f"Total days: {len(master_signal)}")
    print(f"Low risk: {(master_signal['risk_level'] == 'low').sum()}")
    print(f"Medium risk: {(master_signal['risk_level'] == 'medium').sum()}")
    print(f"High risk: {(master_signal['risk_level'] == 'high').sum()}")
    print(f"Critical risk: {(master_signal['risk_level'] == 'critical').sum()}")
    
    # Current signals
    current = generator.get_current_signals(data)
    print("\nCurrent Signals:")
    print(f"Date: {current['date']}")
    print(f"Composite Signal: {current['composite_signal']:.2f}")
    print(f"Risk Level: {current['risk_level']}")
    print(f"Position Multiplier: {current['position_multiplier']:.2f}")
    
    print("\nAsset Signals:")
    for asset, signal in current['asset_signals'].items():
        print(f"  {asset}: {signal:+.3f}")
