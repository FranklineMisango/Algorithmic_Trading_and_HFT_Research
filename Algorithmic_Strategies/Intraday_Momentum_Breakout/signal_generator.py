"""
Signal Generation Module

Generates trading signals based on noise area breakouts with confirmation.

Signal Logic:
- LONG: Price breaks above upper boundary with volume confirmation
- SHORT: Price breaks below lower boundary with volume confirmation
- EXIT: Price re-enters noise area (momentum failure) OR session close

Key Features:
- Confirmation bars to avoid false breakouts
- Volume filter to ensure liquidity
- Session time management (intraday only)
- Signal strength scoring
"""

import numpy as np
import pandas as pd
from datetime import time
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SignalGenerator:
    """
    Generates trading signals based on noise area breakouts.
    """
    
    def __init__(self, config: dict):
        """
        Initialize signal generator.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        
        # Entry/exit rules
        self.entry_rules = config['strategy']['entry_exit']
        self.confirmation_bars = self.entry_rules['confirmation_bars']
        self.volume_threshold = self.entry_rules['volume_threshold_percentile']
        
        # Get trailing stop setting from exit config if not in entry_exit
        self.trailing_stop_enabled = self.entry_rules.get('trailing_stop', 
                                                           config['strategy']['exit'].get('trailing_stop_enabled', False))
        
        # Session timing
        self.session_start = time(9, 30)  # 9:30 AM ET
        self.session_end = time(16, 0)    # 4:00 PM ET
        self.entry_cutoff = time(15, 0)   # Stop entering after 3:00 PM
        
        # Signal tracking
        self.current_position = 0  # 1 = long, -1 = short, 0 = flat
    
    def calculate_volume_percentile(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate rolling volume percentile.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
        window : int
            Rolling window for percentile
            
        Returns
        -------
        pd.Series
            Volume percentile (0-100)
        """
        volume_rank = data['Volume'].rolling(window=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        return volume_rank * 100
    
    def check_breakout_confirmation(self, data: pd.DataFrame, idx: int, direction: str) -> bool:
        """
        Check if breakout is confirmed over multiple bars.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with noise area boundaries
        idx : int
            Current bar index
        direction : str
            'long' or 'short'
            
        Returns
        -------
        bool
            True if breakout is confirmed
        """
        if idx < self.confirmation_bars:
            return False
        
        # Get recent bars
        recent_bars = data.iloc[idx - self.confirmation_bars + 1:idx + 1]
        
        if direction == 'long':
            # High pierced above the upper boundary on all recent bars
            # (matches how break_above is defined: High > upper_boundary)
            confirmed = (recent_bars['High'] > recent_bars['upper_boundary']).all()
        elif direction == 'short':
            # Low pierced below the lower boundary on all recent bars
            # (matches how break_below is defined: Low < lower_boundary)
            confirmed = (recent_bars['Low'] < recent_bars['lower_boundary']).all()
        else:
            confirmed = False
        
        return confirmed
    
    def check_volume_confirmation(self, data: pd.DataFrame, idx: int) -> bool:
        """
        Check if volume supports the breakout.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with volume percentile
        idx : int
            Current bar index
            
        Returns
        -------
        bool
            True if volume is sufficient
        """
        if 'volume_percentile' not in data.columns:
            return True  # Skip volume check if not available
        
        current_volume_pct = data.iloc[idx]['volume_percentile']
        return current_volume_pct >= self.volume_threshold
    
    def check_session_timing(self, timestamp: pd.Timestamp, allow_entry: bool = True) -> bool:
        """
        Check if we're within valid trading session.
        
        Parameters
        ----------
        timestamp : pd.Timestamp
            Current timestamp
        allow_entry : bool
            If True, check if we can enter new positions
            If False, only check if within session
            
        Returns
        -------
        bool
            True if timing is valid
        """
        # Convert to ET so the comparison works regardless of the index timezone
        try:
            ts_et = timestamp.tz_convert('America/New_York')
        except (TypeError, AttributeError):
            ts_et = timestamp  # already naive / already ET
        current_time = ts_et.time()

        if allow_entry:
            # Can enter only before cutoff
            return self.session_start <= current_time < self.entry_cutoff
        else:
            # Just check if within session
            return self.session_start <= current_time <= self.session_end
    
    def calculate_signal_strength(self, data: pd.DataFrame, idx: int, direction: str) -> float:
        """
        Calculate signal strength score (0-100).
        
        Factors:
        - Distance from boundary
        - Volume confirmation
        - Recent momentum
        - Time of day
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with indicators
        idx : int
            Current bar index
        direction : str
            'long' or 'short'
            
        Returns
        -------
        float
            Signal strength (0-100)
        """
        score = 0.0
        row = data.iloc[idx]
        
        # 1. Distance from boundary (max 30 points)
        if direction == 'long':
            boundary_distance = row['Close'] - row['upper_boundary']
            score += min(30, max(0, boundary_distance * 3))
        elif direction == 'short':
            boundary_distance = row['lower_boundary'] - row['Close']
            score += min(30, max(0, boundary_distance * 3))
        
        # 2. Volume confirmation (max 25 points)
        if 'volume_percentile' in data.columns:
            vol_pct = row['volume_percentile']
            score += (vol_pct / 100) * 25
        else:
            score += 12.5  # Neutral if no volume data
        
        # 3. Recent momentum (max 25 points)
        if idx >= 10:
            recent_returns = data['Close'].iloc[idx-10:idx].pct_change().sum()
            if direction == 'long' and recent_returns > 0:
                score += min(25, recent_returns * 500)
            elif direction == 'short' and recent_returns < 0:
                score += min(25, abs(recent_returns) * 500)
        
        # 4. Time of day (max 20 points)
        current_time = row.name.time()
        hour = current_time.hour + current_time.minute / 60
        
        # Prefer trades in first half of session (9:30-12:30)
        if 9.5 <= hour < 12.5:
            score += 20
        elif 12.5 <= hour < 14.5:
            score += 10
        else:
            score += 5
        
        return min(100, max(0, score))
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with confirmation.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with noise area boundaries
            
        Returns
        -------
        pd.DataFrame
            Data with signal columns
        """
        print("="*60)
        print("SIGNAL GENERATION")
        print("="*60)
        
        # Calculate volume percentile
        data['volume_percentile'] = self.calculate_volume_percentile(data, window=20)
        
        # Initialize signal columns
        data['signal'] = 0  # 1 = long, -1 = short, 0 = no signal
        data['signal_strength'] = 0.0
        data['entry_signal'] = False
        data['exit_signal'] = False
        data['exit_reason'] = ''
        
        # Track position for exit logic
        current_position = 0
        entry_idx = None
        
        print(f"Processing {len(data)} bars...")
        
        for idx in range(len(data)):
            row = data.iloc[idx]
            timestamp = row.name
            
            # Check if we're in valid session
            in_session = self.check_session_timing(timestamp, allow_entry=False)
            can_enter = self.check_session_timing(timestamp, allow_entry=True)
            
            # Exit at session close
            if not in_session and current_position != 0:
                data.at[row.name, 'exit_signal'] = True
                data.at[row.name, 'exit_reason'] = 'session_close'
                data.at[row.name, 'signal'] = 0
                current_position = 0
                entry_idx = None
                continue
            
            # Check for exit: momentum failure (price re-enters noise area)
            if current_position != 0 and row['inside_noise']:
                data.at[row.name, 'exit_signal'] = True
                data.at[row.name, 'exit_reason'] = 'momentum_failure'
                data.at[row.name, 'signal'] = 0
                current_position = 0
                entry_idx = None
                continue
            
            # Check for trailing stop exit (if enabled)
            if current_position != 0 and self.trailing_stop_enabled:
                if current_position == 1:  # Long position
                    # Exit if price falls below lower boundary
                    if row['Close'] < row['lower_boundary']:
                        data.at[row.name, 'exit_signal'] = True
                        data.at[row.name, 'exit_reason'] = 'trailing_stop'
                        data.at[row.name, 'signal'] = 0
                        current_position = 0
                        entry_idx = None
                        continue
                elif current_position == -1:  # Short position
                    # Exit if price rises above upper boundary
                    if row['Close'] > row['upper_boundary']:
                        data.at[row.name, 'exit_signal'] = True
                        data.at[row.name, 'exit_reason'] = 'trailing_stop'
                        data.at[row.name, 'signal'] = 0
                        current_position = 0
                        entry_idx = None
                        continue
            
            # Entry logic (only if not in position and can enter)
            if current_position == 0 and can_enter:
                # Check for long signal
                if row['break_above']:
                    if self.check_breakout_confirmation(data, idx, 'long'):
                        if self.check_volume_confirmation(data, idx):
                            signal_strength = self.calculate_signal_strength(data, idx, 'long')
                            
                            data.at[row.name, 'signal'] = 1
                            data.at[row.name, 'signal_strength'] = signal_strength
                            data.at[row.name, 'entry_signal'] = True
                            current_position = 1
                            entry_idx = idx
                
                # Check for short signal
                elif row['break_below']:
                    if self.check_breakout_confirmation(data, idx, 'short'):
                        if self.check_volume_confirmation(data, idx):
                            signal_strength = self.calculate_signal_strength(data, idx, 'short')
                            
                            data.at[row.name, 'signal'] = -1
                            data.at[row.name, 'signal_strength'] = signal_strength
                            data.at[row.name, 'entry_signal'] = True
                            current_position = -1
                            entry_idx = idx
            
            # Maintain current position
            elif current_position != 0:
                data.at[row.name, 'signal'] = current_position
        
        # Statistics
        entry_signals = data['entry_signal'].sum()
        exit_signals = data['exit_signal'].sum()
        long_signals = (data['signal'] == 1).sum()
        short_signals = (data['signal'] == -1).sum()
        
        print(f"\nSignal Statistics:")
        print(f"  Entry signals: {entry_signals}")
        print(f"    Long: {(data['entry_signal'] & (data['signal'] == 1)).sum()}")
        print(f"    Short: {(data['entry_signal'] & (data['signal'] == -1)).sum()}")
        print(f"  Exit signals: {exit_signals}")
        
        if exit_signals > 0:
            exit_reasons = data[data['exit_signal']]['exit_reason'].value_counts()
            print(f"  Exit reasons:")
            for reason, count in exit_reasons.items():
                print(f"    {reason}: {count}")
        
        print(f"  Avg signal strength: {data[data['entry_signal']]['signal_strength'].mean():.1f}")
        
        print("="*60)
        
        return data
    
    def generate_portfolio_signals(self, es_data: pd.DataFrame, nq_data: pd.DataFrame) -> dict:
        """
        Generate signals for the full portfolio (ES momentum + NQ momentum + NQ long-only).
        
        Parameters
        ----------
        es_data : pd.DataFrame
            ES futures data with noise area
        nq_data : pd.DataFrame
            NQ futures data with noise area
            
        Returns
        -------
        dict
            Portfolio signals for each strategy
        """
        print("\n" + "="*60)
        print("PORTFOLIO SIGNAL GENERATION")
        print("="*60)
        
        # Generate momentum signals
        es_momentum = self.generate_signals(es_data.copy())
        nq_momentum = self.generate_signals(nq_data.copy())
        
        # NQ long-only: Always long, no signals needed
        # (This is handled in position sizing)
        
        portfolio = {
            'ES_momentum': es_momentum,
            'NQ_momentum': nq_momentum,
            'NQ_long_only': nq_data.copy()  # No signals, just data
        }
        
        print("\nPortfolio signal generation complete")
        print("="*60)
        
        return portfolio


def main():
    """
    Test signal generator.
    """
    import yaml
    from noise_area import NoiseAreaCalculator
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load real ES data
    print("Loading ES data...")
    data = pd.read_csv(
        'Data/ES_5min_RTH.csv',
        index_col='ts_event',
        parse_dates=True
    )
    print(f"  Loaded {len(data)} bars")
    
    # Calculate noise area
    calculator = NoiseAreaCalculator(config)
    data = calculator.calculate_noise_area(data)
    data = calculator.identify_breakouts(data)
    
    # Generate signals
    signal_gen = SignalGenerator(config)
    data = signal_gen.generate_signals(data)
    
    # Analyze signals
    trades = data[data['entry_signal']].copy()
    print(f"\nTrade Analysis:")
    print(f"  Total trades: {len(trades)}")
    print(f"  Long trades: {(trades['signal'] == 1).sum()}")
    print(f"  Short trades: {(trades['signal'] == -1).sum()}")
    print(f"  Avg signal strength: {trades['signal_strength'].mean():.1f}")
    
    # Save
    data.to_csv('results/signals_es.csv')
    print("\nSignals saved to results/signals_es.csv")


if __name__ == "__main__":
    main()
