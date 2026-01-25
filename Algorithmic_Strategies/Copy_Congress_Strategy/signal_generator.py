"""
Signal Generation Module for Copy Congress Strategy

Aggregates Congressional trade flows and generates trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SignalGenerator:
    """Generate trading signals from Congressional trade data."""
    
    def __init__(self, config: Dict):
        """
        Initialize signal generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.lookback_days = config['signal']['lookback_days']
        self.min_transaction_size = config['signal']['min_transaction_size']
        self.use_filing_date = config['signal']['use_filing_date']
        
        # Feature engineering flags
        self.use_committee_weighting = config['feature_engineering']['committee_weighting']
        self.use_bipartisan_filter = config['feature_engineering']['bipartisan_filter']
        
    def aggregate_trade_flows(self, 
                              congressional_trades: pd.DataFrame,
                              prices: pd.DataFrame,
                              as_of_date: pd.Timestamp) -> pd.DataFrame:
        """
        Aggregate Congressional buy/sell flows for signal generation.
        
        Args:
            congressional_trades: DataFrame with Congressional trades
            prices: DataFrame with prices
            as_of_date: Date to calculate signal for
            
        Returns:
            DataFrame with aggregated signals per ticker
        """
        # Determine which date field to use
        date_field = 'filing_date' if self.use_filing_date else 'transaction_date'
        
        # Filter trades within lookback window
        start_date = as_of_date - pd.Timedelta(days=self.lookback_days)
        trades_window = congressional_trades[
            (congressional_trades[date_field] > start_date) &
            (congressional_trades[date_field] <= as_of_date)
        ].copy()
        
        if len(trades_window) == 0:
            return pd.DataFrame()
        
        # Calculate net flows per ticker
        signals = []
        
        for ticker in trades_window['ticker'].unique():
            ticker_trades = trades_window[trades_window['ticker'] == ticker].copy()
            
            # Calculate buy/sell amounts
            buy_trades = ticker_trades[ticker_trades['transaction_type'] == 'buy']
            sell_trades = ticker_trades[ticker_trades['transaction_type'] == 'sell']
            
            total_buy = buy_trades['amount'].sum()
            total_sell = sell_trades['amount'].sum()
            
            # Net flow
            net_flow = total_buy - total_sell
            total_flow = total_buy + total_sell
            
            # Number of transactions
            n_buys = len(buy_trades)
            n_sells = len(sell_trades)
            n_total = n_buys + n_sells
            
            # Unique politicians
            n_politicians = ticker_trades['politician'].nunique()
            
            # Committee weighting
            committee_score = 1.0
            if self.use_committee_weighting:
                committee_score = self._calculate_committee_score(ticker_trades)
            
            # Bipartisan filter
            bipartisan_score = 1.0
            if self.use_bipartisan_filter:
                bipartisan_score = self._calculate_bipartisan_score(ticker_trades)
            
            signals.append({
                'ticker': ticker,
                'total_buy': total_buy,
                'total_sell': total_sell,
                'net_flow': net_flow,
                'total_flow': total_flow,
                'n_buys': n_buys,
                'n_sells': n_sells,
                'n_total': n_total,
                'n_politicians': n_politicians,
                'committee_score': committee_score,
                'bipartisan_score': bipartisan_score
            })
        
        signals_df = pd.DataFrame(signals)
        
        # Calculate signal strength
        signals_df['signal_raw'] = signals_df['net_flow']
        signals_df['signal_normalized'] = signals_df['net_flow'] / (signals_df['total_flow'] + 1e-8)
        
        # Apply weighting adjustments
        signals_df['signal_weighted'] = (
            signals_df['signal_normalized'] *
            signals_df['committee_score'] *
            signals_df['bipartisan_score']
        )
        
        # Rank signals
        signals_df['signal_rank'] = signals_df['signal_weighted'].rank(ascending=False)
        
        return signals_df
    
    def _calculate_committee_score(self, trades: pd.DataFrame) -> float:
        """
        Calculate committee importance score.
        
        Key committees get higher weight:
        - Finance/Banking: 1.5x
        - Technology/Energy: 1.3x
        - Others: 1.0x
        
        Args:
            trades: DataFrame with trades for a ticker
            
        Returns:
            Committee score
        """
        committee_weights = {
            'Finance': 1.5,
            'Banking': 1.5,
            'Technology': 1.3,
            'Energy': 1.3,
            'Healthcare': 1.2,
            'Defense': 1.2
        }
        
        # Calculate weighted average
        scores = []
        for _, trade in trades.iterrows():
            committee = trade['committee']
            weight = committee_weights.get(committee, 1.0)
            scores.append(weight)
        
        return np.mean(scores) if scores else 1.0
    
    def _calculate_bipartisan_score(self, trades: pd.DataFrame) -> float:
        """
        Calculate bipartisan agreement score.
        
        Higher score if both Democrats and Republicans are trading same direction.
        
        Args:
            trades: DataFrame with trades for a ticker
            
        Returns:
            Bipartisan score (1.0 to 1.5)
        """
        if 'party' not in trades.columns:
            return 1.0
        
        # Count buy/sell by party
        dem_trades = trades[trades['party'] == 'Democrat']
        rep_trades = trades[trades['party'] == 'Republican']
        
        if len(dem_trades) == 0 or len(rep_trades) == 0:
            return 1.0  # No bipartisan signal
        
        # Net flows by party
        dem_net = (
            dem_trades[dem_trades['transaction_type'] == 'buy']['amount'].sum() -
            dem_trades[dem_trades['transaction_type'] == 'sell']['amount'].sum()
        )
        
        rep_net = (
            rep_trades[rep_trades['transaction_type'] == 'buy']['amount'].sum() -
            rep_trades[rep_trades['transaction_type'] == 'sell']['amount'].sum()
        )
        
        # Same direction = higher score
        if (dem_net > 0 and rep_net > 0) or (dem_net < 0 and rep_net < 0):
            return 1.5  # Strong bipartisan agreement
        else:
            return 0.8  # Disagreement = lower confidence
    
    def generate_signals_timeseries(self,
                                   congressional_trades: pd.DataFrame,
                                   prices: pd.DataFrame,
                                   rebalance_dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Generate signals for all rebalance dates.
        
        Args:
            congressional_trades: DataFrame with Congressional trades
            prices: DataFrame with prices
            rebalance_dates: List of dates to generate signals
            
        Returns:
            Dictionary mapping dates to signal DataFrames
        """
        print(f"Generating signals for {len(rebalance_dates)} rebalance dates...")
        
        signals_history = {}
        
        for i, date in enumerate(rebalance_dates):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(rebalance_dates)} ({i/len(rebalance_dates)*100:.1f}%)")
            
            signals = self.aggregate_trade_flows(congressional_trades, prices, date)
            
            if len(signals) > 0:
                signals_history[date] = signals
        
        print(f"Generated signals for {len(signals_history)} dates")
        
        return signals_history
    
    def filter_top_signals(self, 
                          signals: pd.DataFrame,
                          max_holdings: int,
                          min_signal_threshold: float = 0.0) -> pd.DataFrame:
        """
        Filter to top signals only.
        
        Args:
            signals: DataFrame with signals
            max_holdings: Maximum number of positions
            min_signal_threshold: Minimum signal strength
            
        Returns:
            Filtered DataFrame
        """
        # Filter by minimum signal
        signals = signals[signals['signal_weighted'] >= min_signal_threshold].copy()
        
        # Sort by signal strength
        signals = signals.sort_values('signal_weighted', ascending=False)
        
        # Take top N
        signals = signals.head(max_holdings)
        
        return signals
    
    def analyze_signal_quality(self, signals_history: Dict[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame:
        """
        Analyze signal quality metrics.
        
        Args:
            signals_history: Dictionary of signals over time
            
        Returns:
            DataFrame with quality metrics
        """
        print("\nAnalyzing signal quality...")
        
        metrics = []
        
        for date, signals in signals_history.items():
            if len(signals) == 0:
                continue
            
            metrics.append({
                'date': date,
                'n_signals': len(signals),
                'avg_net_flow': signals['net_flow'].mean(),
                'avg_total_flow': signals['total_flow'].mean(),
                'avg_politicians': signals['n_politicians'].mean(),
                'avg_committee_score': signals['committee_score'].mean(),
                'avg_bipartisan_score': signals['bipartisan_score'].mean(),
                'pct_buy_signals': (signals['net_flow'] > 0).sum() / len(signals)
            })
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.set_index('date')
        
        # Summary statistics
        print("\nSignal Quality Summary:")
        print(f"  Average signals per period: {metrics_df['n_signals'].mean():.1f}")
        print(f"  Average net flow: ${metrics_df['avg_net_flow'].mean():,.0f}")
        print(f"  Average politicians per signal: {metrics_df['avg_politicians'].mean():.1f}")
        print(f"  Buy signal ratio: {metrics_df['pct_buy_signals'].mean():.1%}")
        
        return metrics_df


if __name__ == "__main__":
    # Test the module
    import yaml
    from data_acquisition import CongressionalDataAcquisition
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get data
    data_acq = CongressionalDataAcquisition(config)
    congressional_trades, prices, volumes, market_caps, volatility = data_acq.get_full_dataset()
    
    # Generate signals
    signal_gen = SignalGenerator(config)
    
    # Create rebalance dates (weekly)
    rebalance_dates = pd.date_range(
        start=prices.index[0],
        end=prices.index[-1],
        freq='W-FRI'
    )
    
    # Generate signals
    signals_history = signal_gen.generate_signals_timeseries(
        congressional_trades,
        prices,
        rebalance_dates.tolist()
    )
    
    # Analyze quality
    quality_metrics = signal_gen.analyze_signal_quality(signals_history)
    
    print("\n" + "="*60)
    print("Signal Generation Complete!")
    print("="*60)
    
    # Show example signals
    if len(signals_history) > 0:
        example_date = list(signals_history.keys())[len(signals_history)//2]
        print(f"\nExample signals for {example_date}:")
        print(signals_history[example_date].head(10))
