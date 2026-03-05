"""
PortWatch IMF Shipping Data Loader
Loads real shipping traffic data from IMF PortWatch CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class PortWatchLoader:
    """Load and process PortWatch IMF shipping data."""
    
    def __init__(self, data_dir: str = 'shipping_data_portwatch_imf'):
        """Initialize with data directory."""
        self.data_dir = Path(data_dir)
        self.arrivals_file = self.data_dir / 'arrivals-of-ships.csv'
        
        if not self.arrivals_file.exists():
            raise FileNotFoundError(
                f"PortWatch data not found at {self.arrivals_file}\n"
                f"Please download from: https://portwatch.imf.org/"
            )
    
    def load_arrivals(self) -> pd.DataFrame:
        """Load ship arrivals data."""
        print("Loading PortWatch IMF shipping data...")
        
        # Read CSV
        df = pd.read_csv(self.arrivals_file)
        
        # Parse datetime
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        print(f"✓ Loaded {len(df)} days of shipping data")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Vessel types: {', '.join([c for c in df.columns if c not in ['7-day Moving Average', 'Prior Year: 7-day Moving Average']])}")
        
        return df
    
    def get_tanker_traffic(self, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get tanker traffic data for specified date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD), defaults to data start
            end_date: End date (YYYY-MM-DD), defaults to data end
        
        Returns:
            DataFrame with tanker traffic and related metrics
        """
        df = self.load_arrivals()
        
        # Filter date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Create output dataframe focused on tankers
        output = pd.DataFrame(index=df.index)
        
        # Tanker transits (primary focus for oil/gas)
        output['tanker_transits'] = df['Tanker']
        
        # Other vessel types (for context)
        output['container_transits'] = df['Container']
        output['dry_bulk_transits'] = df['Dry Bulk']
        output['general_cargo_transits'] = df['General Cargo']
        output['roro_transits'] = df['Roll-on/roll-off']
        
        # Total transits
        output['total_transits'] = (
            output['tanker_transits'] + 
            output['container_transits'] + 
            output['dry_bulk_transits'] +
            output['general_cargo_transits'] +
            output['roro_transits']
        )
        
        # LNG carriers (estimate as ~40% of tankers based on typical mix)
        output['lng_transits'] = output['tanker_transits'] * 0.4
        
        # Add moving averages if available
        if '7-day Moving Average' in df.columns:
            output['ma_7day'] = df['7-day Moving Average']
        
        if 'Prior Year: 7-day Moving Average' in df.columns:
            output['ma_7day_prior_year'] = df['Prior Year: 7-day Moving Average']
        
        return output
    
    def get_summary_statistics(self) -> dict:
        """Get summary statistics for the shipping data."""
        df = self.load_arrivals()
        
        stats = {
            'total_days': len(df),
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'tanker_stats': {
                'mean': df['Tanker'].mean(),
                'median': df['Tanker'].median(),
                'std': df['Tanker'].std(),
                'min': df['Tanker'].min(),
                'max': df['Tanker'].max(),
                'percentile_25': df['Tanker'].quantile(0.25),
                'percentile_75': df['Tanker'].quantile(0.75)
            },
            'total_traffic_stats': {
                'mean': df[['Container', 'Dry Bulk', 'General Cargo', 'Roll-on/roll-off', 'Tanker']].sum(axis=1).mean(),
                'median': df[['Container', 'Dry Bulk', 'General Cargo', 'Roll-on/roll-off', 'Tanker']].sum(axis=1).median(),
                'std': df[['Container', 'Dry Bulk', 'General Cargo', 'Roll-on/roll-off', 'Tanker']].sum(axis=1).std()
            }
        }
        
        return stats
    
    def detect_anomalies(self, threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect anomalies in tanker traffic using z-score method.
        
        Args:
            threshold: Z-score threshold for anomaly detection (default: 2.0)
        
        Returns:
            DataFrame with anomaly flags
        """
        df = self.load_arrivals()
        
        # Calculate z-scores for tanker traffic
        tanker_mean = df['Tanker'].rolling(window=30, min_periods=1).mean()
        tanker_std = df['Tanker'].rolling(window=30, min_periods=1).std()
        
        z_scores = (df['Tanker'] - tanker_mean) / (tanker_std + 1e-6)
        
        # Flag anomalies
        anomalies = pd.DataFrame(index=df.index)
        anomalies['tanker_transits'] = df['Tanker']
        anomalies['z_score'] = z_scores
        anomalies['is_anomaly'] = np.abs(z_scores) > threshold
        anomalies['anomaly_type'] = 'normal'
        anomalies.loc[z_scores > threshold, 'anomaly_type'] = 'spike'
        anomalies.loc[z_scores < -threshold, 'anomaly_type'] = 'drop'
        
        return anomalies
    
    def compare_to_prior_year(self) -> pd.DataFrame:
        """Compare current traffic to prior year."""
        df = self.load_arrivals()
        
        if '7-day Moving Average' not in df.columns or 'Prior Year: 7-day Moving Average' not in df.columns:
            print("⚠ Prior year comparison data not available")
            return pd.DataFrame()
        
        comparison = pd.DataFrame(index=df.index)
        comparison['current_ma'] = df['7-day Moving Average']
        comparison['prior_year_ma'] = df['Prior Year: 7-day Moving Average']
        comparison['yoy_change'] = comparison['current_ma'] - comparison['prior_year_ma']
        comparison['yoy_change_pct'] = (comparison['yoy_change'] / comparison['prior_year_ma']) * 100
        
        # Remove NaN values
        comparison = comparison.dropna()
        
        return comparison


if __name__ == "__main__":
    # Test PortWatch loader
    print("\n" + "="*70)
    print("PORTWATCH IMF SHIPPING DATA LOADER - TEST")
    print("="*70 + "\n")
    
    try:
        loader = PortWatchLoader()
        
        # Load data
        tanker_data = loader.get_tanker_traffic()
        
        print("\nTanker Traffic Data:")
        print(f"  Shape: {tanker_data.shape}")
        print(f"  Columns: {list(tanker_data.columns)}")
        print(f"\nFirst 5 days:")
        print(tanker_data.head())
        print(f"\nLast 5 days:")
        print(tanker_data.tail())
        
        # Summary statistics
        print("\n" + "-"*70)
        print("SUMMARY STATISTICS")
        print("-"*70)
        stats = loader.get_summary_statistics()
        print(f"\nData Coverage:")
        print(f"  Total days: {stats['total_days']}")
        print(f"  Start: {stats['start_date'].date()}")
        print(f"  End: {stats['end_date'].date()}")
        
        print(f"\nTanker Traffic Statistics:")
        for key, value in stats['tanker_stats'].items():
            print(f"  {key.capitalize()}: {value:.2f}")
        
        # Anomaly detection
        print("\n" + "-"*70)
        print("ANOMALY DETECTION")
        print("-"*70)
        anomalies = loader.detect_anomalies(threshold=2.0)
        
        spike_count = (anomalies['anomaly_type'] == 'spike').sum()
        drop_count = (anomalies['anomaly_type'] == 'drop').sum()
        
        print(f"\nAnomalies detected (z-score > 2.0):")
        print(f"  Traffic spikes: {spike_count}")
        print(f"  Traffic drops: {drop_count}")
        
        if drop_count > 0:
            print(f"\nTop 5 traffic drops:")
            drops = anomalies[anomalies['anomaly_type'] == 'drop'].nsmallest(5, 'z_score')
            for date, row in drops.iterrows():
                print(f"  {date.date()}: {row['tanker_transits']:.0f} tankers (z-score: {row['z_score']:.2f})")
        
        # Year-over-year comparison
        print("\n" + "-"*70)
        print("YEAR-OVER-YEAR COMPARISON")
        print("-"*70)
        yoy = loader.compare_to_prior_year()
        
        if len(yoy) > 0:
            print(f"\nAverage YoY change: {yoy['yoy_change_pct'].mean():.2f}%")
            print(f"Max YoY increase: {yoy['yoy_change_pct'].max():.2f}%")
            print(f"Max YoY decrease: {yoy['yoy_change_pct'].min():.2f}%")
            
            # Find periods with significant YoY drops
            significant_drops = yoy[yoy['yoy_change_pct'] < -20]
            if len(significant_drops) > 0:
                print(f"\nPeriods with >20% YoY drop: {len(significant_drops)} days")
                print("Top 5 largest drops:")
                for date, row in significant_drops.nsmallest(5, 'yoy_change_pct').iterrows():
                    print(f"  {date.date()}: {row['yoy_change_pct']:.1f}% drop")
        
        print("\n" + "="*70)
        print("✓ PortWatch data loaded successfully!")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}\n")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
