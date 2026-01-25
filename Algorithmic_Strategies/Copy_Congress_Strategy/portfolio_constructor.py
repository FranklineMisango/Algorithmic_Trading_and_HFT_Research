"""
Portfolio Construction Module for Copy Congress Strategy

Implements inverse volatility weighting and portfolio constraints.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class PortfolioConstructor:
    """Construct portfolios using Congressional trade signals."""
    
    def __init__(self, config: Dict):
        """
        Initialize portfolio constructor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_holdings = config['portfolio']['min_holdings']
        self.max_holdings = config['portfolio']['max_holdings']
        self.max_position_size = config['portfolio']['max_position_size']
        self.weighting_method = config['weighting']['method']
        self.min_volatility = config['weighting']['min_volatility']
        self.max_sector_weight = config['portfolio']['max_sector_weight']
        
    def calculate_inverse_volatility_weights(self,
                                            tickers: List[str],
                                            volatility: pd.DataFrame,
                                            as_of_date: pd.Timestamp) -> pd.Series:
        """
        Calculate inverse volatility weights.
        
        Formula: Weight_i = (1/σ_i) / Σ(1/σ_j)
        
        Args:
            tickers: List of tickers to weight
            volatility: DataFrame with volatility data
            as_of_date: Date to calculate weights for
            
        Returns:
            Series with weights
        """
        # Get volatility as of date
        vol_date = volatility.index[volatility.index <= as_of_date][-1]
        vols = volatility.loc[vol_date, tickers]
        
        # Apply minimum volatility floor
        vols = vols.clip(lower=self.min_volatility)
        
        # Calculate inverse volatility weights
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()
        
        return weights
    
    def calculate_equal_weights(self, tickers: List[str]) -> pd.Series:
        """
        Calculate equal weights.
        
        Args:
            tickers: List of tickers
            
        Returns:
            Series with equal weights
        """
        n = len(tickers)
        weights = pd.Series(1.0 / n, index=tickers)
        return weights
    
    def calculate_signal_weights(self, signals: pd.DataFrame) -> pd.Series:
        """
        Calculate weights based on signal strength.
        
        Args:
            signals: DataFrame with signal data
            
        Returns:
            Series with signal-based weights
        """
        # Use signal strength for weighting
        signal_vals = signals.set_index('ticker')['signal_weighted']
        
        # Normalize to positive values
        signal_vals = signal_vals - signal_vals.min() + 1e-8
        
        # Calculate weights
        weights = signal_vals / signal_vals.sum()
        
        return weights
    
    def apply_position_limits(self, weights: pd.Series) -> pd.Series:
        """
        Apply position size limits.
        
        Args:
            weights: Series with raw weights
            
        Returns:
            Series with constrained weights
        """
        # Cap individual positions
        weights = weights.clip(upper=self.max_position_size)
        
        # Renormalize
        weights = weights / weights.sum()
        
        return weights
    
    def apply_holding_constraints(self, weights: pd.Series) -> pd.Series:
        """
        Apply minimum/maximum holding constraints.
        
        Args:
            weights: Series with weights
            
        Returns:
            Series with constrained weights
        """
        # Sort by weight
        weights_sorted = weights.sort_values(ascending=False)
        
        # Keep only top holdings
        if len(weights_sorted) > self.max_holdings:
            weights_sorted = weights_sorted.head(self.max_holdings)
        
        # Ensure minimum holdings
        if len(weights_sorted) < self.min_holdings:
            # If not enough signals, return empty
            return pd.Series()
        
        # Renormalize
        weights_sorted = weights_sorted / weights_sorted.sum()
        
        return weights_sorted
    
    def construct_portfolio(self,
                           signals: pd.DataFrame,
                           volatility: pd.DataFrame,
                           as_of_date: pd.Timestamp) -> pd.Series:
        """
        Construct portfolio weights.
        
        Args:
            signals: DataFrame with signals
            volatility: DataFrame with volatility
            as_of_date: Date to construct portfolio for
            
        Returns:
            Series with portfolio weights
        """
        if len(signals) == 0:
            return pd.Series()
        
        tickers = signals['ticker'].tolist()
        
        # Calculate base weights
        if self.weighting_method == 'inverse_volatility':
            weights = self.calculate_inverse_volatility_weights(tickers, volatility, as_of_date)
        elif self.weighting_method == 'equal':
            weights = self.calculate_equal_weights(tickers)
        elif self.weighting_method == 'signal':
            weights = self.calculate_signal_weights(signals)
        else:
            raise ValueError(f"Unknown weighting method: {self.weighting_method}")
        
        # Apply position limits
        weights = self.apply_position_limits(weights)
        
        # Apply holding constraints
        weights = self.apply_holding_constraints(weights)
        
        return weights
    
    def calculate_portfolio_turnover(self,
                                    old_weights: pd.Series,
                                    new_weights: pd.Series) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            old_weights: Previous portfolio weights
            new_weights: New portfolio weights
            
        Returns:
            Turnover as fraction
        """
        # Align weights
        all_tickers = list(set(old_weights.index.tolist() + new_weights.index.tolist()))
        
        old_aligned = pd.Series(0.0, index=all_tickers)
        old_aligned[old_weights.index] = old_weights
        
        new_aligned = pd.Series(0.0, index=all_tickers)
        new_aligned[new_weights.index] = new_weights
        
        # Calculate turnover
        turnover = (old_aligned - new_aligned).abs().sum() / 2.0
        
        return turnover
    
    def generate_portfolio_timeseries(self,
                                     signals_history: Dict[pd.Timestamp, pd.DataFrame],
                                     volatility: pd.DataFrame,
                                     rebalance_dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, pd.Series]:
        """
        Generate portfolio weights for all rebalance dates.
        
        Args:
            signals_history: Dictionary of signals over time
            volatility: DataFrame with volatility
            rebalance_dates: List of rebalance dates
            
        Returns:
            Dictionary mapping dates to portfolio weights
        """
        print(f"Constructing portfolios for {len(rebalance_dates)} rebalance dates...")
        
        portfolio_history = {}
        previous_weights = pd.Series()
        turnover_list = []
        
        for i, date in enumerate(rebalance_dates):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(rebalance_dates)} ({i/len(rebalance_dates)*100:.1f}%)")
            
            # Get signals for this date
            if date not in signals_history:
                # No signals, keep previous portfolio
                if len(previous_weights) > 0:
                    portfolio_history[date] = previous_weights
                continue
            
            signals = signals_history[date]
            
            # Construct portfolio
            weights = self.construct_portfolio(signals, volatility, date)
            
            if len(weights) == 0:
                # Failed constraints, keep previous
                if len(previous_weights) > 0:
                    portfolio_history[date] = previous_weights
                continue
            
            # Calculate turnover
            if len(previous_weights) > 0:
                turnover = self.calculate_portfolio_turnover(previous_weights, weights)
                turnover_list.append(turnover)
            
            # Store portfolio
            portfolio_history[date] = weights
            previous_weights = weights
        
        # Summary statistics
        if len(turnover_list) > 0:
            print(f"\nPortfolio Construction Summary:")
            print(f"  Portfolios created: {len(portfolio_history)}")
            print(f"  Average turnover: {np.mean(turnover_list):.1%}")
            print(f"  Max turnover: {np.max(turnover_list):.1%}")
            print(f"  Average holdings: {np.mean([len(w) for w in portfolio_history.values()]):.1f}")
        
        return portfolio_history
    
    def analyze_portfolio_characteristics(self, 
                                         portfolio_history: Dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
        """
        Analyze portfolio characteristics over time.
        
        Args:
            portfolio_history: Dictionary of portfolio weights
            
        Returns:
            DataFrame with portfolio metrics
        """
        print("\nAnalyzing portfolio characteristics...")
        
        metrics = []
        
        for date, weights in portfolio_history.items():
            metrics.append({
                'date': date,
                'n_holdings': len(weights),
                'max_weight': weights.max(),
                'min_weight': weights.min(),
                'avg_weight': weights.mean(),
                'weight_concentration': (weights ** 2).sum(),  # Herfindahl index
                'top5_weight': weights.nlargest(5).sum(),
                'top10_weight': weights.nlargest(min(10, len(weights))).sum()
            })
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.set_index('date')
        
        # Summary
        print("\nPortfolio Characteristics Summary:")
        print(f"  Average holdings: {metrics_df['n_holdings'].mean():.1f}")
        print(f"  Average max position: {metrics_df['max_weight'].mean():.1%}")
        print(f"  Average top 5 weight: {metrics_df['top5_weight'].mean():.1%}")
        print(f"  Average concentration: {metrics_df['weight_concentration'].mean():.3f}")
        
        return metrics_df


if __name__ == "__main__":
    # Test the module
    import yaml
    from data_acquisition import CongressionalDataAcquisition
    from signal_generator import SignalGenerator
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get data
    print("Loading data...")
    data_acq = CongressionalDataAcquisition(config)
    congressional_trades, prices, volumes, market_caps, volatility = data_acq.get_full_dataset()
    
    # Generate signals
    print("\nGenerating signals...")
    signal_gen = SignalGenerator(config)
    rebalance_dates = pd.date_range(
        start=prices.index[0],
        end=prices.index[-1],
        freq='W-FRI'
    )
    signals_history = signal_gen.generate_signals_timeseries(
        congressional_trades,
        prices,
        rebalance_dates.tolist()
    )
    
    # Construct portfolios
    print("\nConstructing portfolios...")
    portfolio_constructor = PortfolioConstructor(config)
    portfolio_history = portfolio_constructor.generate_portfolio_timeseries(
        signals_history,
        volatility,
        rebalance_dates.tolist()
    )
    
    # Analyze characteristics
    characteristics = portfolio_constructor.analyze_portfolio_characteristics(portfolio_history)
    
    print("\n" + "="*60)
    print("Portfolio Construction Complete!")
    print("="*60)
    
    # Show example portfolio
    if len(portfolio_history) > 0:
        example_date = list(portfolio_history.keys())[len(portfolio_history)//2]
        print(f"\nExample portfolio for {example_date}:")
        print(portfolio_history[example_date].sort_values(ascending=False).head(10))
