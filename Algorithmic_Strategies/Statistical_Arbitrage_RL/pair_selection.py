"""
Pair Selection Module using Grid Search to Minimize EMRT

Systematically searches through candidate pairs to find those with fastest mean reversion.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from itertools import combinations
import yaml
from tqdm import tqdm

from emrt_calculator import EMRTCalculator


class PairSelector:
    """Select optimal stock pairs based on minimum EMRT."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pair_config = self.config['pair_selection']
        self.emrt_calculator = EMRTCalculator(config_path)
        
    def calculate_correlations(self, 
                                prices: pd.DataFrame,
                                window: int = None) -> pd.DataFrame:
        """
        Calculate pairwise correlations between all stocks.
        
        Args:
            prices: DataFrame with stock prices
            window: Lookback window (default from config)
            
        Returns:
            Correlation matrix
        """
        if window is None:
            window = self.pair_config['lookback_window']
        
        # Use only recent data for correlation
        recent_prices = prices.iloc[-window:]
        
        # Calculate returns
        returns = recent_prices.pct_change().dropna()
        
        # Correlation matrix
        corr_matrix = returns.corr()
        
        return corr_matrix
    
    def get_candidate_pairs(self, 
                            constituents: pd.DataFrame,
                            corr_matrix: pd.DataFrame) -> List[Tuple[str, str, str]]:
        """
        Generate candidate pairs based on sector and correlation.
        
        Args:
            constituents: DataFrame with ticker and sector info
            corr_matrix: Correlation matrix
            
        Returns:
            List of (ticker1, ticker2, sector) tuples
        """
        min_corr = self.pair_config['min_correlation']
        candidate_pairs = []
        
        # Group by sector
        for sector in constituents['sector'].unique():
            sector_tickers = constituents[constituents['sector'] == sector]['ticker'].tolist()
            
            # Generate all combinations within sector
            for ticker1, ticker2 in combinations(sector_tickers, 2):
                # Check if both tickers in correlation matrix
                if ticker1 not in corr_matrix.index or ticker2 not in corr_matrix.index:
                    continue
                
                # Check correlation threshold
                correlation = corr_matrix.loc[ticker1, ticker2]
                if correlation >= min_corr:
                    candidate_pairs.append((ticker1, ticker2, sector))
        
        return candidate_pairs
    
    def grid_search_pairs(self, 
                          prices: pd.DataFrame,
                          candidates: List[Tuple[str, str, str]]) -> pd.DataFrame:
        """
        Grid search over candidate pairs to calculate EMRT.
        
        Args:
            prices: Stock price DataFrame
            candidates: List of (ticker1, ticker2, sector) tuples
            
        Returns:
            DataFrame with pair info and EMRT scores
        """
        print(f"Grid searching {len(candidates)} candidate pairs...")
        
        results = []
        
        for ticker1, ticker2, sector in tqdm(candidates):
            # Calculate EMRT
            emrt, details = self.emrt_calculator.calculate_emrt(
                prices[ticker1],
                prices[ticker2]
            )
            
            # Calculate correlation
            corr = prices[[ticker1, ticker2]].pct_change().corr().iloc[0, 1]
            
            # Calculate volatility of spread
            spread = self.emrt_calculator.calculate_spread(
                prices[ticker1],
                prices[ticker2]
            )
            spread_vol = spread.std()
            
            results.append({
                'ticker1': ticker1,
                'ticker2': ticker2,
                'sector': sector,
                'emrt': emrt,
                'correlation': corr,
                'spread_volatility': spread_vol,
                'num_events': details['num_events'],
                'std_reversion_time': details.get('std_reversion_time', np.nan)
            })
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def select_top_pairs(self, 
                         pair_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Select top pairs with minimum EMRT.
        
        Args:
            pair_metrics: DataFrame with EMRT and other metrics
            
        Returns:
            Top N pairs based on EMRT percentile
        """
        max_pairs = self.pair_config['max_pairs']
        emrt_percentile = self.pair_config['emrt_percentile']
        
        # Remove pairs with infinite EMRT (no deviation events)
        valid_pairs = pair_metrics[pair_metrics['emrt'] < np.inf].copy()
        
        # Remove pairs with too few events
        valid_pairs = valid_pairs[valid_pairs['num_events'] >= 5]
        
        # Calculate EMRT percentile threshold
        threshold = np.percentile(valid_pairs['emrt'], emrt_percentile)
        
        # Select pairs below threshold
        selected = valid_pairs[valid_pairs['emrt'] <= threshold].copy()
        
        # Sort by EMRT and take top N
        selected = selected.sort_values('emrt').head(max_pairs)
        
        # Add pair identifier
        selected['pair_id'] = selected.apply(
            lambda row: f"{row['ticker1']}_{row['ticker2']}", 
            axis=1
        )
        
        return selected
    
    def run_selection(self, 
                      prices: pd.DataFrame,
                      constituents: pd.DataFrame) -> Dict:
        """
        Complete pair selection pipeline.
        
        Args:
            prices: Stock price DataFrame
            constituents: Constituent info with sectors
            
        Returns:
            Dictionary with selected pairs and metadata
        """
        print("=== Pair Selection Pipeline ===\n")
        
        # Step 1: Calculate correlations
        print("Step 1: Calculating correlations...")
        corr_matrix = self.calculate_correlations(prices)
        
        # Step 2: Generate candidate pairs
        print("Step 2: Generating candidate pairs...")
        candidates = self.get_candidate_pairs(constituents, corr_matrix)
        print(f"Found {len(candidates)} candidate pairs\n")
        
        # Step 3: Grid search EMRT
        print("Step 3: Grid searching EMRT...")
        pair_metrics = self.grid_search_pairs(prices, candidates)
        
        # Step 4: Select top pairs
        print("\nStep 4: Selecting top pairs...")
        selected_pairs = self.select_top_pairs(pair_metrics)
        
        print(f"\n=== Selected {len(selected_pairs)} pairs ===")
        print(selected_pairs[['pair_id', 'sector', 'emrt', 'correlation']])
        
        return {
            'selected_pairs': selected_pairs,
            'all_metrics': pair_metrics,
            'correlation_matrix': corr_matrix,
            'num_candidates': len(candidates)
        }


if __name__ == "__main__":
    # Test pair selection
    from data_acquisition import DataAcquisition
    
    # Fetch data
    print("Fetching data...")
    data_acq = DataAcquisition()
    dataset = data_acq.fetch_full_dataset()
    train_prices, _ = data_acq.split_train_test(dataset['prices'])
    
    # Run pair selection
    selector = PairSelector()
    selection_results = selector.run_selection(
        train_prices,
        dataset['constituents']
    )
    
    # Save results
    selection_results['selected_pairs'].to_csv('selected_pairs.csv', index=False)
    print("\nSelected pairs saved to selected_pairs.csv")
