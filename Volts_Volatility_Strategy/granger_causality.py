"""
Granger Causality Testing Module

Implements Granger causality tests to identify predictive relationships
between volatility time series of different stocks.

If Stock A's volatility "Granger-causes" Stock B's volatility, then
changes in A's volatility can predict future changes in B's volatility.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, List, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


class GrangerCausalityAnalyzer:
    """
    Perform Granger causality tests on volatility time series.
    """
    
    def __init__(
        self,
        max_lag: int = 30,
        significance_level: float = 0.05
    ):
        """
        Initialize the Granger causality analyzer.
        
        Parameters:
        -----------
        max_lag : int
            Maximum lag to test (default: 30)
        significance_level : float
            Significance level for hypothesis test (default: 0.05)
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.causality_results = {}
        self.optimal_lags = {}
        
    def test_causality(
        self,
        x: pd.Series,
        y: pd.Series,
        max_lag: Optional[int] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Test if x Granger-causes y.
        
        H0: x does NOT Granger-cause y
        H1: x Granger-causes y (reject H0 if p-value < significance_level)
        
        Parameters:
        -----------
        x : pd.Series
            Predictor time series
        y : pd.Series
            Target time series
        max_lag : int, optional
            Maximum lag to test (uses instance default if not provided)
        verbose : bool
            Print detailed results
            
        Returns:
        --------
        Dict : Dictionary with causality test results
        """
        if max_lag is None:
            max_lag = self.max_lag
        
        # Align series and remove NaNs
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        if len(df) < max_lag + 10:
            raise ValueError(f"Insufficient data points ({len(df)}) for max_lag={max_lag}")
        
        # Run Granger causality test
        # Note: grangercausalitytests expects [y, x] order
        try:
            gc_results = grangercausalitytests(
                df[['y', 'x']], 
                maxlag=max_lag,
                verbose=False
            )
        except Exception as e:
            if verbose:
                print(f"Granger causality test failed: {e}")
            return {
                'causality_detected': False,
                'error': str(e)
            }
        
        # Extract F-test p-values for each lag
        lag_results = {}
        for lag in range(1, max_lag + 1):
            # Use F-test (ssr_ftest)
            f_test = gc_results[lag][0]['ssr_ftest']
            lag_results[lag] = {
                'f_statistic': f_test[0],
                'p_value': f_test[1],
                'significant': f_test[1] < self.significance_level
            }
        
        # Find optimal lag (lowest p-value among significant lags)
        significant_lags = [
            lag for lag, result in lag_results.items() 
            if result['significant']
        ]
        
        if significant_lags:
            optimal_lag = min(
                significant_lags,
                key=lambda l: lag_results[l]['p_value']
            )
            causality_detected = True
        else:
            optimal_lag = None
            causality_detected = False
        
        result = {
            'causality_detected': causality_detected,
            'optimal_lag': optimal_lag,
            'lag_results': lag_results,
            'n_significant_lags': len(significant_lags)
        }
        
        if verbose and causality_detected:
            print(f"Granger causality detected at lag {optimal_lag}")
            print(f"P-value: {lag_results[optimal_lag]['p_value']:.4f}")
        
        return result
    
    def test_all_pairs(
        self,
        volatility_df: pd.DataFrame,
        tickers: Optional[List[str]] = None,
        max_lag: Optional[int] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Test Granger causality for all pairs of tickers.
        
        Parameters:
        -----------
        volatility_df : pd.DataFrame
            DataFrame with volatility time series (tickers as columns)
        tickers : List[str], optional
            List of tickers to test (uses all if not provided)
        max_lag : int, optional
            Maximum lag to test
        verbose : bool
            Print progress
            
        Returns:
        --------
        pd.DataFrame : Summary of causality relationships
        """
        if tickers is None:
            tickers = volatility_df.columns.tolist()
        
        if max_lag is None:
            max_lag = self.max_lag
        
        # Test all directed pairs (A->B and B->A are different)
        results = []
        total_tests = len(tickers) * (len(tickers) - 1)
        test_count = 0
        
        for x_ticker in tickers:
            for y_ticker in tickers:
                if x_ticker == y_ticker:
                    continue
                
                test_count += 1
                if verbose and test_count % 10 == 0:
                    print(f"Testing pair {test_count}/{total_tests}...")
                
                try:
                    result = self.test_causality(
                        volatility_df[x_ticker],
                        volatility_df[y_ticker],
                        max_lag=max_lag,
                        verbose=False
                    )
                    
                    # Store key for future reference
                    key = (x_ticker, y_ticker)
                    self.causality_results[key] = result
                    self.optimal_lags[key] = result.get('optimal_lag')
                    
                    # Add to results list
                    if result['causality_detected']:
                        results.append({
                            'predictor': x_ticker,
                            'target': y_ticker,
                            'optimal_lag': result['optimal_lag'],
                            'p_value': result['lag_results'][result['optimal_lag']]['p_value'],
                            'f_statistic': result['lag_results'][result['optimal_lag']]['f_statistic'],
                            'n_significant_lags': result['n_significant_lags']
                        })
                except Exception as e:
                    if verbose:
                        print(f"Error testing {x_ticker} -> {y_ticker}: {e}")
        
        if not results:
            print("No significant Granger causality relationships found.")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('p_value').reset_index(drop=True)
        
        if verbose:
            print(f"\nFound {len(results_df)} significant causality relationships")
        
        return results_df
    
    def filter_by_lag(
        self,
        results_df: pd.DataFrame,
        target_lag: int = 5,
        lag_tolerance: int = 2
    ) -> pd.DataFrame:
        """
        Filter causality results to specific lag range.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from test_all_pairs
        target_lag : int
            Target lag value
        lag_tolerance : int
            Tolerance around target lag
            
        Returns:
        --------
        pd.DataFrame : Filtered results
        """
        filtered = results_df[
            (results_df['optimal_lag'] >= target_lag - lag_tolerance) &
            (results_df['optimal_lag'] <= target_lag + lag_tolerance)
        ].copy()
        
        print(f"Filtered to {len(filtered)} relationships with lag {target_lag} ± {lag_tolerance}")
        
        return filtered
    
    def remove_circular_relationships(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Remove circular causality relationships (A->B and B->A).
        Keep the relationship with lower p-value.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from test_all_pairs
            
        Returns:
        --------
        pd.DataFrame : Results with circular relationships removed
        """
        filtered_results = []
        processed_pairs = set()
        
        for _, row in results_df.iterrows():
            pred, tgt = row['predictor'], row['target']
            pair = tuple(sorted([pred, tgt]))
            
            if pair in processed_pairs:
                continue
            
            # Check if reverse relationship exists
            reverse = results_df[
                (results_df['predictor'] == tgt) & 
                (results_df['target'] == pred)
            ]
            
            if len(reverse) > 0:
                # Keep the one with lower p-value
                if row['p_value'] <= reverse.iloc[0]['p_value']:
                    filtered_results.append(row)
            else:
                filtered_results.append(row)
            
            processed_pairs.add(pair)
        
        filtered_df = pd.DataFrame(filtered_results).reset_index(drop=True)
        
        print(f"Removed circular relationships: {len(results_df)} -> {len(filtered_df)}")
        
        return filtered_df
    
    def build_causality_network(
        self,
        results_df: pd.DataFrame
    ) -> nx.DiGraph:
        """
        Build a directed network graph of causality relationships.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from test_all_pairs
            
        Returns:
        --------
        nx.DiGraph : Directed graph of causality relationships
        """
        G = nx.DiGraph()
        
        for _, row in results_df.iterrows():
            G.add_edge(
                row['predictor'],
                row['target'],
                lag=row['optimal_lag'],
                p_value=row['p_value'],
                f_statistic=row['f_statistic']
            )
        
        return G
    
    def plot_causality_network(
        self,
        results_df: pd.DataFrame,
        save_path: str = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Visualize causality relationships as a network graph.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results from test_all_pairs
        save_path : str, optional
            Path to save the plot
        figsize : Tuple[int, int]
            Figure size
        """
        G = self.build_causality_network(results_df)
        
        plt.figure(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=2000,
            alpha=0.9
        )
        
        # Draw edges with varying width based on significance
        for (u, v, d) in G.edges(data=True):
            # Edge width inversely proportional to p-value
            width = 1 + (1 - d['p_value']) * 4
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=width,
                alpha=0.6,
                edge_color='gray',
                arrowsize=20,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1'
            )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold'
        )
        
        # Draw edge labels (lag values)
        edge_labels = {
            (u, v): f"lag={d['lag']}" 
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels,
            font_size=8
        )
        
        plt.title(
            'Granger Causality Network\n(Arrow from X to Y means X Granger-causes Y)',
            fontsize=14,
            fontweight='bold'
        )
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network plot saved to {save_path}")
        else:
            plt.show()


def identify_trading_pairs(
    volatility_df: pd.DataFrame,
    mid_cluster_tickers: List[str],
    target_lag: int = 5,
    significance_level: float = 0.05,
    max_lag: int = 30,
    remove_circular: bool = True
) -> Tuple[pd.DataFrame, GrangerCausalityAnalyzer]:
    """
    Convenience function to identify trading pairs using Granger causality.
    
    Parameters:
    -----------
    volatility_df : pd.DataFrame
        Volatility time series for all assets
    mid_cluster_tickers : List[str]
        Tickers in the mid-volatility cluster
    target_lag : int
        Target lag for filtering
    significance_level : float
        Significance level for tests
    max_lag : int
        Maximum lag to test
    remove_circular : bool
        Whether to remove circular relationships
        
    Returns:
    --------
    Tuple[pd.DataFrame, GrangerCausalityAnalyzer] :
        Trading pairs and analyzer object
    """
    # Initialize analyzer
    analyzer = GrangerCausalityAnalyzer(
        max_lag=max_lag,
        significance_level=significance_level
    )
    
    # Test all pairs in mid-cluster
    print(f"Testing Granger causality for {len(mid_cluster_tickers)} assets...")
    results = analyzer.test_all_pairs(
        volatility_df[mid_cluster_tickers],
        verbose=True
    )
    
    if len(results) == 0:
        return results, analyzer
    
    # Filter by target lag
    results = analyzer.filter_by_lag(results, target_lag=target_lag)
    
    # Remove circular relationships
    if remove_circular and len(results) > 0:
        results = analyzer.remove_circular_relationships(results)
    
    print("\nIdentified Trading Pairs:")
    print(results.to_string(index=False))
    
    return results, analyzer


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    from volatility_estimators import calculate_volatility_for_assets
    from volatility_clustering import cluster_assets_by_volatility
    
    # Download data
    tickers = ['MSFT', 'GOOGL', 'NVDA', 'AMZN', 'META', 'QCOM', 'IBM', 'INTC', 'MU']
    
    print("Downloading data...")
    data_dict = {}
    for ticker in tickers:
        df = yf.download(ticker, start="2020-05-01", end="2023-05-31", progress=False)
        data_dict[ticker] = df
    
    # Calculate volatility
    print("\nCalculating volatility...")
    volatility_df = calculate_volatility_for_assets(
        data_dict,
        estimator='yang_zhang',
        rolling_window=20
    )
    
    # Cluster assets
    print("\nClustering assets...")
    clustering, mid_cluster_members = cluster_assets_by_volatility(
        volatility_df,
        n_clusters=3,
        target_cluster='mid'
    )
    
    # Identify trading pairs
    print("\n" + "="*60)
    print("GRANGER CAUSALITY ANALYSIS")
    print("="*60)
    
    trading_pairs, analyzer = identify_trading_pairs(
        volatility_df,
        mid_cluster_members,
        target_lag=5,
        max_lag=30
    )
    
    # Visualize network
    if len(trading_pairs) > 0:
        analyzer.plot_causality_network(trading_pairs)
