"""
Copula Analyzer - Phase 2
Models tail dependence and jump contagion using copulas
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from scipy import stats
from itertools import combinations
import logging

# Copulas library
try:
    from copulas.bivariate import Clayton, Gumbel, StudentT
    COPULAS_AVAILABLE = True
except ImportError:
    COPULAS_AVAILABLE = False
    logging.warning("copulas library not available. Install with: pip install copulas")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CopulaAnalyzer:
    """
    Analyzes jump contagion using copula-based tail dependence
    """
    
    def __init__(self, config: Dict):
        """
        Initialize copula analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.copula_config = config['copula_analysis']
        self.fitted_copulas = {}
        
        if not COPULAS_AVAILABLE:
            logger.warning("Copulas library not available. Using simplified correlation analysis.")
    
    def analyze_tail_dependence(self, df_with_jumps: pd.DataFrame) -> Dict:
        """
        Analyze tail dependence for all asset pairs
        
        Args:
            df_with_jumps: DataFrame with jump indicators
            
        Returns:
            Dictionary of tail dependence results
        """
        logger.info("Analyzing tail dependence with copulas...")
        
        # Get returns in wide format
        returns_wide = df_with_jumps.pivot(
            index='date', 
            columns='asset', 
            values='returns'
        ).dropna()
        
        assets = list(returns_wide.columns)
        
        # Analyze all pairs
        results = {}
        
        for asset1, asset2 in combinations(assets, 2):
            pair_name = f"{asset1}-{asset2}"
            
            X = returns_wide[asset1].values
            Y = returns_wide[asset2].values
            
            # Fit copulas
            copula_results = self._fit_copulas(X, Y, pair_name)
            
            results[pair_name] = copula_results
        
        logger.info(f"Analyzed {len(results)} asset pairs")
        
        return results
    
    def _fit_copulas(self, X: np.ndarray, Y: np.ndarray, pair_name: str) -> Dict:
        """
        Fit multiple copula models to asset pair
        
        Args:
            X: Returns for asset 1
            Y: Returns for asset 2
            pair_name: Name of asset pair
            
        Returns:
            Dictionary with copula results
        """
        results = {}
        
        if COPULAS_AVAILABLE:
            # Fit Clayton copula (lower tail dependence)
            clayton_results = self._fit_clayton(X, Y)
            results['clayton'] = clayton_results
            
            # Fit Gumbel copula (upper tail dependence)
            gumbel_results = self._fit_gumbel(X, Y)
            results['gumbel'] = gumbel_results
            
            # Fit Student-t copula (symmetric tail dependence)
            studentt_results = self._fit_studentt(X, Y)
            results['studentt'] = studentt_results
        else:
            # Fallback: empirical tail dependence
            results = self._empirical_tail_dependence(X, Y)
        
        return results
    
    def _fit_clayton(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """
        Fit Clayton copula (models lower tail dependence)
        
        Args:
            X, Y: Asset returns
            
        Returns:
            Dictionary with Clayton results
        """
        try:
            # Create DataFrame for copulas library
            data = pd.DataFrame({'X': X, 'Y': Y})
            
            # Fit copula
            copula = Clayton()
            copula.fit(data)
            
            # Extract tail dependence parameter
            theta = copula.theta
            lambda_l = 2 ** (-1 / theta)  # Lower tail dependence
            
            return {
                'theta': theta,
                'lambda_lower': lambda_l,
                'lambda_upper': 0  # Clayton has no upper tail dependence
            }
        except Exception as e:
            logger.warning(f"Clayton copula failed: {e}")
            return {'theta': None, 'lambda_lower': 0, 'lambda_upper': 0}
    
    def _fit_gumbel(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """
        Fit Gumbel copula (models upper tail dependence)
        
        Args:
            X, Y: Asset returns
            
        Returns:
            Dictionary with Gumbel results
        """
        try:
            data = pd.DataFrame({'X': X, 'Y': Y})
            
            copula = Gumbel()
            copula.fit(data)
            
            theta = copula.theta
            lambda_u = 2 - 2 ** (1 / theta)  # Upper tail dependence
            
            return {
                'theta': theta,
                'lambda_lower': 0,  # Gumbel has no lower tail dependence
                'lambda_upper': lambda_u
            }
        except Exception as e:
            logger.warning(f"Gumbel copula failed: {e}")
            return {'theta': None, 'lambda_lower': 0, 'lambda_upper': 0}
    
    def _fit_studentt(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """
        Fit Student-t copula (symmetric tail dependence)
        
        Args:
            X, Y: Asset returns
            
        Returns:
            Dictionary with Student-t results
        """
        try:
            data = pd.DataFrame({'X': X, 'Y': Y})
            
            copula = StudentT()
            copula.fit(data)
            
            # Student-t has symmetric tail dependence
            return {
                'correlation': copula.correlation if hasattr(copula, 'correlation') else None,
                'lambda_lower': None,  # Depends on correlation and df
                'lambda_upper': None
            }
        except Exception as e:
            logger.warning(f"Student-t copula failed: {e}")
            return {'correlation': None, 'lambda_lower': 0, 'lambda_upper': 0}
    
    def _empirical_tail_dependence(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """
        Calculate empirical tail dependence (fallback method)
        
        Args:
            X, Y: Asset returns
            
        Returns:
            Dictionary with empirical tail dependence
        """
        # Use quantiles to estimate tail dependence
        threshold = 0.05  # 5% tail
        
        # Lower tail (crashes)
        lower_threshold_x = np.quantile(X, threshold)
        lower_threshold_y = np.quantile(Y, threshold)
        
        in_lower_tail_x = X <= lower_threshold_x
        in_lower_tail_y = Y <= lower_threshold_y
        lambda_l = (in_lower_tail_x & in_lower_tail_y).sum() / in_lower_tail_x.sum()
        
        # Upper tail (surges)
        upper_threshold_x = np.quantile(X, 1 - threshold)
        upper_threshold_y = np.quantile(Y, 1 - threshold)
        
        in_upper_tail_x = X >= upper_threshold_x
        in_upper_tail_y = Y >= upper_threshold_y
        lambda_u = (in_upper_tail_x & in_upper_tail_y).sum() / in_upper_tail_x.sum()
        
        return {
            'empirical': True,
            'lambda_lower': lambda_l,
            'lambda_upper': lambda_u,
            'pearson_corr': np.corrcoef(X, Y)[0, 1]
        }
    
    def calculate_jump_ratios(self, df_with_jumps: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate jump ratio: jump_covariance / total_covariance
        
        Args:
            df_with_jumps: DataFrame with jump indicators
            
        Returns:
            DataFrame with jump ratios for all pairs
        """
        logger.info("Calculating jump ratios...")
        
        # Get returns and jumps in wide format
        returns_wide = df_with_jumps.pivot(
            index='date', columns='asset', values='returns'
        ).dropna()
        
        jump_returns_wide = df_with_jumps.pivot(
            index='date', columns='asset', values='jump_size'
        ).fillna(0)
        
        # Align indices
        common_dates = returns_wide.index.intersection(jump_returns_wide.index)
        returns_wide = returns_wide.loc[common_dates]
        jump_returns_wide = jump_returns_wide.loc[common_dates]
        
        assets = list(returns_wide.columns)
        
        # Calculate ratios for all pairs
        ratio_results = []
        
        for asset1, asset2 in combinations(assets, 2):
            # Total covariance
            total_cov = np.cov(returns_wide[asset1], returns_wide[asset2])[0, 1]
            
            # Jump covariance
            jump_cov = np.cov(jump_returns_wide[asset1], jump_returns_wide[asset2])[0, 1]
            
            # Jump ratio
            if abs(total_cov) > 1e-10:
                jump_ratio = abs(jump_cov / total_cov)
            else:
                jump_ratio = 0
            
            # Classify contagion risk
            if jump_ratio >= self.copula_config['jump_ratio_threshold']['critical']:
                risk_level = 'critical'
            elif jump_ratio >= self.copula_config['jump_ratio_threshold']['high']:
                risk_level = 'high'
            else:
                risk_level = 'low'
            
            ratio_results.append({
                'asset1': asset1,
                'asset2': asset2,
                'total_cov': total_cov,
                'jump_cov': jump_cov,
                'jump_ratio': jump_ratio,
                'risk_level': risk_level
            })
        
        jump_ratio_df = pd.DataFrame(ratio_results)
        jump_ratio_df = jump_ratio_df.sort_values('jump_ratio', ascending=False)
        
        # Log high-risk pairs
        high_risk = jump_ratio_df[jump_ratio_df['risk_level'].isin(['high', 'critical'])]
        logger.info(f"Found {len(high_risk)} high-risk contagion pairs")
        
        return jump_ratio_df
    
    def identify_contagion_clusters(self, jump_ratio_df: pd.DataFrame) -> List[List[str]]:
        """
        Identify clusters of highly connected assets (contagion risk)
        
        Args:
            jump_ratio_df: DataFrame with jump ratios
            
        Returns:
            List of asset clusters
        """
        logger.info("Identifying contagion clusters...")
        
        # Build adjacency matrix for high jump ratio pairs
        high_threshold = self.copula_config['jump_ratio_threshold']['high']
        high_risk_pairs = jump_ratio_df[jump_ratio_df['jump_ratio'] >= high_threshold]
        
        # Get all assets
        all_assets = set(high_risk_pairs['asset1']).union(set(high_risk_pairs['asset2']))
        
        # Build graph
        graph = {asset: set() for asset in all_assets}
        for _, row in high_risk_pairs.iterrows():
            graph[row['asset1']].add(row['asset2'])
            graph[row['asset2']].add(row['asset1'])
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for asset in all_assets:
            if asset not in visited:
                cluster = self._dfs_cluster(asset, graph, visited)
                if len(cluster) > 1:  # Only keep multi-asset clusters
                    clusters.append(sorted(cluster))
        
        logger.info(f"Identified {len(clusters)} contagion clusters")
        for i, cluster in enumerate(clusters):
            logger.info(f"  Cluster {i+1}: {cluster}")
        
        return clusters
    
    def _dfs_cluster(self, asset: str, graph: Dict, visited: set) -> List[str]:
        """
        Depth-first search to find connected component
        
        Args:
            asset: Starting asset
            graph: Adjacency graph
            visited: Set of visited assets
            
        Returns:
            List of assets in cluster
        """
        cluster = []
        stack = [asset]
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                cluster.append(current)
                stack.extend(graph[current] - visited)
        
        return cluster
    
    def summarize_tail_dependence(self, tail_results: Dict) -> pd.DataFrame:
        """
        Summarize tail dependence results
        
        Args:
            tail_results: Dictionary of copula results
            
        Returns:
            Summary DataFrame
        """
        summary = []
        
        for pair_name, copula_results in tail_results.items():
            if 'empirical' in copula_results:
                # Empirical results
                summary.append({
                    'pair': pair_name,
                    'lambda_lower': copula_results['lambda_lower'],
                    'lambda_upper': copula_results['lambda_upper'],
                    'method': 'empirical'
                })
            else:
                # Take maximum tail dependence across copula models
                lambda_lower = max(
                    copula_results.get('clayton', {}).get('lambda_lower', 0),
                    copula_results.get('studentt', {}).get('lambda_lower', 0) or 0
                )
                lambda_upper = max(
                    copula_results.get('gumbel', {}).get('lambda_upper', 0),
                    copula_results.get('studentt', {}).get('lambda_upper', 0) or 0
                )
                
                summary.append({
                    'pair': pair_name,
                    'lambda_lower': lambda_lower,
                    'lambda_upper': lambda_upper,
                    'method': 'copula'
                })
        
        summary_df = pd.DataFrame(summary)
        
        # Log key findings
        avg_lower = summary_df['lambda_lower'].mean()
        avg_upper = summary_df['lambda_upper'].mean()
        logger.info(f"\nTail Dependence Summary:")
        logger.info(f"  Average λ_lower (crashes): {avg_lower:.3f}")
        logger.info(f"  Average λ_upper (surges): {avg_upper:.3f}")
        
        if avg_upper > avg_lower:
            logger.info("  → Upper tail dependence stronger (synchronized surges)")
        else:
            logger.info("  → Lower tail dependence stronger (synchronized crashes)")
        
        return summary_df


def analyze_contagion(df_with_jumps: pd.DataFrame, config: Dict) -> Dict:
    """
    Convenience function for complete contagion analysis
    
    Args:
        df_with_jumps: DataFrame with jump indicators
        config: Configuration dictionary
        
    Returns:
        Dictionary with all contagion results
    """
    analyzer = CopulaAnalyzer(config)
    
    # Analyze tail dependence
    tail_results = analyzer.analyze_tail_dependence(df_with_jumps)
    tail_summary = analyzer.summarize_tail_dependence(tail_results)
    
    # Calculate jump ratios
    jump_ratio_df = analyzer.calculate_jump_ratios(df_with_jumps)
    
    # Identify clusters
    clusters = analyzer.identify_contagion_clusters(jump_ratio_df)
    
    return {
        'tail_results': tail_results,
        'tail_summary': tail_summary,
        'jump_ratios': jump_ratio_df,
        'clusters': clusters
    }


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from data_loader import load_and_prepare_data
    from jump_detector import detect_and_analyze_jumps
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data and detect jumps
    data_splits = load_and_prepare_data(config)
    train_df = data_splits['train']
    df_with_jumps, metrics, cojump_df = detect_and_analyze_jumps(train_df, config)
    
    # Analyze contagion
    contagion_results = analyze_contagion(df_with_jumps, config)
    
    print("\n=== Contagion Analysis Complete ===")
    print(f"High-risk pairs: {(contagion_results['jump_ratios']['risk_level'] != 'low').sum()}")
    print(f"Contagion clusters: {len(contagion_results['clusters'])}")
