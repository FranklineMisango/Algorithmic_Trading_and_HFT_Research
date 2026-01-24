"""
Portfolio Optimizer - Phase 3
Implements jump-adjusted minimum variance portfolio optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

# CVXPY for convex optimization
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("cvxpy not available. Install with: pip install cvxpy")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Optimizes portfolio using jump-adjusted covariance matrix
    """
    
    def __init__(self, config: Dict):
        """
        Initialize portfolio optimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.portfolio_config = config['portfolio_optimization']
        self.constraints_config = self.portfolio_config['constraints']
        
        if not CVXPY_AVAILABLE:
            logger.warning("CVXPY not available. Using simplified optimization.")
    
    def optimize_portfolio(
        self, 
        returns_df: pd.DataFrame, 
        jump_returns_df: pd.DataFrame
    ) -> Dict:
        """
        Optimize portfolio with jump-adjusted covariance
        
        Args:
            returns_df: DataFrame of asset returns (date × assets)
            jump_returns_df: DataFrame of jump returns (date × assets)
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        logger.info("Optimizing portfolio with jump-adjusted covariance...")
        
        # Calculate covariance matrices
        total_cov = self._calculate_total_covariance(returns_df, jump_returns_df)
        standard_cov = returns_df.cov().values
        
        # Optimize with jump-adjusted covariance
        if CVXPY_AVAILABLE:
            optimal_weights = self._cvxpy_optimize(total_cov, list(returns_df.columns))
        else:
            optimal_weights = self._simplified_optimize(total_cov, list(returns_df.columns))
        
        # Also optimize with standard covariance for comparison
        if CVXPY_AVAILABLE:
            standard_weights = self._cvxpy_optimize(standard_cov, list(returns_df.columns))
        else:
            standard_weights = self._simplified_optimize(standard_cov, list(returns_df.columns))
        
        # Calculate portfolio statistics
        optimal_stats = self._calculate_portfolio_stats(
            optimal_weights, returns_df, jump_returns_df
        )
        standard_stats = self._calculate_portfolio_stats(
            standard_weights, returns_df, jump_returns_df
        )
        
        return {
            'optimal_weights': optimal_weights,
            'standard_weights': standard_weights,
            'optimal_stats': optimal_stats,
            'standard_stats': standard_stats,
            'total_cov': total_cov,
            'standard_cov': standard_cov
        }
    
    def _calculate_total_covariance(
        self, 
        returns_df: pd.DataFrame, 
        jump_returns_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate jump-adjusted covariance: Σ_total = Σ_returns + Σ_jumps
        
        Args:
            returns_df: Total returns
            jump_returns_df: Jump component of returns
            
        Returns:
            Jump-adjusted covariance matrix
        """
        # Covariance of total returns
        cov_returns = returns_df.cov().values
        
        # Covariance of jumps
        cov_jumps = jump_returns_df.cov().values
        
        # Total covariance
        total_cov = cov_returns + cov_jumps
        
        # Ensure positive semi-definite
        total_cov = self._ensure_psd(total_cov)
        
        logger.info(f"  Standard vol range: [{np.sqrt(np.diag(cov_returns)).min():.4f}, "
                   f"{np.sqrt(np.diag(cov_returns)).max():.4f}]")
        logger.info(f"  Jump-adjusted vol range: [{np.sqrt(np.diag(total_cov)).min():.4f}, "
                   f"{np.sqrt(np.diag(total_cov)).max():.4f}]")
        
        return total_cov
    
    def _ensure_psd(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Ensure covariance matrix is positive semi-definite
        
        Args:
            cov_matrix: Input covariance matrix
            
        Returns:
            Adjusted PSD matrix
        """
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Replace negative eigenvalues with small positive value
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        
        # Reconstruct matrix
        cov_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return cov_psd
    
    def _cvxpy_optimize(self, cov_matrix: np.ndarray, assets: list) -> Dict[str, float]:
        """
        Optimize using CVXPY (convex optimization)
        
        Args:
            cov_matrix: Covariance matrix
            assets: List of asset names
            
        Returns:
            Dictionary of optimal weights
        """
        n_assets = len(assets)
        
        # Decision variables
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= 0  # Long-only
        ]
        
        # Max position size constraint
        max_weight = self.constraints_config['max_single_asset_weight']
        if max_weight < 1.0:
            constraints.append(w <= max_weight)
        
        # Min number of assets constraint
        min_assets = self.constraints_config['min_assets']
        if min_assets > 1:
            # Binary variables for asset selection
            z = cp.Variable(n_assets, boolean=True)
            min_weight = 0.01  # Minimum weight if asset is included
            
            constraints.extend([
                w <= z,  # If weight > 0, then z = 1
                w >= min_weight * z,  # If z = 1, weight >= min_weight
                cp.sum(z) >= min_assets  # At least min_assets selected
            ])
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status != cp.OPTIMAL:
            logger.warning(f"Optimization status: {problem.status}")
        
        # Extract weights
        weights = {asset: max(0, w.value[i]) for i, asset in enumerate(assets)}
        
        # Normalize to sum to 1 (handle numerical errors)
        total = sum(weights.values())
        if total > 0:
            weights = {asset: weight / total for asset, weight in weights.items()}
        
        return weights
    
    def _simplified_optimize(self, cov_matrix: np.ndarray, assets: list) -> Dict[str, float]:
        """
        Simplified optimization (fallback without CVXPY)
        Uses inverse variance weighting with constraints
        
        Args:
            cov_matrix: Covariance matrix
            assets: List of asset names
            
        Returns:
            Dictionary of weights
        """
        # Inverse variance weights
        variances = np.diag(cov_matrix)
        inv_var_weights = 1 / variances
        
        # Normalize
        inv_var_weights = inv_var_weights / inv_var_weights.sum()
        
        # Apply max weight constraint
        max_weight = self.constraints_config['max_single_asset_weight']
        inv_var_weights = np.minimum(inv_var_weights, max_weight)
        
        # Re-normalize
        inv_var_weights = inv_var_weights / inv_var_weights.sum()
        
        weights = {asset: inv_var_weights[i] for i, asset in enumerate(assets)}
        
        return weights
    
    def _calculate_portfolio_stats(
        self, 
        weights: Dict[str, float], 
        returns_df: pd.DataFrame,
        jump_returns_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate portfolio statistics
        
        Args:
            weights: Portfolio weights
            returns_df: Asset returns
            jump_returns_df: Jump returns
            
        Returns:
            Dictionary of statistics
        """
        # Convert weights to array (aligned with returns_df columns)
        weight_array = np.array([weights.get(asset, 0) for asset in returns_df.columns])
        
        # Portfolio returns
        portfolio_returns = (returns_df.values @ weight_array)
        
        # Jump exposure
        jump_returns = (jump_returns_df.values @ weight_array)
        
        # Calculate metrics
        mean_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        sharpe = mean_return / volatility if volatility > 0 else 0
        
        # Jump metrics
        jump_exposure = (jump_returns != 0).sum() / len(jump_returns)
        avg_jump_size = np.abs(jump_returns[jump_returns != 0]).mean() if (jump_returns != 0).any() else 0
        
        # Concentration
        weight_values = list(weights.values())
        n_assets = sum(1 for w in weight_values if w > 0.01)
        herfindahl = sum(w**2 for w in weight_values)
        
        stats = {
            'mean_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'jump_exposure': jump_exposure,
            'avg_jump_size': avg_jump_size,
            'n_assets': n_assets,
            'herfindahl': herfindahl,
            'max_weight': max(weight_values),
            'min_weight': min(w for w in weight_values if w > 0) if any(w > 0 for w in weight_values) else 0
        }
        
        return stats
    
    def compare_strategies(self, optimization_result: Dict) -> pd.DataFrame:
        """
        Compare jump-adjusted vs standard portfolio
        
        Args:
            optimization_result: Result from optimize_portfolio
            
        Returns:
            Comparison DataFrame
        """
        optimal_stats = optimization_result['optimal_stats']
        standard_stats = optimization_result['standard_stats']
        
        comparison = pd.DataFrame({
            'Jump-Adjusted': optimal_stats,
            'Standard': standard_stats
        }).T
        
        # Add improvement metrics
        comparison['Sharpe_Improvement'] = (
            optimal_stats['sharpe_ratio'] - standard_stats['sharpe_ratio']
        )
        
        logger.info("\n=== Portfolio Comparison ===")
        logger.info(f"Jump-Adjusted Sharpe: {optimal_stats['sharpe_ratio']:.4f}")
        logger.info(f"Standard Sharpe: {standard_stats['sharpe_ratio']:.4f}")
        logger.info(f"Improvement: {comparison.loc['Jump-Adjusted', 'Sharpe_Improvement']:.4f}")
        
        return comparison
    
    def construct_benchmark_portfolios(self, returns_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Construct benchmark portfolios for comparison
        
        Args:
            returns_df: Asset returns
            
        Returns:
            Dictionary of benchmark weights and stats
        """
        assets = list(returns_df.columns)
        n_assets = len(assets)
        
        benchmarks = {}
        
        # Equal weight
        equal_weights = {asset: 1.0 / n_assets for asset in assets}
        benchmarks['equal_weight'] = equal_weights
        
        # BTC/ETH 60/40
        if 'BTC' in assets and 'ETH' in assets:
            btc_eth_weights = {asset: 0 for asset in assets}
            btc_eth_weights['BTC'] = 0.6
            btc_eth_weights['ETH'] = 0.4
            benchmarks['btc_eth_6040'] = btc_eth_weights
        
        logger.info(f"Constructed {len(benchmarks)} benchmark portfolios")
        
        return benchmarks


def optimize_and_compare(
    returns_df: pd.DataFrame, 
    jump_returns_df: pd.DataFrame, 
    config: Dict
) -> Dict:
    """
    Convenience function for portfolio optimization and comparison
    
    Args:
        returns_df: Asset returns
        jump_returns_df: Jump returns
        config: Configuration dictionary
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = PortfolioOptimizer(config)
    
    # Optimize portfolio
    result = optimizer.optimize_portfolio(returns_df, jump_returns_df)
    
    # Compare strategies
    comparison = optimizer.compare_strategies(result)
    
    # Construct benchmarks
    benchmarks = optimizer.construct_benchmark_portfolios(returns_df)
    
    result['comparison'] = comparison
    result['benchmarks'] = benchmarks
    
    return result


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
    
    # Prepare data for optimization
    returns_df = df_with_jumps.pivot(
        index='date', columns='asset', values='returns'
    ).dropna()
    
    jump_returns_df = df_with_jumps.pivot(
        index='date', columns='asset', values='jump_size'
    ).fillna(0)
    
    # Ensure same dates
    common_dates = returns_df.index.intersection(jump_returns_df.index)
    returns_df = returns_df.loc[common_dates]
    jump_returns_df = jump_returns_df.loc[common_dates]
    
    # Optimize
    opt_result = optimize_and_compare(returns_df, jump_returns_df, config)
    
    print("\n=== Portfolio Optimization Complete ===")
    print("\nOptimal Weights (Jump-Adjusted):")
    for asset, weight in sorted(opt_result['optimal_weights'].items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:
            print(f"  {asset}: {weight*100:.1f}%")
