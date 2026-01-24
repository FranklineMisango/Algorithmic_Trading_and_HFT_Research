"""
Portfolio Construction for Music Royalties Strategy
Ranks assets by mispricing and constructs diversified portfolio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """
    Constructs portfolio of undervalued music royalty assets
    """
    
    def __init__(self, config: Dict):
        """
        Initialize portfolio constructor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.selection_method = config['portfolio']['selection']['method']
        self.weighting_scheme = config['portfolio']['weighting_scheme']
        self.max_single_asset = config['portfolio']['selection']['diversification']['max_single_asset']
        self.max_genre_concentration = config['portfolio']['selection']['diversification']['max_genre_concentration']
        self.min_portfolio_size = config['portfolio']['selection']['diversification']['min_portfolio_size']
        
    def rank_assets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank assets by model mispricing (predicted fair value - market price)
        
        Positive mispricing = undervalued (buy signal)
        
        Args:
            df: DataFrame with mispricing column
            
        Returns:
            DataFrame sorted by mispricing (descending)
        """
        if 'mispricing' not in df.columns:
            raise ValueError("DataFrame must contain 'mispricing' column")
        
        # Sort by mispricing (highest = most undervalued)
        df_ranked = df.sort_values('mispricing', ascending=False).reset_index(drop=True)
        df_ranked['rank'] = df_ranked.index + 1
        
        logger.info(f"Ranked {len(df_ranked)} assets by mispricing")
        logger.info(f"Top asset mispricing: {df_ranked.iloc[0]['mispricing']:.2f}")
        
        return df_ranked
    
    def select_top_quintile(self, df_ranked: pd.DataFrame) -> pd.DataFrame:
        """
        Select top quintile (20%) most undervalued assets
        
        Args:
            df_ranked: Ranked DataFrame
            
        Returns:
            DataFrame with selected assets
        """
        n_select = max(int(len(df_ranked) * 0.2), self.min_portfolio_size)
        
        # Ensure minimum portfolio size
        n_select = min(n_select, len(df_ranked))
        
        selected = df_ranked.head(n_select).copy()
        
        logger.info(f"Selected top {n_select} assets (top {n_select/len(df_ranked)*100:.1f}%)")
        
        return selected
    
    def apply_diversification_constraints(self, selected: pd.DataFrame) -> pd.DataFrame:
        """
        Apply diversification constraints (genre limits, position sizes)
        
        Args:
            selected: Selected assets
            
        Returns:
            Filtered and constrained portfolio
        """
        selected = selected.copy()
        
        # Genre diversification (if genre data available)
        if 'genre' in selected.columns:
            # Calculate genre exposures
            genre_counts = selected['genre'].value_counts()
            max_per_genre = int(len(selected) * self.max_genre_concentration)
            
            # Filter if any genre exceeds limit
            filtered_assets = []
            genre_tracker = {}
            
            for idx, row in selected.iterrows():
                genre = row['genre']
                current_count = genre_tracker.get(genre, 0)
                
                if current_count < max_per_genre:
                    filtered_assets.append(idx)
                    genre_tracker[genre] = current_count + 1
                else:
                    logger.debug(f"Skipping asset {row['asset_id']} - genre {genre} at limit")
            
            selected = selected.loc[filtered_assets]
            logger.info(f"After genre constraints: {len(selected)} assets")
        
        return selected
    
    def calculate_weights(self, selected: pd.DataFrame, 
                         weighting_scheme: str = None) -> pd.DataFrame:
        """
        Calculate portfolio weights for selected assets
        
        Args:
            selected: Selected assets
            weighting_scheme: 'equal_weight', 'mispricing_weighted', or 'risk_parity'
            
        Returns:
            DataFrame with weights column
        """
        if weighting_scheme is None:
            weighting_scheme = self.weighting_scheme
        
        selected = selected.copy()
        
        if weighting_scheme == 'equal_weight':
            selected['weight'] = 1.0 / len(selected)
            
        elif weighting_scheme == 'mispricing_weighted':
            # Weight by mispricing magnitude (more undervalued = higher weight)
            mispricing_pos = np.maximum(selected['mispricing'], 0)
            total_mispricing = mispricing_pos.sum()
            if total_mispricing > 0:
                selected['weight'] = mispricing_pos / total_mispricing
            else:
                selected['weight'] = 1.0 / len(selected)
            
        elif weighting_scheme == 'risk_parity':
            # Weight by inverse volatility (if available)
            # Fallback to equal weight if no volatility data
            if 'volatility' in selected.columns:
                inv_vol = 1.0 / selected['volatility']
                selected['weight'] = inv_vol / inv_vol.sum()
            else:
                logger.warning("No volatility data, using equal weights")
                selected['weight'] = 1.0 / len(selected)
        
        else:
            raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")
        
        # Cap individual weights
        selected['weight'] = selected['weight'].clip(upper=self.max_single_asset)
        
        # Renormalize
        selected['weight'] = selected['weight'] / selected['weight'].sum()
        
        logger.info(f"Calculated {weighting_scheme} weights: max={selected['weight'].max():.2%}")
        
        return selected
    
    def construct_portfolio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full portfolio construction pipeline
        
        Args:
            df: DataFrame with features and mispricing
            
        Returns:
            Portfolio DataFrame with selected assets and weights
        """
        logger.info("=== Constructing Portfolio ===")
        
        # Filter to positive mispricing only (undervalued)
        df_undervalued = df[df['mispricing'] > 0].copy()
        logger.info(f"Found {len(df_undervalued)} undervalued assets")
        
        if len(df_undervalued) == 0:
            logger.warning("No undervalued assets found!")
            return pd.DataFrame()
        
        # Rank assets
        df_ranked = self.rank_assets(df_undervalued)
        
        # Select top quintile
        if self.selection_method == 'top_quintile':
            selected = self.select_top_quintile(df_ranked)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
        # Apply diversification constraints
        selected = self.apply_diversification_constraints(selected)
        
        # Calculate weights
        portfolio = self.calculate_weights(selected)
        
        # Validate portfolio
        self._validate_portfolio(portfolio)
        
        logger.info(f"=== Portfolio Construction Complete: {len(portfolio)} assets ===")
        
        return portfolio
    
    def _validate_portfolio(self, portfolio: pd.DataFrame) -> None:
        """
        Validate portfolio constraints
        
        Args:
            portfolio: Portfolio DataFrame
        """
        # Check weights sum to 1
        weight_sum = portfolio['weight'].sum()
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            logger.warning(f"Weights sum to {weight_sum:.6f}, not 1.0")
        
        # Check max single position
        max_weight = portfolio['weight'].max()
        if max_weight > self.max_single_asset * 1.01:  # Small tolerance
            logger.warning(f"Max weight {max_weight:.2%} exceeds limit {self.max_single_asset:.2%}")
        
        # Check min portfolio size
        if len(portfolio) < self.min_portfolio_size:
            logger.warning(f"Portfolio size {len(portfolio)} below minimum {self.min_portfolio_size}")
        
        # Check genre concentration
        if 'genre' in portfolio.columns:
            for genre, group in portfolio.groupby('genre'):
                genre_weight = group['weight'].sum()
                if genre_weight > self.max_genre_concentration * 1.01:
                    logger.warning(f"Genre {genre} weight {genre_weight:.2%} exceeds limit")
    
    def calculate_portfolio_characteristics(self, portfolio: pd.DataFrame) -> Dict:
        """
        Calculate portfolio-level characteristics
        
        Args:
            portfolio: Portfolio DataFrame
            
        Returns:
            Dictionary of portfolio characteristics
        """
        if len(portfolio) == 0:
            return {}
        
        chars = {
            'n_assets': len(portfolio),
            'avg_mispricing': (portfolio['mispricing'] * portfolio['weight']).sum(),
            'avg_stability_ratio': (portfolio['stability_ratio'] * portfolio['weight']).sum() if 'stability_ratio' in portfolio.columns else None,
            'avg_catalog_age': (portfolio['catalog_age'] * portfolio['weight']).sum() if 'catalog_age' in portfolio.columns else None,
            'avg_predicted_multiplier': (portfolio['predicted_multiplier'] * portfolio['weight']).sum() if 'predicted_multiplier' in portfolio.columns else None,
            'total_revenue_ltm': portfolio['revenue_ltm'].sum() if 'revenue_ltm' in portfolio.columns else None,
            'max_weight': portfolio['weight'].max(),
            'min_weight': portfolio['weight'].min(),
            'weight_concentration': (portfolio['weight'] ** 2).sum()  # HHI
        }
        
        # Genre distribution
        if 'genre' in portfolio.columns:
            genre_weights = portfolio.groupby('genre')['weight'].sum().to_dict()
            chars['genre_distribution'] = genre_weights
        
        return chars
    
    def rebalance_portfolio(self, current_portfolio: pd.DataFrame,
                          new_universe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Rebalance portfolio with new universe of assets
        
        Args:
            current_portfolio: Current holdings
            new_universe: New universe to select from
            
        Returns:
            (new_portfolio, trades) - new portfolio and required trades
        """
        # Construct new portfolio
        new_portfolio = self.construct_portfolio(new_universe)
        
        # Calculate trades needed
        trades = self._calculate_trades(current_portfolio, new_portfolio)
        
        logger.info(f"Rebalancing: {len(trades)} trades required")
        
        return new_portfolio, trades
    
    def _calculate_trades(self, current: pd.DataFrame, 
                         new: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trades needed to move from current to new portfolio
        
        Args:
            current: Current portfolio
            new: New target portfolio
            
        Returns:
            DataFrame with trades (sells and buys)
        """
        # Assets in current portfolio
        current_assets = set(current['asset_id']) if len(current) > 0 else set()
        new_assets = set(new['asset_id']) if len(new) > 0 else set()
        
        # Sells: assets in current but not in new
        sells = current[~current['asset_id'].isin(new_assets)].copy()
        sells['action'] = 'sell'
        sells['trade_weight'] = -sells['weight']
        
        # Buys: assets in new but not in current
        buys = new[~new['asset_id'].isin(current_assets)].copy()
        buys['action'] = 'buy'
        buys['trade_weight'] = buys['weight']
        
        # Adjustments: assets in both (weight changes)
        adjusts = []
        for asset_id in current_assets & new_assets:
            old_weight = current[current['asset_id'] == asset_id]['weight'].values[0]
            new_weight = new[new['asset_id'] == asset_id]['weight'].values[0]
            weight_delta = new_weight - old_weight
            
            if abs(weight_delta) > 0.001:  # Threshold for rebalancing
                row = new[new['asset_id'] == asset_id].copy()
                row['action'] = 'adjust'
                row['trade_weight'] = weight_delta
                adjusts.append(row)
        
        # Combine all trades
        all_trades = pd.concat([sells, buys] + adjusts, ignore_index=True) if (len(sells) > 0 or len(buys) > 0 or len(adjusts) > 0) else pd.DataFrame()
        
        return all_trades


def construct_portfolio_from_model(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Convenience function to construct portfolio from data with model predictions
    
    Args:
        df: DataFrame with features and mispricing
        config: Configuration dictionary
        
    Returns:
        Portfolio DataFrame
    """
    constructor = PortfolioConstructor(config)
    portfolio = constructor.construct_portfolio(df)
    
    # Calculate characteristics
    if len(portfolio) > 0:
        chars = constructor.calculate_portfolio_characteristics(portfolio)
        logger.info("\n=== Portfolio Characteristics ===")
        for key, value in chars.items():
            if key != 'genre_distribution':
                logger.info(f"{key}: {value}")
    
    return portfolio


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from data_loader import load_and_prepare_data
    from feature_engineering import engineer_all_features
    from model_trainer import train_and_validate_model
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and prepare data
    data_splits = load_and_prepare_data(config)
    
    # Engineer features
    train_df = engineer_all_features(data_splits['train'], config)
    test_df = engineer_all_features(data_splits['test'], config)
    
    # Train model
    val_df = engineer_all_features(data_splits['validation'], config)
    model, _ = train_and_validate_model(train_df, val_df, config)
    
    # Calculate mispricing on test set
    test_df = model.calculate_mispricing(test_df)
    
    # Construct portfolio
    portfolio = construct_portfolio_from_model(test_df, config)
    
    print("\n=== Portfolio Construction Complete ===")
    print(f"Selected {len(portfolio)} assets")
    print(f"\nTop 5 holdings:")
    print(portfolio[['asset_id', 'mispricing', 'weight']].head())
