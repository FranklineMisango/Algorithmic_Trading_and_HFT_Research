"""
Volatility Clustering Module

Implements K-means++ clustering to partition stocks into:
- Low volatility cluster
- Mid volatility cluster (target for strategy)
- High volatility cluster

Based on mean historical volatility over the analysis period.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class VolatilityClustering:
    """
    Cluster stocks based on their historical volatility characteristics.
    """
    
    def __init__(
        self, 
        n_clusters: int = 3,
        random_state: int = 42
    ):
        """
        Initialize the clustering model.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (default: 3 for low/mid/high)
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.cluster_centers = None
        
    def fit(self, volatility_df: pd.DataFrame) -> 'VolatilityClustering':
        """
        Fit K-means++ clustering on mean volatility values.
        
        Parameters:
        -----------
        volatility_df : pd.DataFrame
            DataFrame with tickers as columns and volatility time series
            
        Returns:
        --------
        self : VolatilityClustering
            Fitted clustering model
        """
        # Calculate mean volatility for each asset
        mean_volatility = volatility_df.mean()
        
        # Reshape for sklearn (needs 2D array)
        X = mean_volatility.values.reshape(-1, 1)
        
        # Standardize the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply K-means++ clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            random_state=self.random_state,
            n_init=10
        )
        
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Store results
        self.cluster_labels = pd.Series(
            labels, 
            index=mean_volatility.index,
            name='cluster'
        )
        
        # Get cluster centers in original scale
        self.cluster_centers = self.scaler.inverse_transform(
            self.kmeans.cluster_centers_
        ).flatten()
        
        return self
    
    def get_cluster_assignments(self) -> pd.Series:
        """
        Get cluster assignment for each ticker.
        
        Returns:
        --------
        pd.Series : Cluster labels for each ticker
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.cluster_labels
    
    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """
        Get list of tickers in a specific cluster.
        
        Parameters:
        -----------
        cluster_id : int
            Cluster ID (0, 1, 2 for low/mid/high)
            
        Returns:
        --------
        List[str] : List of tickers in the cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.cluster_labels[self.cluster_labels == cluster_id].index.tolist()
    
    def identify_mid_cluster(self) -> int:
        """
        Identify which cluster is the "mid-volatility" cluster.
        Assumes 3 clusters: low (0), mid (1), high (2).
        
        Returns:
        --------
        int : Cluster ID of mid-volatility cluster
        """
        if self.cluster_centers is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Sort cluster centers by volatility
        sorted_indices = np.argsort(self.cluster_centers)
        
        # Mid cluster is the middle one
        mid_cluster_id = sorted_indices[len(sorted_indices) // 2]
        
        return mid_cluster_id
    
    def get_cluster_statistics(
        self, 
        volatility_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get statistics for each cluster.
        
        Parameters:
        -----------
        volatility_df : pd.DataFrame
            DataFrame with volatility time series
            
        Returns:
        --------
        pd.DataFrame : Statistics for each cluster
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        stats = []
        mean_volatility = volatility_df.mean()
        
        for cluster_id in range(self.n_clusters):
            members = self.get_cluster_members(cluster_id)
            cluster_vols = mean_volatility[members]
            
            stats.append({
                'cluster_id': cluster_id,
                'n_members': len(members),
                'mean_volatility': cluster_vols.mean(),
                'median_volatility': cluster_vols.median(),
                'std_volatility': cluster_vols.std(),
                'min_volatility': cluster_vols.min(),
                'max_volatility': cluster_vols.max(),
                'members': ', '.join(members)
            })
        
        stats_df = pd.DataFrame(stats)
        
        # Sort by mean volatility
        stats_df = stats_df.sort_values('mean_volatility').reset_index(drop=True)
        stats_df['cluster_type'] = ['Low', 'Mid', 'High'][:len(stats_df)]
        
        return stats_df
    
    def plot_clusters(
        self, 
        volatility_df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Visualize the clustering results.
        
        Parameters:
        -----------
        volatility_df : pd.DataFrame
            DataFrame with volatility time series
        save_path : str, optional
            Path to save the plot
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        mean_volatility = volatility_df.mean()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Scatter plot of mean volatility with cluster colors
        colors = ['blue', 'green', 'red']
        for cluster_id in range(self.n_clusters):
            members = self.get_cluster_members(cluster_id)
            cluster_vols = mean_volatility[members]
            
            axes[0].scatter(
                range(len(cluster_vols)),
                cluster_vols,
                c=colors[cluster_id],
                label=f'Cluster {cluster_id}',
                s=100,
                alpha=0.6
            )
            
            # Add ticker labels
            for i, ticker in enumerate(members):
                axes[0].annotate(
                    ticker,
                    (i, cluster_vols[ticker]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
        
        axes[0].set_xlabel('Asset Index')
        axes[0].set_ylabel('Mean Volatility')
        axes[0].set_title('K-means++ Clustering of Assets by Mean Volatility')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot of volatility by cluster
        cluster_data = []
        cluster_labels_list = []
        
        for cluster_id in range(self.n_clusters):
            members = self.get_cluster_members(cluster_id)
            cluster_vols = mean_volatility[members].values
            cluster_data.extend(cluster_vols)
            cluster_labels_list.extend([f'Cluster {cluster_id}'] * len(cluster_vols))
        
        box_df = pd.DataFrame({
            'Volatility': cluster_data,
            'Cluster': cluster_labels_list
        })
        
        sns.boxplot(data=box_df, x='Cluster', y='Volatility', ax=axes[1])
        axes[1].set_title('Volatility Distribution by Cluster')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_time_series_by_cluster(
        self,
        volatility_df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Plot volatility time series grouped by cluster.
        
        Parameters:
        -----------
        volatility_df : pd.DataFrame
            DataFrame with volatility time series
        save_path : str, optional
            Path to save the plot
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        fig, axes = plt.subplots(self.n_clusters, 1, figsize=(14, 4 * self.n_clusters))
        
        if self.n_clusters == 1:
            axes = [axes]
        
        cluster_names = ['Low Volatility', 'Mid Volatility', 'High Volatility']
        
        for cluster_id in range(self.n_clusters):
            members = self.get_cluster_members(cluster_id)
            
            for ticker in members:
                axes[cluster_id].plot(
                    volatility_df.index,
                    volatility_df[ticker],
                    label=ticker,
                    alpha=0.7
                )
            
            axes[cluster_id].set_title(
                f'{cluster_names[cluster_id]} Cluster (n={len(members)})'
            )
            axes[cluster_id].set_xlabel('Date')
            axes[cluster_id].set_ylabel('Volatility')
            axes[cluster_id].legend(loc='best', ncol=3)
            axes[cluster_id].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def cluster_assets_by_volatility(
    volatility_df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
    target_cluster: str = 'mid'
) -> Tuple[VolatilityClustering, List[str]]:
    """
    Convenience function to cluster assets and return mid-volatility members.
    
    Parameters:
    -----------
    volatility_df : pd.DataFrame
        DataFrame with volatility time series
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed
    target_cluster : str
        Which cluster to return ('low', 'mid', 'high')
        
    Returns:
    --------
    Tuple[VolatilityClustering, List[str]] : 
        Fitted clustering model and list of target cluster members
    """
    # Fit clustering
    clustering = VolatilityClustering(n_clusters=n_clusters, random_state=random_state)
    clustering.fit(volatility_df)
    
    # Get statistics
    stats = clustering.get_cluster_statistics(volatility_df)
    print("\nCluster Statistics:")
    print(stats.to_string(index=False))
    
    # Identify target cluster
    if target_cluster == 'mid':
        target_id = clustering.identify_mid_cluster()
    elif target_cluster == 'low':
        target_id = stats.iloc[0]['cluster_id']
    elif target_cluster == 'high':
        target_id = stats.iloc[-1]['cluster_id']
    else:
        raise ValueError(f"Unknown target_cluster: {target_cluster}")
    
    target_members = clustering.get_cluster_members(int(target_id))
    
    print(f"\n{target_cluster.capitalize()}-volatility cluster members: {target_members}")
    
    return clustering, target_members


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    from volatility_estimators import calculate_volatility_for_assets
    
    # Download data for tech stocks
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
    
    # Visualize
    clustering.plot_clusters(volatility_df)
