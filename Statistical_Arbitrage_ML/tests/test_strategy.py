"""
Unit tests for Statistical Arbitrage Strategy modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('..')

from src.data_acquisition import DataAcquisitionEngine
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.portfolio_builder import PortfolioBuilder
from src.backtester import Backtester
from src.risk_manager import RiskManager


class TestDataAcquisition:
    """Test data acquisition module."""
    
    def test_initialization(self):
        engine = DataAcquisitionEngine()
        assert engine.data_dir.exists()
        assert engine.cache_enabled == True
    
    def test_universe_retrieval(self):
        engine = DataAcquisitionEngine()
        universe = engine.get_russell_3000_universe()
        assert len(universe) > 0
        assert all(isinstance(ticker, str) for ticker in universe)


class TestFeatureEngineering:
    """Test feature engineering module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        
        data = []
        for ticker in tickers:
            base_price = np.random.uniform(100, 200)
            for date in dates:
                data.append({
                    'Date': date,
                    'ticker': ticker,
                    'Open': base_price * (1 + np.random.randn() * 0.01),
                    'High': base_price * (1 + abs(np.random.randn() * 0.02)),
                    'Low': base_price * (1 - abs(np.random.randn() * 0.02)),
                    'Close': base_price * (1 + np.random.randn() * 0.01),
                    'Volume': np.random.randint(1_000_000, 10_000_000)
                })
        
        df = pd.DataFrame(data)
        df = df.set_index(['Date', 'ticker'])
        return df
    
    def test_feature_calculation(self, sample_data):
        engineer = FeatureEngineer()
        df_features = engineer.calculate_all_features(sample_data)
        
        assert 'momentum_5d' in df_features.columns
        assert 'distance_from_ma_20d' in df_features.columns
        assert 'relative_volume' in df_features.columns
        assert 'forward_return_3d' in df_features.columns
    
    def test_ml_dataset_preparation(self, sample_data):
        engineer = FeatureEngineer()
        df_features = engineer.calculate_all_features(sample_data)
        X, y = engineer.prepare_ml_dataset(df_features)
        
        assert len(X) == len(y)
        assert X.shape[1] > 0  # Has features
        assert not X.isnull().any().any()  # No NaN


class TestModelTrainer:
    """Test model training module."""
    
    @pytest.fixture
    def sample_ml_data(self):
        """Create sample ML data."""
        n_samples = 1000
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples) * 0.01)
        
        return X, y
    
    def test_model_initialization(self):
        trainer = ModelTrainer(model_type='xgboost')
        assert trainer.model_type == 'xgboost'
        assert trainer.rolling_window_years == 10
    
    def test_model_training(self, sample_ml_data):
        X, y = sample_ml_data
        trainer = ModelTrainer(model_type='ridge')
        
        metrics = trainer.train(X, y)
        
        assert 'train_mse' in metrics
        assert 'train_r2' in metrics
        assert trainer.model is not None
    
    def test_predictions(self, sample_ml_data):
        X, y = sample_ml_data
        trainer = ModelTrainer(model_type='ridge')
        trainer.train(X, y)
        
        predictions = trainer.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)


class TestPortfolioBuilder:
    """Test portfolio construction module."""
    
    def test_initialization(self):
        builder = PortfolioBuilder(n_long=20, n_short=20)
        assert builder.n_long == 20
        assert builder.n_short == 20
    
    def test_stock_ranking(self):
        builder = PortfolioBuilder()
        
        predictions = pd.Series(
            np.random.randn(100) * 0.02,
            index=[f'STOCK{i}' for i in range(100)]
        )
        
        ranking = builder.rank_stocks(predictions, pd.Timestamp('2024-01-15'))
        
        assert len(ranking) == 100
        assert 'ticker' in ranking.columns
        assert 'predicted_return' in ranking.columns
        assert 'rank' in ranking.columns
    
    def test_portfolio_construction(self):
        builder = PortfolioBuilder(n_long=10, n_short=10)
        
        tickers = [f'STOCK{i}' for i in range(50)]
        predictions = pd.Series(
            np.random.randn(50) * 0.02,
            index=tickers
        )
        prices = pd.Series(
            np.random.uniform(50, 200, 50),
            index=tickers
        )
        
        portfolio = builder.build_portfolio(
            predictions, prices, pd.Timestamp('2024-01-15')
        )
        
        assert len(portfolio) == 20  # 10 long + 10 short
        assert 'ticker' in portfolio.columns
        assert 'side' in portfolio.columns
        assert 'shares' in portfolio.columns


class TestBacktester:
    """Test backtesting module."""
    
    def test_initialization(self):
        backtester = Backtester(initial_capital=1_000_000)
        assert backtester.initial_capital == 1_000_000
        assert backtester.transaction_cost_bps == 10.0
    
    def test_performance_metrics(self):
        backtester = Backtester()
        
        # Create sample equity curve
        returns = pd.Series(np.random.randn(252) * 0.01)
        equity = pd.DataFrame({
            'equity': (1 + returns).cumprod() * 1_000_000,
            'returns': returns
        })
        
        trades = pd.DataFrame({
            'net_pnl': np.random.randn(100) * 1000
        })
        
        metrics = backtester.calculate_performance_metrics(equity, trades)
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics


class TestRiskManager:
    """Test risk management module."""
    
    def test_initialization(self):
        risk_mgr = RiskManager()
        assert risk_mgr.max_position_size == 0.04
    
    def test_position_limits(self):
        risk_mgr = RiskManager(max_position_size=0.03)
        
        portfolio = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'side': ['long', 'short'],
            'capital': [50000, 40000]
        })
        
        warnings = risk_mgr.check_position_limits(portfolio, 1_000_000)
        
        # Both positions exceed 3% limit
        assert len(warnings) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
