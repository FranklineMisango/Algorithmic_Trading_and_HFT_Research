"""
Unit tests for signal generation
"""
import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from signal_generator import CurrencyCrashPredictor


@pytest.fixture
def sample_data():
    """Create sample FX and rate data"""
    dates = pd.date_range('2010-01-01', '2020-12-31', freq='M')
    
    fx_data = pd.DataFrame({
        'EUR': np.random.uniform(1.0, 1.5, len(dates)),
        'GBP': np.random.uniform(1.2, 1.7, len(dates)),
    }, index=dates)
    
    rate_data = pd.DataFrame({
        'EUR': np.random.uniform(0, 3, len(dates)),
        'GBP': np.random.uniform(0, 2, len(dates)),
    }, index=dates)
    
    return fx_data, rate_data


def test_predictor_initialization():
    """Test predictor initializes correctly"""
    predictor = CurrencyCrashPredictor()
    assert predictor.lookback == 6
    assert predictor.rate_threshold == 80
    assert predictor.fx_threshold == 33


def test_calculate_features(sample_data):
    """Test feature calculation"""
    fx_data, rate_data = sample_data
    
    predictor = CurrencyCrashPredictor()
    predictor.fx_rates = fx_data
    predictor.interest_rates = rate_data
    
    delta_i, delta_fx, monthly_returns = predictor.calculate_features()
    
    assert delta_i.shape == rate_data.shape
    assert delta_fx.shape == fx_data.shape
    assert monthly_returns.shape == fx_data.shape


def test_threshold_calculation(sample_data):
    """Test threshold calculation"""
    fx_data, rate_data = sample_data
    
    predictor = CurrencyCrashPredictor()
    predictor.fx_rates = fx_data
    predictor.interest_rates = rate_data
    
    delta_i, delta_fx, monthly_returns = predictor.calculate_features()
    thresholds = predictor.calculate_thresholds(
        delta_i, delta_fx, monthly_returns, in_sample_end='2018-12-31'
    )
    
    assert 'EUR' in thresholds
    assert 'rate' in thresholds['EUR']
    assert 'fx' in thresholds['EUR']
    assert 'crash' in thresholds['EUR']


def test_r_zone_signal_generation(sample_data):
    """Test R-Zone signal generation"""
    fx_data, rate_data = sample_data
    
    predictor = CurrencyCrashPredictor()
    predictor.fx_rates = fx_data
    predictor.interest_rates = rate_data
    
    delta_i, delta_fx, monthly_returns = predictor.calculate_features()
    thresholds = predictor.calculate_thresholds(delta_i, delta_fx, monthly_returns)
    r_zone = predictor.generate_r_zone_signals(delta_i, delta_fx, thresholds)
    
    assert r_zone.shape == delta_i.shape
    assert r_zone.isin([0, 1]).all().all()


def test_crash_identification(sample_data):
    """Test crash event identification"""
    fx_data, rate_data = sample_data
    
    predictor = CurrencyCrashPredictor()
    predictor.fx_rates = fx_data
    predictor.interest_rates = rate_data
    
    delta_i, delta_fx, monthly_returns = predictor.calculate_features()
    thresholds = predictor.calculate_thresholds(delta_i, delta_fx, monthly_returns)
    crashes = predictor.identify_crashes(monthly_returns, thresholds)
    
    assert crashes.shape == monthly_returns.shape
    assert crashes.isin([0, 1]).all().all()


def test_full_pipeline(sample_data):
    """Test full signal generation pipeline"""
    fx_data, rate_data = sample_data
    
    # Save sample data
    fx_data.to_csv('../data/fx_rates_test.csv')
    rate_data.to_csv('../data/interest_rates_test.csv')
    
    predictor = CurrencyCrashPredictor()
    predictor.load_data(
        fx_path='../data/fx_rates_test.csv',
        rates_path='../data/interest_rates_test.csv'
    )
    
    signals = predictor.generate_signals()
    
    assert 'r_zone' in signals
    assert 'crashes' in signals
    assert 'delta_i' in signals
    assert 'delta_fx' in signals
    assert predictor.statistics is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
