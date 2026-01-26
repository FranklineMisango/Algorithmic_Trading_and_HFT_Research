import pandas as pd
import numpy as np
from binance.client import Client
import ccxt
from datetime import datetime, timedelta
from loguru import logger

class DataAcquisition:
    def __init__(self, config):
        self.config = config
        self.binance = ccxt.binance({'enableRateLimit': True})
        
    def fetch_perp_prices(self, symbol, start_date, end_date, interval='1h'):
        """Fetch perpetual futures prices from Binance"""
        logger.info(f"Fetching perp prices for {symbol}")
        
        ohlcv = self.binance.fetch_ohlcv(
            symbol + ':USDT',
            timeframe=interval,
            since=int(start_date.timestamp() * 1000),
            limit=1000
        )
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['mid_price'] = (df['high'] + df['low']) / 2
        
        return df[['mid_price', 'volume']]
    
    def fetch_spot_prices(self, symbol, start_date, end_date, interval='1h'):
        """Fetch spot prices from Binance"""
        logger.info(f"Fetching spot prices for {symbol}")
        
        ohlcv = self.binance.fetch_ohlcv(
            symbol,
            timeframe=interval,
            since=int(start_date.timestamp() * 1000),
            limit=1000
        )
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['mid_price'] = (df['high'] + df['low']) / 2
        
        return df[['mid_price', 'volume']]
    
    def fetch_funding_rates(self, symbol, start_date, end_date):
        """Fetch historical funding rates"""
        logger.info(f"Fetching funding rates for {symbol}")
        
        funding_rates = []
        current = start_date
        
        while current < end_date:
            try:
                rates = self.binance.fapiPublic_get_fundingrate({
                    'symbol': symbol.replace('/', ''),
                    'startTime': int(current.timestamp() * 1000),
                    'limit': 1000
                })
                
                for rate in rates:
                    funding_rates.append({
                        'timestamp': pd.to_datetime(rate['fundingTime'], unit='ms'),
                        'funding_rate': float(rate['fundingRate'])
                    })
                
                current += timedelta(days=30)
            except Exception as e:
                logger.error(f"Error fetching funding rates: {e}")
                break
        
        df = pd.DataFrame(funding_rates)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def fetch_risk_free_rate(self):
        """Fetch USD risk-free rate (3-month T-Bill)"""
        logger.info("Fetching risk-free rate")
        
        # Placeholder - integrate with FRED API
        # For now, return constant 5% annualized
        return 0.05
    
    def fetch_crypto_borrow_rate(self, asset='BTC'):
        """Fetch crypto borrowing rate from Binance"""
        logger.info(f"Fetching borrow rate for {asset}")
        
        # Placeholder - integrate with Binance Margin API
        # For now, return constant 8% annualized
        return 0.08
    
    def prepare_dataset(self, symbol, start_date, end_date):
        """Prepare complete dataset with all required features"""
        logger.info(f"Preparing dataset for {symbol}")
        
        # Fetch data
        perp_df = self.fetch_perp_prices(symbol, start_date, end_date)
        spot_df = self.fetch_spot_prices(symbol.replace(':USDT', '/USDT'), start_date, end_date)
        funding_df = self.fetch_funding_rates(symbol, start_date, end_date)
        
        # Merge datasets
        df = pd.DataFrame(index=perp_df.index)
        df['perp_price'] = perp_df['mid_price']
        df['spot_price'] = spot_df['mid_price']
        df['perp_volume'] = perp_df['volume']
        df['spot_volume'] = spot_df['volume']
        
        # Forward fill funding rates (8-hour frequency)
        df['funding_rate'] = funding_df['funding_rate'].reindex(df.index, method='ffill')
        
        # Add rates
        df['risk_free_rate'] = self.fetch_risk_free_rate()
        df['borrow_rate'] = self.fetch_crypto_borrow_rate(symbol.split('/')[0])
        
        # Drop NaN
        df.dropna(inplace=True)
        
        logger.info(f"Dataset prepared: {len(df)} rows")
        return df
