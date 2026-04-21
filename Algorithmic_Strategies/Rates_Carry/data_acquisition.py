"""
Rates Carry Strategy - Data Acquisition Module
Fetches government bond yields and calculates roll-down returns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from pandas_datareader import data as pdr
import time
import yaml
from datetime import datetime
from typing import Dict, Tuple

try:
    import investpy
except ImportError:
    investpy = None


class RatesDataAcquisition:
    """Fetch yield curves and bond price data"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.countries = self.config['data']['countries']
        self.maturities = self.config['data']['maturities']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date'] or datetime.today().strftime('%Y-%m-%d')
        self.bond_etf_tickers = self.config['data'].get('bond_etf_tickers', {})
        
        import os
        fred_api_key = os.getenv('FRED_API_KEY', None)
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None

    @staticmethod
    def _normalize_series(data: pd.Series | pd.DataFrame, series_code: str | None = None) -> pd.Series:
        """Convert provider payload to a clean numeric time series."""
        if isinstance(data, pd.DataFrame):
            if series_code and series_code in data.columns:
                data = data[series_code]
            elif 'Close' in data.columns:
                data = data['Close']
            else:
                data = data.iloc[:, 0]

        series = pd.to_numeric(data, errors='coerce').dropna()
        series.index = pd.to_datetime(series.index)
        return series.sort_index()

    def _fetch_fred_series(self, series_code: str) -> pd.Series:
        if self.fred:
            data = self.fred.get_series(
                series_code,
                observation_start=self.start_date,
                observation_end=self.end_date,
            )
        else:
            data = pdr.DataReader(series_code, 'fred', self.start_date, self.end_date)
        return self._normalize_series(data, series_code)

    def _fetch_investing_bond_series(self, bond_name: str, retries: int = 3) -> pd.Series:
        if investpy is None:
            raise RuntimeError("investpy not installed")

        from_date = pd.to_datetime(self.start_date).strftime('%d/%m/%Y')
        to_date = pd.to_datetime(self.end_date).strftime('%d/%m/%Y')
        last_error = None

        for attempt in range(retries):
            try:
                data = investpy.get_bond_historical_data(
                    bond=bond_name,
                    from_date=from_date,
                    to_date=to_date,
                )
                return self._normalize_series(data)
            except Exception as e:
                last_error = e
                # Small backoff helps with temporary upstream rate limits.
                time.sleep(0.8 * (attempt + 1))

        raise RuntimeError(last_error)

    def _fetch_yahoo_price_series(self, ticker: str) -> pd.Series:
        data = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True,
            actions=False,
            threads=False,
        )
        if data.empty or 'Close' not in data.columns:
            raise RuntimeError("empty price history")
        series = self._normalize_series(data['Close'])
        if series.empty:
            raise RuntimeError("no valid close prices")
        return series.rename(ticker)
    
    def fetch_yield_curves(self) -> pd.DataFrame:
        """Download government bond yields from real external data sources"""
        print(f"Fetching yield data for {len(self.countries)} countries...")

        # High-quality US constant-maturity series from FRED.
        fred_series = {
            ('US', 2): 'DGS2',
            ('US', 5): 'DGS5',
            ('US', 7): 'DGS7',
            ('US', 10): 'DGS10',
            ('US', 30): 'DGS30',
        }

        # Legacy FRED fallback for non-US long tenor only.
        fred_fallback_series = {
            ('Germany', 10): 'IRLTLT01DEM156N',
            ('UK', 10): 'IRLTLT01GBM156N',
            ('Japan', 10): 'IRLTLT01JPM156N',
            ('Australia', 10): 'IRLTLT01AUM156N',
            ('Canada', 10): 'IRLTLT01CAM156N',
        }

        investing_country_prefix = {
            'US': 'U.S.',
            'Germany': 'Germany',
            'UK': 'U.K.',
            'Japan': 'Japan',
            'Australia': 'Australia',
            'Canada': 'Canada',
        }
        
        yields_data = {}
        
        for country in self.countries:
            for maturity in self.maturities:
                key = f"{country}_{maturity}Y"
                source_notes = []

                # 1) Prefer FRED for US where full tenor curve is available.
                if (country, maturity) in fred_series:
                    try:
                        series = self._fetch_fred_series(fred_series[(country, maturity)])
                        if not series.empty:
                            yields_data[key] = series.rename(key)
                            print(f"  ✓ {key}: {len(series)} observations [FRED]")
                            continue
                    except Exception as e:
                        source_notes.append(f"FRED failed: {e}")

                # 2) Try investpy sovereign bond yields for multi-country tenors.
                if country in investing_country_prefix:
                    bond_name = f"{investing_country_prefix[country]} {maturity}Y"
                    try:
                        series = self._fetch_investing_bond_series(bond_name)
                        if not series.empty:
                            yields_data[key] = series.rename(key)
                            print(f"  ✓ {key}: {len(series)} observations [Investing]")
                            continue
                    except Exception as e:
                        source_notes.append(f"Investing failed: {e}")

                # 3) FRED fallback for non-US 10Y macro series.
                if (country, maturity) in fred_fallback_series:
                    try:
                        series = self._fetch_fred_series(fred_fallback_series[(country, maturity)])
                        if not series.empty:
                            yields_data[key] = series.rename(key)
                            print(f"  ✓ {key}: {len(series)} observations [FRED fallback]")
                            continue
                    except Exception as e:
                        source_notes.append(f"FRED fallback failed: {e}")

                reason = '; '.join(source_notes) if source_notes else 'no configured source'
                print(f"  - {key}: unavailable ({reason})")

        if not yields_data:
            raise RuntimeError(
                "No yield data could be loaded from configured real data providers."
            )

        # Build one aligned panel and keep only observed-source values (no synthetic interpolation).
        df_yields = pd.concat(yields_data.values(), axis=1).sort_index()
        if not isinstance(df_yields.index, pd.DatetimeIndex):
            df_yields.index = pd.to_datetime(df_yields.index)

        df_yields = df_yields.resample('D').ffill()

        print(f"\nYield data shape: {df_yields.shape}")
        print(f"Available tenor columns: {len(df_yields.columns)}")
        return df_yields
    
    def calculate_rolldown(self, yields_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate roll-down return for each bond"""
        print("\nCalculating roll-down yields...")
        
        roll_period = self.config['signals']['roll_period']  # days
        
        rolldown = pd.DataFrame(index=yields_df.index)
        
        # For each maturity, calculate expected return from rolling down curve
        for col in yields_df.columns:
            if '10Y' in col:
                # 10Y bond rolling to 9.75Y
                country = col.split('_')[0]
                current_yield = yields_df[col]
                
                # Approximate yield at 9.75Y (roll_period/252 years shorter)
                # Use linear interpolation from 7Y and 10Y
                if f"{country}_7Y" in yields_df.columns:
                    nearby_yield = yields_df[f"{country}_7Y"]
                    roll_yield = current_yield - (current_yield - nearby_yield) * (roll_period/252) / 3
                else:
                    roll_yield = current_yield
                
                # Roll-down return = (current_yield - roll_yield) * duration
                duration = 9.0  # Approx duration of 10Y bond
                rolldown[col] = (current_yield - roll_yield) * duration
        
        print(f"Roll-down data shape: {rolldown.shape}")
        return rolldown

    def fetch_bond_prices(self) -> pd.DataFrame:
        """Download bond ETF proxy prices from Yahoo Finance"""
        print("\nFetching bond ETF proxy prices from Yahoo Finance...")

        prices_data = {}

        for country in self.countries:
            tickers = self.bond_etf_tickers.get(country, [])
            if not tickers:
                print(f"  - {country}: no configured ETF tickers")
                continue

            for ticker in tickers:
                try:
                    series = self._fetch_yahoo_price_series(ticker)
                    prices_data[ticker] = series
                    print(f"  ✓ {ticker}: {len(series)} observations")
                except Exception as e:
                    print(f"  - {ticker}: unavailable ({e})")

        if not prices_data:
            raise RuntimeError("No ETF proxy prices could be loaded from Yahoo Finance.")

        prices_df = pd.concat(prices_data.values(), axis=1).sort_index()
        if not isinstance(prices_df.index, pd.DatetimeIndex):
            prices_df.index = pd.to_datetime(prices_df.index)

        prices_df = prices_df.resample('D').ffill()
        print(f"Bond price data shape: {prices_df.shape}")
        return prices_df
    
    def save_data(
        self,
        yields_df: pd.DataFrame,
        rolldown_df: pd.DataFrame,
        prices_df: pd.DataFrame | None = None,
    ):
        """Save data to CSV"""
        import os
        os.makedirs('data', exist_ok=True)
        
        yields_df.to_csv('data/bond_yields.csv')
        rolldown_df.to_csv('data/rolldown.csv')
        if prices_df is not None:
            prices_df.to_csv('data/bond_prices.csv')
        print("\n✓ Data saved to data/ directory")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load saved data"""
        yields = pd.read_csv('data/bond_yields.csv', index_col=0, parse_dates=True)
        rolldown = pd.read_csv('data/rolldown.csv', index_col=0, parse_dates=True)
        return yields, rolldown

    def load_bond_prices(self) -> pd.DataFrame:
        """Load saved bond ETF proxy prices"""
        return pd.read_csv('data/bond_prices.csv', index_col=0, parse_dates=True)


if __name__ == "__main__":
    rates_data = RatesDataAcquisition()
    yields = rates_data.fetch_yield_curves()
    rolldown = rates_data.calculate_rolldown(yields)
    prices = rates_data.fetch_bond_prices()
    rates_data.save_data(yields, rolldown, prices)
