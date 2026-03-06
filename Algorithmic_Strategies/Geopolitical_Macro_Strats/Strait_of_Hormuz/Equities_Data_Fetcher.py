import yfinance as yf
import pandas as pd
import os
import time
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configuration
START_DATE = "2018-12-31"
END_DATE = "2026-03-04"
BASE_DATA_DIR = "Data"

# Mapping Market Data Categories to Directories
# Updated MARKET_MAP for Equities_Data_Fetcher.py
MARKET_MAP = {
    "Equities/US": [
        "SPY", "XLE", "XLI", "ITA", "JETS", "XOM", "CVX", "LMT", "BA", 
        "LNG", "WDS",  # WDS replaces TELL (Tellurian)
        "COP", "EOG", "SLB", "HAL", "FRO", "STNG", "TNK", "DHT"
    ],
    "Equities/Europe": [
        "EZU", "IEO", "EWG", "DXGE", 
        "EWI",
        "EPOL", "EQNR", "NORW"
    ],
    "Equities/Asia": [
        "EWJ", "DXJ", "EWY", "INDA", "INDY", "FXI", "ASHR", "MCHI", "EWT", "EWA"
    ],
    "Fixed_Income": [
        "TLT", "IEF", "SHY", "LQD", "HYG", "EMB", "EMLC"
    ],
    "Commodities": [
        "USO", "BNO", "UNG", "BDRY"
    ]
}


def get_robust_session():
    session = Session()
    retry = Retry(total=5, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def fetch_and_save_market_data():
    session = get_robust_session()

    for sub_dir, tickers in MARKET_MAP.items():
        # Create specific directory path
        target_dir = os.path.join(BASE_DATA_DIR, sub_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"Created directory: {target_dir}")

        for ticker in tickers:
            print(f"Fetching {ticker} for {sub_dir}...")
            
            try:
                # Download historical daily data
                df = yf.download(
                    ticker, 
                    start=START_DATE, 
                    end=END_DATE, 
                    interval="1d", 
                    session=session,
                    progress=False
                )

                if df.empty:
                    print(f"Warning: No data for {ticker}")
                    continue

                # Clean data: Keep Close prices and handle gaps
                df_clean = df[['Close']].ffill().dropna()
                
                # Save CSV
                csv_path = os.path.join(target_dir, f"{ticker}.csv")
                df_clean.to_csv(csv_path)

                # Save Parquet (Preserves metadata/types)
                parquet_path = os.path.join(target_dir, f"{ticker}.parquet")
                df_clean.to_parquet(parquet_path, engine='pyarrow')

                # Respect rate limits: 1.5s delay between individual calls
                time.sleep(1.5)

            except Exception as e:
                print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    print("Starting Financial Market Data Download...")
    fetch_and_save_market_data()
    print("Download Complete.")
