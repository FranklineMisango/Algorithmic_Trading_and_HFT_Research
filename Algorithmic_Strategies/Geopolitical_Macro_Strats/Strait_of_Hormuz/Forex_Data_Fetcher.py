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
BASE_DIR = "Data/Forex"

# Map for filename-safe naming
SYMBOLS = {
    "AUDUSD=X": "AUD_USD_LNG_Exporter",
    "USDCAD=X": "USD_CAD_Oil_Exporter",
    "USDNOK=X": "USD_NOK_Gas_Exporter",
    "USDJPY=X": "USD_JPY_Oil_Importer",
    "USDKRW=X": "USD_KRW_Oil_Importer",
    "USDINR=X": "USD_INR_Oil_Importer",
    "CNH=F":    "USD_CNH_Strategic_Importer"
}

def get_robust_session():
    session = Session()
    retry = Retry(total=5, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def fetch_individual_histories():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    session = get_robust_session()

    for ticker, name in SYMBOLS.items():
        print(f"Fetching: {name} ({ticker})...")
        
        try:
            # Fetch single ticker
            df = yf.download(
                ticker, 
                start=START_DATE, 
                end=END_DATE, 
                interval="1d", 
                session=session,
                progress=False
            )

            if df.empty:
                print(f"Skipping {name}: No data found.")
                continue

            # Clean: Keep only Close, forward fill gaps
            df_clean = df[['Close']].ffill().dropna()
            df_clean.columns = [name] # Rename column to the friendly name

            # Save CSV
            csv_file = os.path.join(BASE_DIR, f"{name}.csv")
            df_clean.to_csv(csv_file)

            # Save Parquet (Requires pyarrow)
            parquet_file = os.path.join(BASE_DIR, f"{name}.parquet")
            df_clean.to_parquet(parquet_file, engine='pyarrow')

            print(f"Saved {name} to {BASE_DIR}")

            # Rate Limit Protection: Wait 2 seconds between separate ticker calls
            time.sleep(2)

        except Exception as e:
            print(f"Failed to fetch {name}: {e}")

if __name__ == "__main__":
    fetch_individual_histories()
    print("\nAll individual files processed.")
