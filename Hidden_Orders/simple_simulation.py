import pandas as pd

def detect_hidden_orders(df, progress_interval=1000):
    """Identify trades at prices not visible in the order book."""
    hidden_signals = []
    total_rows = len(df)
    
    print(f"Starting analysis of {total_rows} rows...")
    
    for i in range(len(df)):
        row = df.iloc[i]
        trade_price = row['trade_price']
        bid = row['bid_price']
        ask = row['ask_price']

        # Show progress periodically
        if i % progress_interval == 0:
            print(f"Processing row {i}/{total_rows} ({i/total_rows*100:.1f}%)")
            
        if pd.notnull(trade_price):
            if trade_price < bid or trade_price > ask:
                # Print alert as soon as hidden order is detected
                print(f"🔍 Hidden order at {row['timestamp']}: Trade price {trade_price}, outside of bid {bid} - ask {ask}")
                
                hidden_signals.append({
                    'timestamp': row['timestamp'],
                    'trade_price': trade_price,
                    'bid': bid,
                    'ask': ask,
                    'note': 'Possible hidden order detected'
                })

    hidden_df = pd.DataFrame(hidden_signals)
    if not hidden_df.empty:
        print(f"\n⚠️  Total hidden order signals: {len(hidden_df)}")
        print(hidden_df.head(3))
    else:
        print("\nNo hidden orders detected.")
    return hidden_df


#sample data 
data = pd.read_csv("HFT_Binance_data_fetcher/HFT_100ms_unresampled_data_combined_data.csv")
detect_hidden_orders(data)