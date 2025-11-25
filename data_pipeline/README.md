# LEAN Data Pipeline

This data pipeline downloads financial data from Alpaca (for US equities) and Binance (for cryptocurrencies) and converts it to LEAN format for backtesting.

## Features

- **Alpaca Integration**: Download US equity data (OHLCV) from Alpaca Markets
- **Binance Integration**: Download cryptocurrency data (OHLCV) from Binance
- **Lean Format**: Automatically converts data to LEAN's CSV format with proper compression
- **Multiple Resolutions**: Supports minute, hour, and daily data
- **Rate Limiting**: Built-in rate limiting to respect API limits
- **Data Validation**: Validates OHLCV data integrity
- **Timezone Handling**: Proper timezone conversion for different markets

## Setup

1. **Install Dependencies**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Set API Keys** (create a `.env` file or export environment variables):
   ```bash
   # Alpaca API keys (required for equity data)
   export ALPACA_API_KEY="your_alpaca_api_key"
   export ALPACA_SECRET_KEY="your_alpaca_secret_key"
   
   # Binance API keys (optional for public data)
   export BINANCE_API_KEY="your_binance_api_key"
   export BINANCE_SECRET_KEY="your_binance_secret_key"
   ```

3. **Activate Virtual Environment**:
   ```bash
   source data_pipeline_env/bin/activate
   ```

## Getting API Keys

### Alpaca (for US Equity Data)
1. Go to [Alpaca Markets](https://alpaca.markets/)
2. Create a free account
3. Get your API keys from the dashboard
4. Use paper trading keys for testing

### Binance (for Cryptocurrency Data)
1. Go to [Binance](https://www.binance.com/)
2. Create an account
3. Generate API keys (optional - public data works without keys)
4. For public data, you can leave the keys empty

## Usage

### Basic Usage

```bash
# Test with limited data
python main.py --test

# Download daily equity data from Alpaca
python main.py --source alpaca --resolution daily

# Download minute crypto data from Binance
python main.py --source binance --resolution minute

# Download from both sources
python main.py --source both --resolution daily
```

### Advanced Usage

```bash
# Custom symbols and date range
python main.py --source alpaca \
               --equity-symbols AAPL GOOGL MSFT TSLA \
               --start-date 2024-01-01 \
               --end-date 2024-12-31 \
               --resolution minute

# Custom crypto symbols
python main.py --source binance \
               --crypto-symbols BTCUSDT ETHUSDT ADAUSDT \
               --start-date 2024-06-01 \
               --end-date 2024-07-01 \
               --resolution hour
```

### Command Line Options

- `--source`: Data source (`alpaca`, `binance`, or `both`)
- `--equity-symbols`: List of equity symbols to download
- `--crypto-symbols`: List of crypto symbols to download
- `--start-date`: Start date (YYYY-MM-DD format)
- `--end-date`: End date (YYYY-MM-DD format)
- `--resolution`: Data resolution (`minute`, `hour`, `daily`)
- `--test`: Run in test mode with limited data

## Data Format

The pipeline converts data to LEAN's standard format:

### Equity Data (Alpaca)
- **Location**: `data/equity/usa/resolution/symbol/`
- **Format**: CSV files compressed in ZIP
- **Filename**: `YYYYMMDD_trade.zip` (for minute/hour) or `symbol.zip` (for daily)
- **Fields**: Time (ms since midnight), Open, High, Low, Close, Volume
- **Price Format**: Deci-cents (price * 10000)

### Crypto Data (Binance)
- **Location**: `data/crypto/binance/resolution/symbol/`
- **Format**: CSV files compressed in ZIP
- **Filename**: `YYYYMMDD_trade.zip` (for minute/hour) or `symbol.zip` (for daily)
- **Fields**: Time (ms since midnight), Open, High, Low, Close, Volume
- **Price Format**: Actual prices (no conversion)

## Directory Structure

```
data_pipeline/
 config.py              # Configuration settings
 utils.py               # Utility functions
 alpaca_downloader.py   # Alpaca data downloader
 binance_downloader.py  # Binance data downloader
 main.py               # Main pipeline script
 setup.sh              # Setup script
 requirements.txt      # Python dependencies
 README.md            # This file
```

## Configuration

Edit `config.py` to customize:
- Default symbols to download
- Date ranges
- File paths
- Rate limiting settings
- API endpoints

## Rate Limits

The pipeline respects API rate limits:
- **Alpaca**: 200 requests per minute
- **Binance**: 1200 requests per minute

## Troubleshooting

### Common Issues

1. **API Key Errors**: Make sure your API keys are set correctly
2. **Rate Limiting**: The pipeline includes rate limiting, but if you get errors, try increasing the delay
3. **Data Gaps**: Some symbols may not have data for certain periods
4. **Timezone Issues**: Make sure your system timezone is configured correctly

### Logs

The pipeline creates a log file `data_pipeline.log` with detailed information about the download process.

## Examples

### Download 30 Days of Minute Data
```bash
python main.py --source both \
               --resolution minute \
               --start-date 2024-06-01 \
               --end-date 2024-07-01 \
               --equity-symbols AAPL GOOGL MSFT \
               --crypto-symbols BTCUSDT ETHUSDT
```

### Download 1 Year of Daily Data
```bash
python main.py --source both \
               --resolution daily \
               --start-date 2023-01-01 \
               --end-date 2023-12-31
```

## Integration with LEAN

After downloading data, it will be available in your LEAN backtests. The data structure matches LEAN's expectations:

- Equity data: `data/equity/usa/minute/aapl/20240701_trade.zip`
- Crypto data: `data/crypto/binance/minute/btcusdt/20240701_trade.zip`

## Cost Considerations

- **Alpaca**: Free for real-time and historical data (with some limitations)
- **Binance**: Free for public market data
- **Storage**: Data is compressed but can grow large with high-resolution data

## Next Steps

1. Run the pipeline to download initial data
2. Test with your LEAN strategies
3. Set up automated downloads using cron jobs
4. Monitor data quality and gaps
5. Consider adding more data sources as needed
