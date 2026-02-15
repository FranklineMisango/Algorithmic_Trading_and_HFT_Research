# Databento Options Downloader Guide

## Overview
The Databento Options Downloader provides access to high-quality options data from Databento's OPRA.PILLAR dataset. This includes:
- Real-time and historical options trades
- Consolidated market data (CMBP-1)
- Top of book quotes (TCBBO)
- OHLCV bars at multiple resolutions
- Complete instrument definitions

## Key Features

### Supported Data
- **Dataset**: OPRA.PILLAR (Options Price Reporting Authority)
- **Symbols**: All US equity options (SPY, QQQ, AAPL, NVDA, TSLA, etc.)
- **Schemas**: 
  - `trades` - Actual trade ticks
  - `cmbp-1` - Consolidated Market By Price (top of book)
  - `tcbbo` - Top Consolidated Best Bid/Offer
  - `ohlcv-1s`, `ohlcv-1m`, `ohlcv-1h`, `ohlcv-1d` - OHLCV bars
  - `cbbo-1s`, `cbbo-1m` - Consolidated BBO bars
  - `statistics` - End-of-day statistics
  - `definition` - Instrument definitions

### Important Notes

1. **No Wildcards**: Databento does not support wildcards (e.g., 'SPY*')
2. **Parent Symbology**: Use `TICKER.OPT` format with `stype_in='parent'` to discover contracts
3. **Two-Step Process**:
   - Step 1: Get instrument definitions to discover option symbols
   - Step 2: Query specific option symbols for market data
4. **Option Symbol Format**: OCC format with spaces (e.g., `SPY   260321C00580000`)

## Usage

### Via Main Pipeline

Download Databento options data using the main pipeline:

```bash
# Download SPY options for the last 7 days
python main.py --source databento-options \
    --databento-options-symbols SPY \
    --start-date 2026-02-08 \
    --end-date 2026-02-15 \
    --resolution daily

# Download multiple symbols
python main.py --source databento-options \
    --databento-options-symbols SPY QQQ AAPL \
    --start-date 2026-02-01 \
    --end-date 2026-02-15 \
    --resolution minute

# Use databento as options source in general options download
python main.py --source options \
    --options-source databento \
    --option-symbols SPY AAPL \
    --start-date 2026-02-08 \
    --end-date 2026-02-15

# Test mode (limited data)
python main.py --source databento-options --test
```

### Direct Python Usage

Use the downloader directly in your Python scripts:

```python
from databento_options_downloader import DatabentoOptionsDownloader
from datetime import datetime, timedelta

# Initialize downloader
downloader = DatabentoOptionsDownloader()

# Download options data
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=7)

downloader.download_options(
    underlying='SPY',
    start_date=start_date,
    end_date=end_date,
    resolution='daily',        # 'tick', 'second', 'minute', 'hour', 'daily'
    schema='trades',           # or 'cmbp-1', 'tcbbo', 'ohlcv-1d', etc.
    limit_contracts=20,        # Limit number of contracts
    filter_near_money=True     # Filter to near-the-money options
)
```

### Advanced Usage

```python
# Download trade ticks for specific options
downloader.download_options(
    underlying='QQQ',
    start_date=datetime(2026, 2, 1),
    end_date=datetime(2026, 2, 15),
    resolution='tick',
    schema='trades',
    limit_contracts=10,
    filter_near_money=True
)

# Download minute bars (OHLCV)
downloader.download_options(
    underlying='AAPL',
    start_date=datetime(2026, 2, 10),
    end_date=datetime(2026, 2, 15),
    resolution='minute',
    schema='ohlcv-1m',
    limit_contracts=30,
    filter_near_money=True
)

# Download consolidated market data (quotes)
downloader.download_options(
    underlying='TSLA',
    start_date=datetime(2026, 2, 13),
    end_date=datetime(2026, 2, 14),
    resolution='tick',
    schema='cmbp-1',
    limit_contracts=15,
    filter_near_money=True
)
```

## Configuration

Add your Databento API key to `.env`:

```bash
DATA_BENTO_API_KEY=db-your-api-key-here
```

## Comparison with Other Options Sources

| Feature | Databento | Polygon.io | yfinance |
|---------|-----------|------------|----------|
| Historical Data | ✓ | ✓ (paid) | ✗ |
| Real-time | ✓ | ✓ | ✗ |
| Tick Data | ✓ | ✓ | ✗ |
| OHLCV Bars | ✓ | ✓ | ✗ |
| Greeks | ✗ | ✓ | ✓ |
| Free Tier | Limited | Very Limited | ✓ |
| Data Quality | Excellent | Good | Snapshot only |
| Coverage | All US Options | All US Options | Current chains only |

## Cost Considerations

- Databento charges based on data usage
- Use `limit_contracts` to control data volume
- Use `filter_near_money=True` to focus on liquid contracts
- Start with small date ranges and scale up
- Monitor your usage in the Databento dashboard

## Examples

### Example 1: Research ATM SPY Options

```bash
python main.py --source databento-options \
    --databento-options-symbols SPY \
    --start-date 2026-02-01 \
    --end-date 2026-02-15 \
    --resolution daily
```

### Example 2: High-Frequency Options Trading Data

```bash
python main.py --source databento-options \
    --databento-options-symbols SPY QQQ \
    --start-date 2026-02-14 \
    --end-date 2026-02-15 \
    --resolution tick
```

### Example 3: Multiple Underlyings for Backtesting

```bash
python main.py --source databento-options \
    --databento-options-symbols SPY QQQ AAPL NVDA TSLA \
    --start-date 2026-01-01 \
    --end-date 2026-02-15 \
    --resolution minute
```

## Troubleshooting

### Error: "Schema not supported"
- Make sure you're using the correct schema for OPRA.PILLAR
- Use `cmbp-1` instead of `mbp-1` (consolidated vs. non-consolidated)
- Valid schemas: trades, cmbp-1, tcbbo, cbbo-1s, cbbo-1m, ohlcv-*

### Error: "No data returned"
- Check that the date range has available data (typically T-1 or earlier)
- Verify the underlying symbol is correct
- Try a longer date range to ensure contracts existed during that period

### Rate Limiting
- The downloader includes built-in rate limiting (1 second between requests)
- For large downloads, consider batching or increasing the interval

## Output Format

Data is saved in QuantConnect Lean format:
```
data/option/usa/{underlying}/{resolution}/{symbol}_{date}.csv
```

### CSV Format

OHLCV data:
```csv
timestamp,open,high,low,close,volume
2026-02-15T09:30:00,50000,51000,49500,50500,1234
```

Trade data:
```csv
timestamp,price,size
2026-02-15T09:30:00.123456,50250,10
```

Quote data:
```csv
timestamp,bid_price,bid_size,ask_price,ask_size
2026-02-15T09:30:00.123456,50000,100,50100,150
```

Prices are stored as integers in deci-cents (multiply by 10000).

## References

- [Databento Documentation](https://databento.com/docs)
- [OPRA.PILLAR Dataset](https://databento.com/docs/schemas-and-data-formats)
- [Databento Python API](https://databento-python.readthedocs.io/)
