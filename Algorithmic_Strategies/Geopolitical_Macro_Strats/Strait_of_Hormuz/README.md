# Strait of Hormuz Geopolitical Risk Strategy

## Overview
Multi-asset, cross-regional trading strategy that profits from geopolitical tensions and potential closure of the Strait of Hormuz. The strategy monitors shipping traffic, oil supply disruptions, and cascading effects across global financial markets.

## Strategy Rationale

### Geopolitical Context
- **20% of global oil** passes through Strait of Hormuz
- **30% of seaborne-traded crude oil** transits this chokepoint
- Closure impacts: Energy prices, inflation expectations, supply chains, currency markets

### Market Impact Channels
1. **Energy Markets**: Crude oil, natural gas, refined products spike
2. **Equity Markets**: Energy stocks up, transportation/manufacturing down
3. **Fixed Income**: Flight to quality (US Treasuries), emerging market stress
4. **Currencies**: Oil exporters strengthen, importers weaken
5. **Commodities**: Shipping costs surge, alternative routes premium

## Data Sources

### Shipping Traffic Data

**PRIMARY SOURCE: IMF PortWatch (REAL DATA)** ✓
- **2,617 days** of actual shipping traffic (2019-2026)
- Daily tanker arrivals through key maritime chokepoints
- Average: 56 tankers/day, Range: 11-90 tankers/day
- Captures real crisis events (COVID-19, Suez blockage, geopolitical tensions)
- Source: [IMF PortWatch](https://portwatch.imf.org/)

**Fallback:** Synthetic data (if PortWatch not available)

See [PORTWATCH_DATA_GUIDE.md](PORTWATCH_DATA_GUIDE.md) for details.

### Financial Market Data

**Primary Sources:**
- **yfinance** (FREE): Equities, ETFs, indices, bonds - No API key required
- **Alpaca** (FREE tier): Real-time equities, forex, crypto, news sentiment
- **Databento** (Paid): Institutional-grade futures data (oil, gas, indices)

**Data Coverage:**
- Equities: US, Europe, Asia indices and individual stocks
- Fixed Income: US Treasuries, corporate bonds, EM bonds
- Commodities: Oil (WTI, Brent), Natural Gas, shipping costs
- Currencies: Oil exporters (NOK, CAD) vs importers (JPY, CNH)
- Crypto: BTC, ETH (risk-on/off indicators)

### Geopolitical Risk Indicators
- News sentiment (Reuters, Bloomberg via Alpaca)
- GDELT conflict events database
- US-Iran tension indices
- Military activity reports

## Strategy Components

### 1. Risk Signal Generation
- Shipping traffic anomaly detection
- Geopolitical event scoring
- Market stress indicators

### 2. Multi-Asset Positioning
**Long Positions:**
- Energy equities (XLE, oil majors)
- Oil futures (Brent, WTI)
- US Treasuries (flight to quality)
- Oil exporter currencies (NOK, CAD)
- Defense stocks

**Short Positions:**
- Transportation stocks (airlines, shipping)
- Asian manufacturing exporters
- Oil importer currencies (JPY, INR)
- Emerging market bonds

### 3. Regional Analysis
- **US**: Energy sector outperformance, defensive rotation
- **Europe**: Higher energy import costs, industrial pressure
- **Asia**: Maximum impact (China, Japan, India major importers)

## Files

- `config.yaml`: Strategy parameters and data sources
- `data_acquisition.py`: Fetch shipping, geopolitical, and market data
- `shipping_monitor.py`: AIS data processing and anomaly detection
- `geopolitical_scorer.py`: News sentiment and conflict event scoring
- `feature_engineering.py`: Cross-asset features and risk indicators
- `signal_generator.py`: Multi-asset trading signals
- `portfolio_constructor.py`: Regional and asset class allocation
- `backtester.py`: Historical event analysis and performance
- `main.py`: End-to-end pipeline execution

## Notebooks

1. `01_shipping_traffic_analysis.ipynb`: Analyze historical shipping patterns during conflicts
2. `02_market_response_analysis.ipynb`: Study price reactions across asset classes and regions
3. `03_signal_development.ipynb`: Build and test risk indicators
4. `04_backtest_evaluation.ipynb`: Performance analysis of historical geopolitical events

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional but Recommended)

The strategy works with FREE data sources, but you can enhance it with additional APIs:

**Quick Setup (Recommended):**
```bash
# Copy template
cp .env.template .env

# Edit .env and add your Alpaca keys (FREE)
# Sign up at: https://alpaca.markets
```

**API Options:**
- **yfinance**: FREE, no setup required ✓
- **Alpaca**: FREE tier available (recommended for real-time data)
- **Databento**: Paid, optional (for institutional-grade futures)

See [API_SETUP_GUIDE.md](API_SETUP_GUIDE.md) for detailed instructions.

### 3. Test Your Setup

```bash
python test_apis.py
```

This verifies your API connections and data access.

## Usage

```bash
# Run full pipeline
python main.py

# Monitor real-time shipping traffic
python shipping_monitor.py --realtime

# Generate current risk signals
python signal_generator.py --output signals.csv
```

## Historical Events for Backtesting

- 1980-1988: Iran-Iraq War (Tanker War)
- 2008: US-Iran tensions spike
- 2011-2012: EU oil embargo threats
- 2019: Tanker attacks in Gulf of Oman
- 2020: US-Iran escalation (Soleimani)
- 2024-2025: Current US-Iran conflict

## Risk Management

- Position sizing based on conflict probability
- Stop losses on false signals
- Correlation monitoring across positions
- Liquidity constraints in crisis scenarios
