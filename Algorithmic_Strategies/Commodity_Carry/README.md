# Commodity Carry Strategy

## Overview
Commodity futures carry strategy based on convenience yield. Goes long commodities in backwardation (positive carry) and short those in contango (negative carry).

## Strategy Logic

### Signal Generation
- **Primary Signal**: Convenience yield = Spot - Forward price (storage/financing adjusted)
- **Entry**: Z-score of carry > 1.0 (backwardation)
- **Exit**: Z-score < 0.5

### Backwardation vs Contango
- **Backwardation**: Spot > Forward (positive carry, go long)
- **Contango**: Forward > Spot (negative carry, go short)

### Risk Factors
- **Dollar Index**: USD strength affects commodity prices
- **Equity Market**: Risk-on/risk-off flows
- **Energy Sector**: Oil/gas correlation

### Portfolio Construction
- **Weighting**: Inverse volatility
- **Rebalancing**: Weekly
- **Sector Limits**: Max 40% to energy/metals/agriculture

## Data Requirements

### Futures Curves
- Source: Databento, Polygon, Quandl
- Commodities: CL, HG, GC, C, W, NG, etc.
- Contracts: Front month + 3M, 6M, 12M

### Inventory Data
- EIA petroleum inventories
- USDA crop reports
- LME metals stocks

## File Structure

```
Commodity_Carry/
├── config.yaml
├── requirements.txt
├── README.md
├── data_acquisition.py
├── main.py
└── notebooks/
```

## References
- Gorton, G., Hayashi, F., & Rouwenhorst, K. (2013). The fundamentals of commodity futures returns. *Review of Finance*
- Szymanowska, M., De Roon, F., Nijman, T., & Van Den Goorbergh, R. (2014). An anatomy of commodity futures risk premia. *Journal of Finance*
