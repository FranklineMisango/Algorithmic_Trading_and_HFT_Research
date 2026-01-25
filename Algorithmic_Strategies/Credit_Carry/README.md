# Credit Carry Strategy

## Overview
CDS spread carry strategy exploiting credit risk premia. Sells protection on credit indices with positive carry, neutralized against equity and rates risk.

## Strategy Logic

### Signal Generation
- **Primary Signal**: CDS spread carry (premium collected - expected default loss)
- **Entry**: Z-score of carry > 1.0
- **Exit**: Z-score < 0.5

### Risk Factors
- **Equity Market**: Correlation with S&P 500
- **High Yield Beta**: Sensitivity to junk bond spreads
- **Rates**: Interest rate duration

### Portfolio Construction
- **Weighting**: Inverse volatility
- **Rebalancing**: Weekly
- **Target Vol**: 10%

## Data Requirements

### CDS Spreads
- Source: Markit (via Bloomberg/Refinitiv)
- Indices: CDX.NA.IG, CDX.NA.HY, iTraxx.Europe, etc.
- Frequency: Daily

### Credit Fundamentals
- Default rates
- Recovery rates
- Credit ratings transitions

## File Structure

```
Credit_Carry/
├── config.yaml
├── requirements.txt
├── README.md
├── data_acquisition.py
├── main.py
└── notebooks/
```

## References
- Friewald, N., Wagner, C., & Zechner, J. (2014). The cross-section of credit risk premia. *Journal of Finance*
