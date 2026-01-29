# AI Economy Score Predictor Strategy

## Overview

An advanced macroeconomic prediction strategy that uses Large Language Models (LLMs) to analyze earnings call transcripts and generate forward-looking economic sentiment scores. These scores predict US macroeconomic indicators (GDP, industrial production, employment, wages) with demonstrated superiority over professional forecaster consensus.

## Strategy Summary

- **Type**: Macroeconomic Prediction / Cross-Asset Sentiment Alpha
- **Frequency**: Quarterly (aligned with earnings seasons)
- **Universe**: S&P 500 stocks (transcripts) → Industry ETFs or S&P 500 futures (trading)
- **Backtest Period**: 2021-2025 (training 2015-2020)
- **Data Volume**: ~20,000 transcripts (500 companies × 4 quarters × 10 years)
- **Research Results**: Beat Survey of Professional Forecasters; GDP predictions accurate 1-4 quarters ahead, IP/employment/wages accurate up to 10 quarters ahead

## Core Hypothesis

Corporate managers possess unique "on-the-ground" insights into economic trends. When aggregated at scale using AI-based natural language understanding, their forward-looking statements in earnings calls provide predictive power for macroeconomic outcomes that exceeds traditional econometric models and professional forecasts.

## Signal Generation

### 1. Transcript Acquisition
- Source: S&P 500 earnings call transcripts (Seeking Alpha, CapIQ, Bloomberg)
- Frequency: Quarterly, post-earnings season
- Filter: Companies with >50% US revenue exposure

### 2. LLM Scoring
**Prompt**: Rate company's US economic outlook on 1-5 scale:
- 5 = Increase Substantially
- 4 = Increase Moderately  
- 3 = No Change
- 2 = Decrease Moderately
- 1 = Decrease Substantially

**Processing**:
- Model: GPT-4 (or Claude-3-Opus, Llama-3-70B)
- Temperature: 0.0 (deterministic)
- Focus: MD&A and guidance sections
- Validation: 500 human-labeled samples for quality control

### 3. Score Aggregation

**National Score (AGG_t)**:
```
AGG_t = Σ (w_i × Score_i,t) / Σ w_i
```
where w_i = market cap (value-weighted) or 1 (equal-weighted)

**Industry Score (IND_k,t)**:
```
IND_k,t = Average(Score_i,t) for all firms in GICS sector k
```

### 4. Feature Engineering
- **Normalization**: Z-score over rolling 5-year window
- **Delta Signals**: YoY change (AGG_t - AGG_{t-4})
- **N-grams**: TF-IDF fingerprint phrases (2-4 word combinations)

### 5. Prediction Models

**GDP Model**:
```
GDP_{t+h} = α + β × AGG_t + Γ × Controls_t + ε
```
- Horizons: h = 1, 2, 3, 4 quarters
- Controls: Yield curve slope, consumer sentiment, lagged GDP

**Industry IP Model**:
```
IP_{k,t+h} = α + β_k × IND_{k,t} + ε
```
- Horizons: h = 1 to 10 quarters
- Controls: Lagged IP, PMI

### 6. Trading Signals
- **Long**: If predicted GDP/IP > SPF consensus + 0.5σ
- **Short**: If predicted GDP/IP < SPF consensus - 0.5σ
- **Multi-Industry**: Long top 3 positive industries, short bottom 3

## Portfolio Construction

### National Strategy
- **Asset**: SPY (S&P 500 ETF) or ES (E-mini futures)
- **Allocation**: 100% directional based on GDP prediction
- **Rebalancing**: Quarterly

### Multi-Industry Strategy
- **Assets**: Select Sector SPDR ETFs (XLI, XLF, XLE, XLY, XLP, XLV, XLK, XLB, XLRE, XLU)
- **Allocation**: Equal risk contribution
- **Long/Short**: Top 3 / Bottom 3 by industry score
- **Max Position**: 25% per ETF

## Risk Management

### Position Limits
- Max gross exposure: 100%
- Max single position: 25%
- Max sector exposure: 40%

### Dynamic Risk Controls
- **VIX Filter**: Reduce exposure 50% when VIX > 30
- **Drawdown Stop**: Review strategy at 15% drawdown
- **Volatility Targeting**: Scale positions inversely to 20-day realized vol

### Transaction Costs
- ETFs: 3 bps commission + 5 bps slippage
- Futures: 0.5 bps commission + 1 bp slippage

## Validation & Monitoring

### Model Selection Criteria
- Out-of-sample R² > minimum threshold (GDP: 0.15, IP: 0.10)
- Beat SPF consensus MAPE (Mean Absolute Prediction Error)
- Coefficient β statistically significant at 95% confidence

### Quality Checks
- LLM-human correlation > 0.80 on validation set
- Transcript completeness > 95% of universe by market cap
- Quarterly AGG_t vs GDP surprise correlation > 0.30

### Statistical Tests
- **Diebold-Mariano Test**: Confirm forecast superiority vs SPF
- **Chow Test**: Detect structural breaks post-2020
- **Bootstrap**: Assess prediction robustness

## Key Research Findings

1. **Predictive Accuracy**:
   - GDP: Accurate 1-4 quarters ahead
   - Industrial Production / Employment / Wages: Accurate up to 10 quarters ahead
   
2. **Benchmark Comparison**:
   - AI Economy Score beat Survey of Professional Forecasters
   - Managers have "unique insights even sophisticated models miss"

3. **N-gram Fingerprints**:
   - Positive: "strong financial performance", "robust demand"
   - Negative: "challenging economic environment", "macroeconomic headwinds"

## Failure Modes & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| LLM context shift (misinterprets new slang) | High | Continuous human validation, ensemble models |
| Management bias (over-optimistic guidance) | High | Control for stock performance, guidance vs sentiment divergence |
| Public firms only (misses small business) | Medium | Blend with NFIB Small Business Optimism Index |
| Overfitting to crises (2008/2020) | Medium | Regime-switching model, diverse training periods |
| API dependency (GPT access/cost changes) | Medium | Fine-tune open-source fallback (Llama-3) |
| Missing transcripts | Low | Completeness check, flag if >5% missing by cap |

## Data Requirements

### Primary Data
1. **Earnings Transcripts**: **FREE via Hugging Face** 🎉
   - **Source**: `kurry/sp500_earnings_transcripts` (Hugging Face dataset)
   - **Period**: 2005-2025 (20 years available, using 2015-2025)
   - **Volume**: ~20,000 transcripts (500 companies × 4 quarters × 10 years)
   - **Training Set**: 2015-2020 (~12,000 transcripts)
   - **Test Set**: 2021-2025 (~10,000 transcripts)
   - **Access**: `pip install datasets` + `huggingface-cli login` (free account)
   - **Alternative Paid Sources**: S&P Capital IQ, Bloomberg, Seeking Alpha (~$1,200-20,000/year)
2. **Macroeconomic**: FRED (GDP, INDPRO, PAYEMS, CES wages) - FREE
3. **SPF Consensus**: Philadelphia Fed Survey of Professional Forecasters - FREE
4. **Market Data**: Yahoo Finance (free) or CRSP/Compustat (institutional)

### LLM Access
- **Preferred**: OpenAI GPT-4 API
- **Alternatives**: Anthropic Claude-3-Opus, local Llama-3-70B
- **Cost**: ~$0.01-0.03 per transcript (GPT-4)
- **Total LLM Cost**: ~$200-600 for 20,000 transcripts (one-time processing)
- **Budget Option**: Use GPT-4o-mini (~$0.001/transcript) = ~$20 total

## Implementation Notes

### Prompt Engineering
Critical to strategy success. Template must:
1. Focus on **US economy** specifically (not firm-specific outlook)
2. Use clear Likert scale with anchors
3. Be tested against 500+ human-labeled samples
4. Achieve >85% inter-rater reliability

### Computational Requirements
- **LLM API calls**: ~500 transcripts × 4 quarters/year × $0.02 = ~$40/year
- **Storage**: ~500MB transcripts + 10MB scores per year
- **Compute**: Minimal (regression models, no GPU required for inference)

### Production Considerations
- Quarterly execution (low frequency, low monitoring overhead)
- Transcript availability lag: ~2-4 weeks post-quarter end
- Model retraining: Annually with expanding window
- Fallback: Use prior quarter's score if <95% transcript completeness

## Files

- `config.yaml` - Strategy configuration and parameters
- `requirements.txt` - Python dependencies
- `data_acquisition.py` - Fetch transcripts and macro data
- `llm_scorer.py` - LLM prompt engineering and scoring
- `feature_engineering.py` - N-gram analysis, normalization
- `prediction_model.py` - Regression models for macro predictions
- `signal_generator.py` - Trading signals from predictions
- `backtester.py` - Strategy backtest with costs
- `main.py` - Full pipeline orchestration
- `lean_algorithm.py` - QuantConnect implementation

## References

- Research: "From Earnings Calls to Economic Predictions" (Quant Radio)
- Data: FRED, Philadelphia Fed SPF, earnings transcript vendors
- Models: GPT-4 (OpenAI), Claude-3 (Anthropic), Llama-3 (Meta)
