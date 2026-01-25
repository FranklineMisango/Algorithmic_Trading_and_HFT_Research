# Sentiment-Based Equity Strategy Using Market-Labeled LLMs

## Overview

**Strategy Family**: Cross-sectional, sentiment-driven equity market-neutral strategy

**One-Sentence Objective**: Generate alpha by constructing a daily-rebalanced, equal-weighted portfolio that goes long stocks with the highest AI-derived sentiment scores and short stocks with the lowest sentiment scores, exploiting the predictive relationship between market-mood and subsequent stock returns.

**Target Performance** (from research paper):
- **Annualized Return**: 35.56%
- **Sharpe Ratio**: 2.21
- **Market Regime**: All weather (market-neutral)

---

## Table of Contents

1. [Hypothesis & Economic Rationale](#hypothesis--economic-rationale)
2. [Signal Definition](#signal-definition)
3. [Data Requirements](#data-requirements)
4. [Model Architecture](#model-architecture)
5. [Portfolio Construction](#portfolio-construction)
6. [Risk Management](#risk-management)
7. [Backtesting Specification](#backtesting-specification)
8. [Execution](#execution)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Failure Modes & Mitigations](#failure-modes--mitigations)
11. [Implementation Guide](#implementation-guide)
12. [Research Citations](#research-citations)

---

## Hypothesis & Economic Rationale

### Core Hypothesis

**Sentiment extracted from financial text using a market-labeled Large Language Model (LLM) contains predictive information for future stock returns not captured by traditional factors.**

### Economic Rationale

1. **Negativity Bias in Markets**
   - Markets exhibit a psychological "negativity bias" where negative sentiment has a stronger impact on prices than positive sentiment
   - This can lead to systematic overreactions that can be exploited
   - Source: Discussion on negativity bias in markets and behavioral finance

2. **Information Asymmetry**
   - Sentiment is particularly impactful for **smaller, less-followed companies**
   - These stocks have higher information asymmetry
   - Quote from research: *"smaller companies... tend to be more volatile so it makes sense that their prices would be more sensitive to sentiment"*

3. **Market-Labeled Advantage**
   - Traditional sentiment models use human-labeled data (positive/negative)
   - **Market-labeled approach**: Train the model using actual subsequent stock returns as labels
   - This directly teaches the AI to associate language patterns with market outcomes

---

## Signal Definition

### Formula

```
SentimentScore(i, t) = f_LLM(All relevant text data for stock i up to day t)
```

Where:
- `f_LLM` = "Smarty BERT" model (BERT-based LLM fine-tuned for financial sentiment)
- **Output**: Continuous score for each stock
- **Trading Signal**: Cross-sectional rank of the sentiment score

### Labeling Method (Critical Differentiator)

Instead of human-labeled sentiment, the model is trained using:

```
Label(i, t) = AbnormalReturn(i, t+1)
             = Return(i, t+1) - Beta(i) × MarketReturn(t+1)
```

**Key Points**:
- Label is the stock's **next-day abnormal return** (market-adjusted)
- This directly aligns model training with trading objective
- Quote from research: *"they used the company's actual stock return the following day... they had to control for things like overall market trends"*

### Parameter Range

- **Long Leg**: Top decile (10%) of stocks by sentiment score
- **Short Leg**: Bottom decile (10%) of stocks by sentiment score
- **Rebalancing**: Daily

---

## Data Requirements

### Text Data (Daily Frequency)

| Data Type | Sources | Vendor Examples |
|-----------|---------|-----------------|
| **News Articles** | Financial newswires | Bloomberg, Reuters, PR Newswire, RavenPack |
| **Social Media** | Twitter/X, StockTwits, Reddit (WallStreetBets) | Accern, BuzzFeed |
| **Regulatory Filings** | 8-K, 10-K, 10-Q, Earnings Call Transcripts | SEC EDGAR, Seeking Alpha, Bloomberg Transcripts |
| **Analyst Reports** | Broker research summaries | Bloomberg, FactSet |

**Critical Data Requirements**:
- **Timestamp Accuracy**: All text must be timestamped to the minute
- **Cutoff Time**: Only data published before market close (4:00 PM ET) can be used
- **No Look-Ahead Bias**: Strict temporal alignment

### Market Data (Daily Frequency)

| Field | Description | Vendor |
|-------|-------------|--------|
| **Prices** | Open, High, Low, Close, Volume | CRSP (preferred for survivorship-bias-free) |
| **Market Cap** | Daily market capitalization | CRSP, Compustat |
| **Beta** | Market beta for abnormal return calculation | Computed from 252-day rolling regression |
| **Sector** | GICS Level 1 classification | MSCI, S&P |
| **Adjustment Factors** | Splits, dividends | CRSP |

**Survivorship Bias**: MANDATORY to use survivorship-bias-free database (e.g., CRSP includes delisted returns)

---

## Model Architecture

### Smarty BERT: Market-Labeled Financial Sentiment Model

#### Stage 1: Pre-Training
- **Base Model**: Standard BERT (`bert-base-uncased`)
- **Domain Adaptation**: Further pre-train on massive financial text corpus
  - All SEC filings (1994-present)
  - Financial news (20+ years)
  - Earnings call transcripts

#### Stage 2: Task-Specific Fine-Tuning (Market-Labeled)

**Input**: Text corpus for stock *i* on day *t*

**Label**: Continuous abnormal return on day *t+1*

```python
# Pseudo-code for label construction
abnormal_return = stock_return[t+1] - beta * market_return[t+1]

# Training objective
loss = MSE(model_output, abnormal_return)
```

**Key Innovation**: Unlike traditional sentiment models (classify as positive/negative), this model learns to predict the **magnitude** of price impact.

#### Stage 3: Inference

**Process**:
1. Aggregate all text for stock *i* on day *t*
2. Feed to Smarty BERT model
3. Output: Continuous sentiment score
4. Rank stocks cross-sectionally
5. Form portfolios: Long top 10%, Short bottom 10%

**Computational Requirements**:
- **GPU Recommended**: NVIDIA A100 or V100
- **Inference Time**: ~2 hours for full universe (3000 stocks)
- **Batch Size**: 32 (balance speed vs memory)

---

## Portfolio Construction

### Universe

- **Base**: Russell 3000 constituents
- **Filters**:
  - Share codes 10 or 11 (common stocks only)
  - Price > $5
  - Market cap > 10th percentile of NYSE
  - Min daily dollar volume > $100k

### Construction Rules

1. **Daily Ranking**: Rank all stocks by Smarty BERT sentiment score

2. **Long Leg**:
   - Select **top decile (10%)** of stocks
   - Weighting: **Equal-weighted**
   - Alternative test: Market-cap weighted

3. **Short Leg**:
   - Select **bottom decile (10%)** of stocks
   - Weighting: **Equal-weighted**

4. **Neutrality Constraints**:
   - **Dollar-Neutral**: Equal capital to long and short legs
   - **Sector-Neutral**: Constrain to be sector-neutral (GICS Level 1)
     - Max 5% deviation from sector neutrality
   - **Beta-Neutral** (optional): Can add beta constraint for market neutrality

### Weighting Scheme

**Equal-Weighted (Primary)**:
```
w_i^long = 1 / N_long
w_i^short = -1 / N_short
```

**Value-Weighted (Robustness Test)**:
Research note: *"returns were a little bit lower but the strategy was still very profitable especially when they focused on smaller companies"*

---

## Risk Management

### Position Limits

| Limit Type | Threshold | Rationale |
|------------|-----------|-----------|
| **Max Weight per Stock** | 2% (within leg) | Avoid single-stock concentration |
| **Max Stocks per Sector** | 30% of leg | Sector diversification |
| **Max Single Company Exposure** | 1% (absolute) | Total portfolio risk |

### Sector Neutrality

- **Target**: Zero net exposure to each GICS Level 1 sector
- **Max Deviation**: 5%
- **Rebalancing**: Daily optimization to maintain neutrality

### Drawdown Control

**Stop-Loss (Strategy Level)**:
- **Threshold**: 15% drawdown from peak equity
- **Action**: Trigger strategy review and de-leverage by 50%
- **Resume**: Manual decision after review

### Stress Tests

**Historical Scenarios**:
1. **COVID Crash (March 2020)**: Test performance during extreme sentiment volatility
2. **Financial Crisis (2008)**: Test model during systemic risk event
3. **News Volume Spike**: Simulate 10x normal news volume

**Monitoring Signals**:
- Daily Sharpe ratio (rolling 3-month and 12-month)
- Sentiment spread (Long - Short portfolios)
  - **Alert**: If spread narrows to < 0.5 std deviations below mean
- Turnover correlation with cost assumptions

---

## Backtesting Specification

### Universe & Period

- **Universe**: All US common stocks in Russell 3000
- **Period**: At least 10 years (e.g., 2010-2023)
- **Training**: 2010-2018
- **Validation**: 2019-2020
- **Test**: 2021-2023

### Rebalancing

- **Frequency**: **Daily** at market close
- **Order Type**: Market-On-Close (MOC) or Limit-On-Close (LOC)

### Transaction Costs & Slippage

**Realistic Cost Assumptions**:

| Stock Type | Commission (bps, one-way) | Slippage (bps) | Total Round-Trip (bps) |
|------------|---------------------------|----------------|------------------------|
| **Large Cap** | 5 | 5 | 10 |
| **Small Cap** | 5 | 15 | 20 |

**Example Calculation**:
```
Trade Size: $100,000
Large Cap Cost: $100,000 × 0.0010 = $100
Small Cap Cost: $100,000 × 0.0020 = $200
```

**Market Impact Model**:
- **Model**: Square-root (non-linear)
- **Formula**: `Impact = σ × sqrt(Q / V) × Participation_Rate`
  - σ = Stock volatility (30% annualized)
  - Q = Trade quantity
  - V = Daily volume
  - Participation Rate = 10% (max)

**Conservative Assumptions** (Recommended for Small Caps):
- Use **20-30 bps** total cost for small-cap stocks
- Test sensitivity to cost assumptions

### Survivorship Bias Control

**MANDATORY**:
- Use database that includes **delisted returns** (e.g., CRSP provides delisting returns)
- Critical because low-sentiment stocks are more likely to be delisted
- Without this, backtest will be severely biased upward

---

## Execution

### Sizing & Leverage

- **Dollar-Neutral**: Equal capital to long and short legs
- **Gross Exposure**: 200% (100% long + 100% short)
- **Net Exposure**: ~0% (target market-neutral)
- **Max Net Exposure**: 10% (allow small drift)

### Latency Assumptions

- **Signal Generation**: 2 hours (LLM inference for 3000 stocks)
- **Execution**: Rebalance by next day's market open
- **Not High-Frequency**: This is a daily strategy

### Order Execution

**Primary Method**: Market-On-Close (MOC) orders
- Submit before 3:50 PM ET
- Execute at 4:00 PM ET closing auction

**Alternative**: Limit-On-Close (LOC) for price improvement
- Risk: Non-execution if price moves away

---

## Evaluation Metrics

### Primary Metrics

| Metric | Target Benchmark | Data Source |
|--------|------------------|-------------|
| **Annualized Return** | 35.56% | Research paper |
| **Sharpe Ratio** | 2.21 | Research paper |
| **Maximum Drawdown** | < Market | Strategy requirement |
| **Win Rate** | > 55% | Estimated |
| **Sortino Ratio** | > 2.0 | Downside risk metric |
| **Calmar Ratio** | > 1.5 | Return / Max DD |

### Statistical Tests

#### 1. T-Test of Monthly Returns
```
H0: Mean monthly return = 0
Ha: Mean monthly return > 0
Alpha: 0.05
```

#### 2. Factor Regression (Fama-French + Momentum)

```
R_strategy = α + β1·(Mkt-RF) + β2·SMB + β3·HML + β4·RMW + β5·CMA + β6·UMD + ε
```

**Objective**:
- **Significant positive alpha** (α > 0, p < 0.05)
- **Low factor loadings** (|β| < 0.3 for all factors)
- This confirms the strategy generates alpha independent of traditional factors

#### 3. Newey-West Standard Errors
- Adjust for autocorrelation in returns (12 lags)
- Robust inference on alpha significance

### Capacity Analysis

**Method**: Simulate strategy with increasing capital levels

**Test Levels**:
- $10M (baseline)
- $50M
- $100M
- $500M
- $1B

**Market Impact Scaling**: Square-root law
```
Impact(Capital) = Impact(Baseline) × sqrt(Capital / Baseline)
```

**Capacity Threshold**: Capital level where Sharpe ratio degrades by 10%

**Research Insight**: *"capacity is higher in small-cap stocks but profitability may be lower in large-cap, value-weighted portfolios"*

---

## Failure Modes & Mitigations

### Ranked Failure Modes

| Rank | Failure Mode | Severity | Likelihood | Key Mitigations |
|------|--------------|----------|------------|-----------------|
| **1** | **Model Decay / Signal Crowding** | High | Medium-High | • Quarterly LLM retraining with recent data<br>• Monitor rolling 3-month Sharpe (alert if < 0.5)<br>• Introduce proprietary data sources<br>• Blend with uncorrelated signals (fundamental factors) |
| **2** | **Inadequate Cost Modeling** | High | High | • Use conservative cost assumptions (20-30 bps for small caps)<br>• Test value-weighted portfolio (lower turnover)<br>• Track actual vs assumed execution costs<br>• Implement sophisticated market impact model |
| **3** | **Data Snooping / Overfitting** | High | Medium | • Strict time-series cross-validation (no forward-looking)<br>• Hold-out final 2 years for out-of-sample test<br>• Test strategy on international markets (Europe, Asia)<br>• Use simple model architecture (avoid over-parameterization) |
| **4** | **LLM Operational Risk** | Medium | Medium | • Maintain dictionary-based sentiment as fallback (Loughran-McDonald)<br>• Monitor inference time and GPU costs<br>• Scale compute resources (cloud GPU clusters)<br>• Robust data pipelines with daily validation |
| **5** | **Regulatory & Sentiment Shift** | Low-Medium | Low | • Monitor short-selling regulations<br>• Model adapts via market-labeled training (regime-robust)<br>• Track 6-month rolling correlation (alert if < 0.1)<br>• Geographic diversification |

### Detailed Mitigation: Model Decay

**Symptoms**:
- Rolling 3-month Sharpe ratio < 0.5
- Sentiment spread (Long - Short) narrows
- Correlation with competitor strategies increases

**Actions**:
1. **Immediate**: Retrain LLM on most recent 2 years of data
2. **Medium-term**: Introduce alternative data sources (e.g., satellite imagery, credit card data)
3. **Long-term**: Blend sentiment signal with fundamental factors (blend reduces correlation with pure-sentiment competitors)

**Monitoring**:
```python
rolling_sharpe_90d = strategy_returns.rolling(90).sharpe()
if rolling_sharpe_90d.latest() < 0.5:
    trigger_alert("Model Decay Suspected")
```

### Detailed Mitigation: Inadequate Cost Modeling

**Problem**: Backtests often underestimate transaction costs, leading to overstated performance

**Solution**:
- **Base Case**: 20 bps round-trip for median stock
- **Conservative Case**: 30 bps for small caps
- **Optimistic Case**: 10 bps for large caps

**Sensitivity Test**:
```
Sharpe Ratio @ 10 bps: 2.21 (paper result)
Sharpe Ratio @ 20 bps: ~1.8 (estimated)
Sharpe Ratio @ 30 bps: ~1.4 (estimated)
```

**Mitigation**: Test **value-weighted** portfolio
- Research note: *"returns were a little bit lower but the strategy was still very profitable"*
- Value-weighting reduces turnover (larger stocks more stable in rankings)

---

## Implementation Guide

### File Structure

```
Sentiment_LLM_Equity/
├── config.yaml                     # Strategy configuration
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── data_acquisition.py             # Text + market data fetching
├── text_processing.py              # NER, entity linking, cleaning
├── sentiment_model.py              # BERT fine-tuning & inference
├── portfolio_construction.py       # Long-short, sector-neutral optimization
├── backtester.py                   # Transaction costs, market impact
├── main.py                         # Pipeline orchestration
├── 01_data_exploration.ipynb       # Text data analysis
├── 02_sentiment_analysis.ipynb     # Model output analysis
├── 03_model_training.ipynb         # Fine-tuning walkthrough
└── 04_backtest_evaluation.ipynb    # Performance metrics
```

### Dependencies

**Core Libraries**:
- `transformers` (Hugging Face): BERT model
- `torch` (PyTorch): Deep learning framework
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Utility functions
- `spacy`, `flair`: NLP (NER, entity linking)

**Data Vendors** (APIs):
- `yfinance`: Market data (free, limited)
- `sec-edgar-downloader`: SEC filings
- `praw`: Reddit API (social media)
- `tweepy`: Twitter/X API

**Visualization**:
- `matplotlib`, `seaborn`: Plotting
- `plotly`: Interactive charts

### Computational Requirements

**Minimum**:
- **CPU**: 16 cores
- **RAM**: 64 GB
- **GPU**: NVIDIA RTX 3090 (24 GB VRAM)
- **Storage**: 500 GB SSD (for text data)

**Recommended** (for full backtest):
- **GPU**: NVIDIA A100 (80 GB VRAM)
- **Cloud**: AWS p4d.24xlarge or GCP A2-ultragpu-8g

### Execution Steps

**1. Data Acquisition**:
```bash
python main.py --mode data --start-date 2010-01-01 --end-date 2023-12-31
```

**2. Text Processing**:
```bash
python main.py --mode process_text
```

**3. Model Training** (Fine-Tuning):
```bash
python main.py --mode train --epochs 3 --batch-size 16
```

**4. Backtesting**:
```bash
python main.py --mode backtest --cost-model moderate_bps
```

**5. Full Pipeline**:
```bash
python main.py --mode full
```

---

## Research Citations

### Original Research

**Title**: *"How AI Reads Market Moods to Predict Stock Success"*  
**Source**: Quant Radio (Video Transcript)  
**Key Finding**: Market-labeled LLM sentiment strategy generated 35.56% annualized return with 2.21 Sharpe ratio

### Key Quotes

> *"they used the company's actual stock return the following day... they had to control for things like overall market trends"*

> *"smaller companies... tend to be more volatile so it makes sense that their prices would be more sensitive to sentiment"*

> *"the strategy generated an annualized return of 35.56%... sharp ratio of 2.21"*

> *"sentiment is just one piece of the puzzle... fundamental analysis is still crucial"*

### Recommended Additional Reading

1. **Loughran, T., & McDonald, B.** (2011). "When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks." *Journal of Finance*.
2. **Tetlock, P. C.** (2007). "Giving content to investor sentiment: The role of media in the stock market." *Journal of Finance*.
3. **Da, Z., Engelberg, J., & Gao, P.** (2015). "The sum of all FEARS investor sentiment and asset prices." *Review of Financial Studies*.
4. **Devlin, J., et al.** (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*.

---

## Conclusion

This strategy represents a cutting-edge application of **Natural Language Processing (NLP)** and **Machine Learning** to quantitative finance. The key innovation—**market-labeled training**—differentiates it from traditional sentiment models.

**Critical Success Factors**:
1. **Data Quality**: Robust text data pipelines with strict timestamp controls
2. **Cost Realism**: Conservative transaction cost assumptions
3. **Overfitting Prevention**: Strict time-series validation
4. **Operational Excellence**: Reliable LLM inference infrastructure

**Final Note** (from research):
> *"sentiment is just one piece of the puzzle... fundamental analysis is still crucial"*

**Recommended Usage**: Blend this sentiment signal with traditional fundamental factors in a multi-factor quantitative framework for maximum robustness.

---

## License

This implementation is for research and educational purposes only. Not financial advice.

---

## Contact

For questions or collaboration: [Your Contact Information]
