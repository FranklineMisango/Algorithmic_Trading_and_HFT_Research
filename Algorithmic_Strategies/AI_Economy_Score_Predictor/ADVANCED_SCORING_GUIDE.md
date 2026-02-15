# Advanced Multi-Dimensional Sentiment Scoring

## Overview

The interactive sentiment scorer now includes comprehensive multi-aspect analysis using Mistral-7B-Instruct-v0.2 with contextual market data from Yahoo Finance.

## Features

### 1. **Multi-Aspect Sentiment Analysis**
Scores 5 key dimensions:
- **Revenue Growth**: Sales trends, top-line performance
- **Profitability**: Margins, costs, operating efficiency
- **Forward Guidance**: Future outlook, predictions
- **Management Confidence**: Tone, certainty level
- **Competitive Position**: Market share, competitive advantages

### 2. **Sectional Analysis**
- **Prepared Remarks**: Scripted management presentation (30% weight)
- **Q&A Session**: Unscripted analyst questions (70% weight)
- Automatically detects Q&A section using common markers

### 3. **Chain-of-Thought Reasoning**
For each aspect, the model:
1. Identifies key facts from the text
2. Provides reasoning for the score
3. Assigns a 1-5 sentiment score

### 4. **Market Context Integration**
Fetches from Yahoo Finance:
- Stock price at earnings date
- 1-week post-earnings return
- 1-month post-earnings return
- Enables sentiment-to-returns correlation analysis

### 5. **Intelligent Text Processing**
- Processes full transcripts (no 1000-char limit)
- Splits long text into overlapping 4000-char chunks
- Samples representative sections for efficiency
- Handles transcripts of any length

## Output Schema

### Simple Mode
```csv
symbol, date, text, sentiment_score
AAPL, 2015-01-27, ..., 4
```

### Comprehensive Mode
```csv
symbol, date, overall_sentiment, overall_sentiment_int, 
revenue_growth, profitability, forward_guidance, 
management_confidence, competitive_position,
price_at_earnings, return_1week, return_1month
```

### Detailed Results (JSONL)
```json
{
  "symbol": "AAPL",
  "date": "2015-01-27",
  "overall_sentiment": 4.3,
  "prepared_remarks": {
    "revenue_growth": {
      "score": 4.5,
      "facts": ["Revenue grew 30% YoY", "iPhone sales exceeded expectations"],
      "reasoning": "Strong top-line growth with momentum"
    },
    ...
  },
  "qa_session": { ... },
  "composite_scores": { ... }
}
```

## Usage

```bash
python interactive_sentiment_scorer.py
```

### Step-by-Step Flow

1. **Model Setup**
   - Choose 4-bit quantization (recommended for GPU)
   - Downloads Mistral-7B (~14GB, or ~4GB quantized)

2. **Data Loading**
   - Option 1: S&P 500 Earnings Transcripts (recommended)
   - Filter by year (e.g., 2015)
   - Auto-detects 'content' column

3. **Scoring Configuration**
   - Mode 1: Simple (fast, single score)
   - Mode 2: Comprehensive (30-60s per transcript, detailed)
   - Batch size: 1-5 for comprehensive, 1-10 for simple
   - Enable/disable Yahoo Finance market data

4. **Results**
   - CSV with scores and market data
   - JSON summary with statistics
   - JSONL with detailed chain-of-thought results

## Performance

### Processing Time
- **Simple mode**: ~5-10 seconds per transcript
- **Comprehensive mode**: ~30-60 seconds per transcript
  - 5 aspects × 2 sections × 3 chunks = ~30 model calls

### GPU Requirements
- **Minimum**: 8GB VRAM (with 4-bit quantization)
- **Recommended**: 16GB+ VRAM (for full precision)
- **CPU fallback**: Available but very slow (10x slower)

### Batch Processing
For 1000 transcripts:
- **Simple mode**: ~2-3 hours
- **Comprehensive mode**: ~10-15 hours (highly parallelizable)

## Analysis Examples

### 1. Sentiment-to-Returns Correlation
```python
import pandas as pd
import numpy as np

df = pd.read_csv('sentiment_scores_20260215_123456.csv')

# Correlation between sentiment and 1-week returns
correlation = df[['overall_sentiment', 'return_1week']].corr()
print(f"Sentiment-Returns correlation: {correlation.iloc[0, 1]:.3f}")
```

### 2. Track Company Sentiment Over Time
```python
# AAPL sentiment trend
aapl = df[df['symbol'] == 'AAPL'].sort_values('date')
aapl[['date', 'overall_sentiment', 'revenue_growth', 'profitability']].plot(x='date')
```

### 3. Identify Sentiment Inflection Points
```python
# Find companies with improving sentiment
df['sentiment_change'] = df.groupby('symbol')['overall_sentiment'].diff()
improving = df[df['sentiment_change'] > 1.0]  # Big jump in sentiment
```

### 4. Aspect Analysis
```python
# Which aspect predicts returns best?
for aspect in ['revenue_growth', 'profitability', 'forward_guidance', 
               'management_confidence', 'competitive_position']:
    corr = df[[aspect, 'return_1month']].corr().iloc[0, 1]
    print(f"{aspect}: {corr:.3f}")
```

## Comparison: Simple vs Comprehensive

| Feature | Simple | Comprehensive |
|---------|--------|---------------|
| Speed | ✅ Fast (5-10s) | ⚠️ Slower (30-60s) |
| Depth | Single score | 5 aspects + reasoning |
| Sections | Full text only | Prepared + Q&A split |
| Reasoning | None | Chain-of-thought |
| Market Data | No | Yes (Yahoo Finance) |
| Research Quality | Basic | Publication-grade |

## Tips for Best Results

1. **Use comprehensive mode** for research and backtesting
2. **Enable market data** to validate sentiment predictiveness
3. **Start with small samples** (e.g., 50 transcripts) to test
4. **Monitor GPU memory** - use 4-bit quantization if running low
5. **Save intermediate results** - checkpointing for long runs
6. **Analyze aspect correlations** - which aspects matter most?

## Troubleshooting

### Out of Memory
- Use 4-bit quantization
- Reduce batch size to 1
- Close other GPU applications

### Slow Processing
- Check GPU utilization (`nvidia-smi`)
- Reduce chunk size in code (`chunk_size = 2000`)
- Use simple mode for initial exploration

### Market Data Missing
- Some symbols may not be in Yahoo Finance
- Use older date range (data availability varies)
- Returns will be None if stock didn't trade those days

## Next Steps

After scoring, use the results for:
1. **Backtesting**: Build trading signals from sentiment
2. **Feature Engineering**: Use aspects as ML features
3. **Correlation Analysis**: Which aspects predict returns?
4. **Sentiment Alpha**: Combine with other factors
5. **Risk Management**: Detect deteriorating fundamentals

Happy analyzing! 
