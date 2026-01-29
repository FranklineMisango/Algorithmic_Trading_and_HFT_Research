# Setup Guide - AI Economy Score Predictor

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Free API Keys

#### FRED API (Required - FREE)
1. Visit https://fred.stlouisfed.org/docs/api/api_key.html
2. Create free account
3. Copy your API key
4. Add to `config.yaml`:
```yaml
fred_api_key: "your_key_here"
```

#### Hugging Face (Required for Transcripts - FREE)
```bash
# Install CLI
pip install -U "huggingface_hub[cli]"

# Login (creates free account if needed)
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens
```

#### OpenAI API (Required for LLM Scoring - PAID)
1. Visit https://platform.openai.com/api-keys
2. Create account and add billing ($5 minimum)
3. Copy API key
4. Add to `config.yaml`:
```yaml
llm:
  api_key: "sk-..."
  model: "gpt-4o-mini"  # Cheapest option (~$20 for 20k transcripts)
```

**Budget Options:**
- **GPT-4o-mini**: ~$0.001/transcript = ~$20 total ✅ Recommended
- **GPT-4**: ~$0.03/transcript = ~$600 total (higher quality)
- **Local Llama-3**: Free but requires GPU (~24GB VRAM)

### 3. Test Data Access

```python
import pandas as pd
from data_acquisition import DataAcquisition

# Initialize
data = DataAcquisition("config.yaml")

# Test 1: Fetch transcripts from Hugging Face (FREE)
transcripts = data.fetch_earnings_transcripts('2015-01-01', '2015-03-31')
print(f"Loaded {len(transcripts)} transcripts for Q1 2015")

# Test 2: Fetch macro data from FRED
macro = data.fetch_macro_data('2015-01-01', '2025-12-31')
print(f"Loaded {len(macro)} macro indicators")

# Test 3: Fetch S&P 500 constituents
sp500 = data.fetch_sp500_constituents()
print(f"Loaded {len(sp500)} S&P 500 stocks")
```

## Full Pipeline Run

```bash
# Run complete pipeline (scores all 20k transcripts + backtest)
python main.py --mode full

# Or step-by-step:
python main.py --mode data      # 1. Fetch data only
python main.py --mode score     # 2. LLM scoring only
python main.py --mode backtest  # 3. Backtest only
```

## Cost Breakdown (Realistic Budget)

| Item | Cost | Status |
|------|------|--------|
| **Transcripts** (HF dataset) | **FREE** ✅ | Available now |
| **FRED macro data** | **FREE** ✅ | API key needed |
| **SPF forecasts** | **FREE** ✅ | Public data |
| **Market data** (Yahoo Finance) | **FREE** ✅ | yfinance package |
| **LLM scoring** (GPT-4o-mini) | **~$20** | One-time cost |
| **Compute** | **FREE** | Runs on laptop |
| **Total** | **~$20** 🎉 | vs $1,500-7,000 with paid transcripts |

## Expected Data Volumes

**Transcripts:**
- 2015-2020 (training): ~12,000 transcripts
- 2021-2025 (testing): ~10,000 transcripts
- Total: ~20,000 transcripts (~500-2000 MB storage)

**Macro Data:**
- GDP: ~40 quarterly observations
- Industrial Production: ~120 monthly observations
- Employment/Wages: ~120 monthly observations

**Processing Time:**
- Download transcripts: 5-10 minutes (first time, cached after)
- LLM scoring: 2-5 hours (20k transcripts @ ~1 sec each)
- Feature engineering: <5 minutes
- Model training: <1 minute
- Backtesting: <1 minute

## Troubleshooting

### "No module named 'datasets'"
```bash
pip install datasets huggingface-hub
```

### "Authentication required for Hugging Face"
```bash
huggingface-cli login
# Get token from: https://huggingface.co/settings/tokens
```

### "FRED API key invalid"
- Check key at: https://fred.stlouisfed.org/docs/api/api_key.html
- Verify in `config.yaml`: `fred_api_key: "your_actual_key"`

### "OpenAI API rate limit"
- Reduce batch size in `llm_scorer.py`
- Add `time.sleep(0.5)` between API calls
- Upgrade to paid tier for higher limits

### "Out of memory during LLM scoring"
- Process transcripts in smaller batches
- Use GPT-4o-mini instead of GPT-4
- Reduce chunk_size in config.yaml

## Next Steps

1. **Explore the data**: Open `00_full_pipeline.ipynb`
2. **Run validation**: Check transcript quality and LLM scores
3. **Train models**: GDP and industry prediction models
4. **Backtest strategy**: See historical performance
5. **Analyze results**: Compare to SPF benchmarks

## References

- **Transcripts Dataset**: https://huggingface.co/datasets/kurry/sp500_earnings_transcripts
- **FRED API Docs**: https://fred.stlouisfed.org/docs/api/
- **OpenAI Pricing**: https://openai.com/api/pricing/
- **SPF Data**: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters
