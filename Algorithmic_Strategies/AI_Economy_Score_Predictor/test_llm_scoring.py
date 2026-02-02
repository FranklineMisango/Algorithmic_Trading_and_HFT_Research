"""
Test LLM Scoring to Diagnose Why All Scores Are 3.0
"""

import pandas as pd
import yaml
from llm_scorer import LLMScorer
import random

# Load actual transcript data
print("="*70)
print("LLM SCORING DIAGNOSTIC TEST")
print("="*70)

# Initialize scorer
scorer = LLMScorer('config.yaml')

# Load a sample of transcripts with different scores
df = pd.read_csv('all_scored_transcripts_2015_2025.csv')

# Get samples: 
# - Some with score 3.0 (the majority)
# - Some with non-3.0 scores (if they exist meaningfully)
score_3 = df[df['score'] == 3.0].sample(3)
score_not_3 = df[df['score'] != 3.0].head(5)

samples = pd.concat([score_3, score_not_3])

print(f"\nTesting {len(samples)} transcripts...")
print(samples[['symbol', 'date', 'score']].to_string())

# Test with sample text (simulate what we'd see in a transcript)
test_cases = [
    {
        'name': 'Very Positive',
        'text': """
        We're seeing exceptional strength across all our business segments. 
        The US economy is incredibly robust, with consumer spending at record highs. 
        We're expanding aggressively and hiring thousands of new employees. 
        Our outlook for next quarter and the full year is extremely optimistic. 
        We expect GDP growth to accelerate and unemployment to remain low.
        """
    },
    {
        'name': 'Neutral',
        'text': """
        Our results were in line with expectations this quarter.
        The economic environment remains stable with no major changes anticipated.
        We're maintaining current operations and monitoring conditions closely.
        Guidance for next quarter is consistent with previous expectations.
        """
    },
    {
        'name': 'Negative', 
        'text': """
        We're facing significant headwinds from the weakening economy.
        Consumer sentiment has deteriorated and spending is slowing.
        We're implementing cost reduction measures and pausing hiring.
        The outlook is challenging with recession risks increasing.
        We expect GDP growth to slow and unemployment to rise.
        """
    },
    {
        'name': 'Generic Boilerplate',
        'text': """
        Thank you for joining our earnings call today.
        I'll now discuss our financial results for the quarter.
        Revenue was in line with guidance. Expenses were well controlled.
        We continue to focus on operational excellence and shareholder value.
        I'll now turn it over to our CFO for detailed financials.
        """
    }
]

print("\n" + "="*70)
print("TESTING LLM WITH DIFFERENT SENTIMENT TEXTS")
print("="*70)

for test in test_cases:
    print(f"\n{test['name']}:")
    print("-" * 50)
    print(f"Text: {test['text'][:150]}...")
    
    # Score the text
    score = scorer.score_text(test['text'])
    
    if score is not None:
        print(f"✓ Score: {score}")
    else:
        print(f"✗ Failed to get score (API error or no client)")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

if scorer.client is None:
    print("❌ CRITICAL: No LLM client initialized!")
    print("   This means scoring is using the fallback (random scores)")
    print("   Check API key configuration in config.yaml")
else:
    print("✓ LLM client initialized")
    print("\nIf all test scores came back as 3, the issues could be:")
    print("1. Prompt is too conservative → LLM defaults to neutral")
    print("2. gpt-4o-mini may be worse at following instructions than gpt-4")
    print("3. 2000 char chunks lose context → everything looks neutral")
    print("4. Temperature=0.0 causes deterministic anchoring to 3")
    print("\nRECOMMENDATIONS:")
    print("- Test with temperature=0.3 for slight variation")
    print("- Use full transcript or larger chunks (5000+ chars)")
    print("- Try gpt-4 or claude-3-opus for better instruction following")
    print("- Add examples in prompt (few-shot learning)")
    print("- Extract specific metrics/keywords first, then score based on those")

print("="*70)
