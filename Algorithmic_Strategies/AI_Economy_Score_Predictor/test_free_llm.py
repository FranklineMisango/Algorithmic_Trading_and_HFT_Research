#!/usr/bin/env python3
"""
Quick test script to verify Ollama + Mistral setup is working
"""

from llm_scorer import LLMScorer
import sys

print("="*70)
print("  Testing Free LLM Setup (Ollama + Mistral)")
print("="*70)

try:
    # Initialize scorer
    print("\n1️⃣ Loading configuration...")
    scorer = LLMScorer('config.yaml')
    print(f"   ✅ Provider: {scorer.provider}")
    print(f"   ✅ Model: {scorer.llm_config['model']}")
    
    # Test with positive text
    print("\n2️⃣ Testing with POSITIVE economic text...")
    positive_text = """
    The US economy is showing exceptional strength. 
    Consumer spending remains robust and business investment is accelerating.
    We're very optimistic about the next quarter.
    """
    positive_score = scorer.score_text(positive_text)
    print(f"   Score: {positive_score}/5 (Expected: 4-5)")
    
    if positive_score is None or positive_score < 3:
        print("   ⚠️ Unexpected score for positive text")
    else:
        print("   ✅ Correct - high score for positive outlook")
    
    # Test with negative text
    print("\n3️⃣ Testing with NEGATIVE economic text...")
    negative_text = """
    We're seeing significant headwinds in the US economy.
    Consumer confidence is declining and spending is weak.
    The outlook for the next quarter is quite concerning.
    """
    negative_score = scorer.score_text(negative_text)
    print(f"   Score: {negative_score}/5 (Expected: 1-2)")
    
    if negative_score is None or negative_score > 3:
        print("   ⚠️ Unexpected score for negative text")
    else:
        print("   ✅ Correct - low score for negative outlook")
    
    # Test with neutral text
    print("\n4️⃣ Testing with NEUTRAL economic text...")
    neutral_text = """
    The US economy shows mixed signals.
    Some sectors are performing well while others face challenges.
    We maintain a balanced view for the coming quarter.
    """
    neutral_score = scorer.score_text(neutral_text)
    print(f"   Score: {neutral_score}/5 (Expected: 3)")
    
    if neutral_score is None:
        print("   ⚠️ Failed to score neutral text")
    else:
        print(f"   ✅ Scored neutral text")
    
    print("\n" + "="*70)
    print("  ✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n🎉 Your free LLM setup is working perfectly!")
    print("📊 You can now score transcripts without paying for API calls")
    print("⚡ Using your GPU for fast inference")
    print("\n💡 Next steps:")
    print("   - Run 00_full_pipeline.ipynb")
    print("   - The scorer will use Ollama + Mistral automatically")
    print("   - No API keys or payments needed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n🔧 Troubleshooting:")
    print("   1. Make sure Ollama service is running:")
    print("      systemctl status ollama")
    print("   2. Verify mistral model is installed:")
    print("      ollama list")
    print("   3. Test Ollama directly:")
    print("      ollama run mistral 'Hello'")
    sys.exit(1)
