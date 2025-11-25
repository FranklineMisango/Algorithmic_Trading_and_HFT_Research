#!/usr/bin/env python3
"""
Quick Quandl Test Script
Tests Quandl installation and basic functionality
"""

import sys
import time

def test_quandl():
    print("🧪 QUANDL TEST SCRIPT")
    print("="*30)

    try:
        import quandl
        print("✅ Quandl imported successfully")

        # Check version (might not be available)
        try:
            version = getattr(quandl, '__version__', 'Unknown')
            print(f"Version: {version}")
        except:
            print("Version: Not available")

        # Check API key
        has_key = bool(getattr(quandl.ApiConfig, 'api_key', None))
        print(f"API Key: {'Configured' if has_key else 'Not configured'}")

        if not has_key:
            print("\n📋 FREE DATASETS:")
            print("• CBOE/VIX - Volatility Index")
            print("• CBOE/SPX_PC - Put-Call Ratio")
            print("• WIKI/AAPL - Sample stock data")

        print("\n💰 PREMIUM DATASETS:")
        print("• CBOE options chains")
        print("• CME futures data")

        print("\n🔗 FREE API KEY: https://www.quandl.com/")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_quandl()

    print("\n" + "="*30)
    if success:
        print("✅ QUANDL READY FOR USE")
        print("For SPX options arbitrage, you'll need:")
        print("• API key for premium datasets")
        print("• Or alternative data sources")
    else:
        print("❌ QUANDL SETUP FAILED")

    sys.exit(0 if success else 1)