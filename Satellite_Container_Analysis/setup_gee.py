"""
Google Earth Engine Setup Script
Run this first to authenticate with Google Earth Engine
"""
import ee

def setup_earth_engine():
    """
    One-time setup for Google Earth Engine OAuth authentication
    """
    print("="*70)
    print("Google Earth Engine Authentication Setup")
    print("="*70)
    print("\nThis will open your browser for OAuth login.")
    print("Please sign in with your Google account.")
    print("\nNote: You only need to do this once per machine.")
    print("-"*70)
    
    try:
        # Run OAuth authentication
        ee.Authenticate()
        print("\n✓ Authentication successful!")
        
        # Test initialization
        print("\nTesting Earth Engine initialization...")
        ee.Initialize(project='algorithmictrading-414808')
        print("✓ Earth Engine is ready to use!")
        
        print("\n" + "="*70)
        print("Setup Complete!")
        print("="*70)
        print("\nYou can now use the GoogleEarthCollector:")
        print("""
from gee_collector import GoogleEarthCollector

# Initialize collector (no authentication needed after setup)
collector = GoogleEarthCollector()

# Download images for all ports (last 30 days, max 5 images per port)
df = collector.collect_port_images(days_back=30, max_images=5)

# Download for specific port
df = collector.collect_port_images(port_name='Shanghai', days_back=30)
        """)
        
    except Exception as e:
        print(f"\n✗ Authentication failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Check if earthengine-api is installed: pip install earthengine-api")
        print("3. Try running in terminal: earthengine authenticate")
        return False
    
    return True


if __name__ == "__main__":
    setup_earth_engine()
