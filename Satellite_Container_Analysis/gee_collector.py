"""
Google Earth Engine Satellite Image Collector
Uses OAuth authentication for high-quality satellite imagery
"""
import ee
import json
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


class GoogleEarthCollector:
    """Collect satellite imagery using Google Earth Engine"""
    
    def __init__(self, config_path='config.json', authenticate=False):
        """
        Initialize Google Earth Engine collector
        
        Args:
            config_path: Path to config.json
            authenticate: If True, run OAuth authentication flow
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.project_id = self.config['GoogleOAuth']['ProjectId']
        self.ports = self.config['ports']
        
        # Authenticate and initialize Earth Engine
        if authenticate:
            print("Starting Google Earth Engine authentication...")
            print("This will open your browser for OAuth login.")
            ee.Authenticate()
        
        try:
            ee.Initialize(project=self.project_id)
            print(f"✓ Google Earth Engine initialized with project: {self.project_id}")
        except Exception as e:
            print(f"✗ Failed to initialize Earth Engine: {e}")
            print("Run with authenticate=True or run 'earthengine authenticate' in terminal")
            raise
    
    def get_sentinel2_imagery(self, port, start_date, end_date, max_cloud=10):
        """
        Get Sentinel-2 imagery for a port location
        
        Args:
            port: Port dictionary with lat/lon
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_cloud: Maximum cloud cover percentage (default: 10%)
        
        Returns:
            ImageCollection, Region of Interest
        """
        lon, lat = port['lon'], port['lat']
        
        # Create region of interest (5km buffer around port)
        buffer = 0.045  # ~5km at equator
        roi = ee.Geometry.Rectangle([
            lon - buffer, 
            lat - buffer, 
            lon + buffer, 
            lat + buffer
        ])
        
        # Query Sentinel-2 Surface Reflectance
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(roi)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
                      .select(['B4', 'B3', 'B2'])  # RGB bands
                      .sort('system:time_start', False))  # Most recent first
        
        return collection, roi
    
    def get_landsat_imagery(self, port, start_date, end_date, max_cloud=10):
        """
        Get Landsat 8/9 imagery for a port location (30m resolution)
        
        Args:
            port: Port dictionary with lat/lon
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_cloud: Maximum cloud cover percentage
        
        Returns:
            ImageCollection, Region of Interest
        """
        lon, lat = port['lon'], port['lat']
        buffer = 0.045
        roi = ee.Geometry.Rectangle([lon - buffer, lat - buffer, lon + buffer, lat + buffer])
        
        # Landsat 8/9 Collection 2 Tier 1 Level 2
        collection = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                      .filterBounds(roi)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUD_COVER', max_cloud))
                      .select(['SR_B4', 'SR_B3', 'SR_B2'])  # RGB bands
                      .sort('system:time_start', False))
        
        return collection, roi
    
    def download_image(self, image, roi, filename, download_dir='data/images', resolution=10):
        """
        Download satellite image from Earth Engine
        
        Args:
            image: Earth Engine Image
            roi: Region of interest
            filename: Output filename
            download_dir: Directory to save images
            resolution: Spatial resolution in meters (10m for Sentinel-2, 30m for Landsat)
        
        Returns:
            Path to downloaded image or None
        """
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Get download URL with high resolution
            url = image.getThumbURL({
                'region': roi,
                'dimensions': 2048,  # High resolution output
                'format': 'png',
                'min': 0,
                'max': 3000  # Adjust for Sentinel-2 surface reflectance
            })
            
            # Download image
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                filepath = Path(download_dir) / f"{filename}.png"
                img.save(filepath, quality=95)
                return filepath
            else:
                print(f"✗ Download failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"✗ Error downloading image: {e}")
            return None
    
    def collect_port_images(self, port_name=None, days_back=30, max_images=10, 
                           max_cloud=10, source='sentinel2'):
        """
        Collect satellite images for port(s)
        
        Args:
            port_name: Specific port name or None for all ports
            days_back: Number of days to look back
            max_images: Maximum images per port
            max_cloud: Maximum cloud cover percentage
            source: 'sentinel2' or 'landsat'
        
        Returns:
            DataFrame with download information
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Filter ports
        ports_to_process = [p for p in self.ports if port_name is None or p['name'] == port_name]
        
        if not ports_to_process:
            print(f"✗ Port '{port_name}' not found in config")
            return pd.DataFrame()
        
        all_downloads = []
        
        for port in ports_to_process:
            print(f"\n{'='*60}")
            print(f"Processing: {port['name']} ({port['country']})")
            print(f"Coordinates: {port['lat']}, {port['lon']}")
            print(f"Date range: {start_str} to {end_str}")
            print(f"Max cloud cover: {max_cloud}%")
            print(f"{'='*60}")
            
            # Get imagery based on source
            if source == 'sentinel2':
                collection, roi = self.get_sentinel2_imagery(port, start_str, end_str, max_cloud)
                resolution = 10
            elif source == 'landsat':
                collection, roi = self.get_landsat_imagery(port, start_str, end_str, max_cloud)
                resolution = 30
            else:
                print(f"✗ Unknown source: {source}")
                continue
            
            # Get available images
            image_list = collection.toList(max_images)
            count = image_list.size().getInfo()
            
            print(f"Found {count} images with <{max_cloud}% cloud cover")
            
            if count == 0:
                print(f"⚠ No suitable images found for {port['name']}")
                continue
            
            # Download images
            for i in range(min(count, max_images)):
                try:
                    image = ee.Image(image_list.get(i))
                    
                    # Get image metadata
                    props = image.getInfo()['properties']
                    timestamp = props.get('system:time_start')
                    date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
                    cloud_cover = props.get('CLOUDY_PIXEL_PERCENTAGE', props.get('CLOUD_COVER', 0))
                    
                    # Create filename
                    filename = f"{port['name'].replace(' ', '_')}_{date}_{source}"
                    
                    print(f"  Downloading {i+1}/{min(count, max_images)}: {date} (cloud: {cloud_cover:.1f}%)")
                    
                    filepath = self.download_image(image, roi, filename, resolution=resolution)
                    
                    if filepath:
                        all_downloads.append({
                            'port': port['name'],
                            'country': port['country'],
                            'lat': port['lat'],
                            'lon': port['lon'],
                            'date': date,
                            'cloud_cover': cloud_cover,
                            'source': source,
                            'resolution_m': resolution,
                            'filepath': str(filepath),
                            'ticker': port.get('ticker', '')
                        })
                        print(f"  ✓ Saved: {filepath.name}")
                    
                except Exception as e:
                    print(f"  ✗ Failed to download image {i+1}: {e}")
                    continue
            
            print(f"✓ Downloaded {len([d for d in all_downloads if d['port'] == port['name']])} images for {port['name']}")
        
        # Create DataFrame
        df = pd.DataFrame(all_downloads)
        
        if not df.empty:
            # Save metadata
            metadata_path = Path('data/results/image_metadata.csv')
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(metadata_path, index=False)
            print(f"\n✓ Saved metadata to {metadata_path}")
            print(f"✓ Total images downloaded: {len(df)}")
        
        return df


def authenticate_gee():
    """
    Run Google Earth Engine authentication flow
    This only needs to be done once per machine
    """
    print("Starting Google Earth Engine OAuth authentication...")
    print("A browser window will open for you to log in.")
    ee.Authenticate()
    print("✓ Authentication complete!")
    print("You can now use the GoogleEarthCollector class.")


if __name__ == "__main__":
    import sys
    
    # Check if user wants to authenticate
    if len(sys.argv) > 1 and sys.argv[1] == 'auth':
        authenticate_gee()
    else:
        # Example usage
        print("Google Earth Engine Collector")
        print("-" * 60)
        print("\nTo authenticate (first time only):")
        print("  python gee_collector.py auth")
        print("\nExample usage in code:")
        print("""
from gee_collector import GoogleEarthCollector

# Initialize (authenticate=True only first time)
collector = GoogleEarthCollector(authenticate=False)

# Collect images for all ports
df = collector.collect_port_images(days_back=30, max_images=5)

# Or collect for specific port
df = collector.collect_port_images(port_name='Shanghai', days_back=30)
        """)
