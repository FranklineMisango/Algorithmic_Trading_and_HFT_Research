"""
Improved Satellite Data Collector
Fixes: Resolution, cloud filtering, data quality
"""
import ee
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path

class ImprovedSatelliteCollector:
    """Enhanced satellite imagery collection"""
    
    def __init__(self, ports):
        self.ports = ports
        
    def get_port_imagery(self, port, start_date, end_date, max_cloud=5):
        """Get high-quality satellite imagery with strict cloud filtering"""
        lon, lat = port['lon'], port['lat']
        buffer = 0.05
        roi = ee.Geometry.Rectangle([lon - buffer, lat - buffer, lon + buffer, lat + buffer])
        
        # Sentinel-2 with strict cloud filtering
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(roi)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
                      .filter(ee.Filter.lt('CLOUD_COVERAGE_ASSESSMENT', max_cloud))
                      .select(['B4', 'B3', 'B2']))
        
        return collection, roi
    
    def download_image(self, image, roi, port_name, date_str, download_dir='data/images'):
        """Download high-resolution satellite image"""
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        
        # Higher resolution for better detection
        url = image.getThumbURL({
            'region': roi,
            'dimensions': 4096,  # Increased from 1024
            'format': 'png'  # Better quality than jpg
        })
        
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            filepath = Path(download_dir) / f"{port_name}_{date_str}.png"
            img.save(filepath)
            return filepath
        return None
    
    def collect_port_images(self, port, start_date, end_date, max_images=50):
        """Collect multiple high-quality images"""
        collection, roi = self.get_port_imagery(port, start_date, end_date)
        
        image_list = collection.toList(max_images)
        count = image_list.size().getInfo()
        
        print(f"Found {count} high-quality images for {port['name']} (<5% clouds)")
        
        downloaded = []
        for i in range(min(count, max_images)):
            try:
                image = ee.Image(image_list.get(i))
                date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                
                filepath = self.download_image(image, roi, port['name'], date)
                if filepath:
                    downloaded.append({
                        'port': port['name'],
                        'filepath': filepath,
                        'date': date
                    })
                    print(f"  ✓ Downloaded {date}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
        
        return downloaded
