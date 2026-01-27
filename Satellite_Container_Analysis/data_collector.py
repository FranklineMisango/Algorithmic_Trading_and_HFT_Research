import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

class CopernicusDataCollector:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.username = self.config['ESACopernicus']['Username']
        self.password = self.config['ESACopernicus']['Password']
        self.base_url = 'https://apihub.copernicus.eu/apihub'
        
    def search_products(self, port, start_date, end_date, max_cloud=20):
        """Search for Sentinel-2 products over a port area"""
        # Create bounding box (0.1 degree buffer around port)
        lat, lon = port['lat'], port['lon']
        bbox = f"{lon-0.1},{lat-0.1},{lon+0.1},{lat+0.1}"
        
        query_params = {
            'q': f'platformname:Sentinel-2 AND cloudcoverpercentage:[0 TO {max_cloud}]',
            'rows': 100,
            'start': 0,
            'format': 'json'
        }
        
        response = requests.get(
            f"{self.base_url}/search",
            params=query_params,
            auth=(self.username, self.password)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error searching products: {response.status_code}")
            return None
    
    def download_product(self, product_id, download_dir='data/images'):
        """Download a specific product"""
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        
        url = f"{self.base_url}/odata/v1/Products('{product_id}')/$value"
        response = requests.get(url, auth=(self.username, self.password), stream=True)
        
        if response.status_code == 200:
            filename = f"{product_id}.zip"
            filepath = Path(download_dir) / filename
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return filepath
        else:
            print(f"Error downloading product: {response.status_code}")
            return None

def collect_port_data(ports, days_back=30):
    """Collect satellite data for all ports"""
    collector = CopernicusDataCollector()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    all_products = []
    
    for port in ports:
        print(f"Searching products for {port['name']}...")
        products = collector.search_products(port, start_date, end_date)
        
        if products and 'feed' in products:
            for entry in products['feed'].get('entry', []):
                all_products.append({
                    'port': port['name'],
                    'product_id': entry['id'],
                    'title': entry['title'],
                    'date': entry['date'][0]['content'],
                    'cloud_cover': entry.get('double', [{}])[0].get('content', 0)
                })
    
    return pd.DataFrame(all_products)