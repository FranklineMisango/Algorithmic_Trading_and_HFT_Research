"""
Improved Satellite Data Collector
Now uses GoogleEarthCollector from gee_collector.py
"""
from gee_collector import GoogleEarthCollector
from datetime import datetime, timedelta
import pandas as pd

# Backward compatibility wrapper
class ImprovedSatelliteCollector:
    """Wrapper for GoogleEarthCollector with legacy interface"""
    
    def __init__(self, ports, config_path='config.json'):
        self.ports = ports
        self.gee_collector = GoogleEarthCollector(config_path=config_path, authenticate=False)
    
    def collect_port_images(self, port, start_date, end_date, max_images=50, max_cloud=10):
        """Collect images using GoogleEarthCollector
        
        Args:
            port: Port dict with 'name', 'lat', 'lon'
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            max_images: Maximum number of images
            max_cloud: Maximum cloud cover percentage
            
        Returns:
            List of dicts with 'port', 'filepath', 'date' (legacy format)
        """
        # Calculate days_back from date range
        end = datetime.strptime(end_date, '%Y-%m-%d')
        start = datetime.strptime(start_date, '%Y-%m-%d')
        days_back = (end - start).days
        
        # Use GoogleEarthCollector
        df = self.gee_collector.collect_port_images(
            port_name=port['name'],
            days_back=days_back,
            max_images=max_images,
            max_cloud=max_cloud,
            source='sentinel2'
        )
        
        # Convert DataFrame to legacy list format
        if df.empty:
            return []
        
        return [{
            'port': row['port'],
            'filepath': row['filepath'],
            'date': row['date']
        } for _, row in df.iterrows()]
