import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import pandas as pd

class ContainerDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        # Container-like object classes in COCO dataset
        self.container_classes = [2, 5, 7, 3]  # car, bus, truck, motorcycle
        
    def detect_objects(self, image_path, conf_threshold=0.5):
        """Detect objects in satellite image"""
        results = self.model(str(image_path))
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) in self.container_classes and box.conf > conf_threshold:
                        detections.append({
                            'bbox': box.xyxy.cpu().numpy()[0],
                            'confidence': float(box.conf),
                            'class': int(box.cls),
                            'class_name': result.names[int(box.cls)]
                        })
        
        return detections
    
    def count_containers(self, image_path):
        """Count container-like objects in image"""
        detections = self.detect_objects(image_path)
        return len(detections), detections
    
    def process_port_images(self, image_dir, port_name):
        """Process all images for a port and count containers"""
        image_dir = Path(image_dir)
        results = []
        
        for img_path in image_dir.glob('*.jp*g'):
            try:
                count, detections = self.count_containers(img_path)
                results.append({
                    'port': port_name,
                    'image_file': img_path.name,
                    'container_count': count,
                    'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
                    'timestamp': img_path.stat().st_mtime
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        return pd.DataFrame(results)

def analyze_container_trends(container_data):
    """Analyze container count trends"""
    if container_data.empty:
        return pd.DataFrame()
    
    # Ensure datetime column exists
    if 'timestamp' in container_data.columns:
        container_data['datetime'] = pd.to_datetime(container_data['timestamp'], unit='s')
    elif 'datetime' not in container_data.columns:
        return container_data
    
    # Calculate rolling statistics
    container_data = container_data.sort_values('datetime')
    container_data['ma_7'] = container_data.groupby('port')['container_count'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    container_data['ma_30'] = container_data.groupby('port')['container_count'].transform(lambda x: x.rolling(30, min_periods=1).mean())
    
    return container_data