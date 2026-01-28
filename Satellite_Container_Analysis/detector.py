"""
Improved Container Detector with DOTA model support
Fixes: Model mismatch, preprocessing, multi-scale detection
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch

class ImprovedContainerDetector:
    """Satellite-optimized container detection"""
    
    def __init__(self, model_path=None):
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load DOTA-trained model or fallback"""
        # Try DOTA models in order of preference
        model_paths = [
            model_path,
            'models/yolov8n-dota.pt',
            'models/yolov8m-dota.pt',
            'yolov8n.pt'  # Fallback
        ]
        
        for path in model_paths:
            if path and Path(path).exists():
                print(f"✓ Loaded model: {path}")
                return YOLO(path)
        
        # Download DOTA model from HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading DOTA-trained model from HuggingFace...")
            
            model_file = hf_hub_download(
                repo_id="keremberke/yolov8m-dota-v8",
                filename="best.pt",
                cache_dir="./models"
            )
            print(f"✓ Downloaded DOTA model: {model_file}")
            return YOLO(model_file)
        except Exception as e:
            print(f"⚠️  Could not download DOTA model: {e}")
            print("Using YOLOv8 with satellite optimizations")
            return YOLO('yolov8n.pt')
    
    def preprocess_satellite_image(self, img):
        """Apply satellite-specific preprocessing"""
        # 1. CLAHE contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # 2. Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        
        # 3. Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        return img
    
    def detect_containers(self, image_path, conf_threshold=0.15):
        """Multi-scale detection with post-processing"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.preprocess_satellite_image(img)
        
        # Multi-scale inference
        results = self.model(
            img,
            conf=conf_threshold,
            imgsz=[1024, 2048],  # Multiple scales
            augment=True,  # Test-time augmentation
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    
                    # Filter by relevant classes (DOTA or COCO)
                    class_name = self.model.names[cls]
                    if self._is_port_object(class_name):
                        xywh = box.xywh[0].cpu().numpy()
                        w, h = xywh[2], xywh[3]
                        aspect_ratio = w / h if h > 0 else 0
                        
                        # Filter by aspect ratio (containers are 1:2 to 2:1)
                        if 0.5 <= aspect_ratio <= 2.0:
                            detections.append({
                                'bbox': box.xyxy.cpu().numpy(),
                                'confidence': conf,
                                'class': cls,
                                'class_name': class_name,
                                'aspect_ratio': aspect_ratio
                            })
        
        return len(detections), detections
    
    def _is_port_object(self, class_name):
        """Check if class is relevant to ports"""
        port_classes = [
            'ship', 'harbor', 'large-vehicle', 'small-vehicle', 
            'storage-tank', 'truck', 'car', 'boat', 'bus'
        ]
        return any(pc in class_name.lower() for pc in port_classes)
    
    def batch_detect(self, image_dir):
        """Process multiple images"""
        results = []
        for img_path in Path(image_dir).glob('*.png'):
            count, detections = self.detect_containers(img_path)
            results.append({
                'image': img_path.name,
                'count': count,
                'detections': detections
            })
        return results
