#!/usr/bin/env python3
"""
YOLO Training Script for xView Dataset
Optimized for overnight training on 16GB GPU
Run in tmux: python3 train_xview.py
"""

# Suppress TIFF warnings BEFORE importing any libraries
import os
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '1000000000'

import warnings
warnings.filterwarnings('ignore')

import sys
import logging

# Redirect OpenCV stderr messages
class TIFFWarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(x in msg.lower() for x in ['tiff', 'geotiff', 'tag', 'exif', 'unknown field'])

# Configure logging
logging.basicConfig(level=logging.ERROR)
for logger_name in ['ultralytics', 'torch', 'PIL']:
    logger = logging.getLogger(logger_name)
    logger.addFilter(TIFFWarningFilter())
    logger.setLevel(logging.ERROR)

from ultralytics import YOLO
import torch
from datetime import datetime

def main():
    print("="*70)
    print("YOLO xView Training - Satellite Container Detection")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # GPU setup
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected - training will be slow!")
        sys.exit(1)
    
    print()
    print("Training Configuration (Accuracy-Optimized):")
    print("  • Model: YOLO11s (small - 3x parameters vs nano)")
    print("  • Dataset: xView (satellite imagery)")
    print("  • Epochs: 300 (patience=100 for convergence)")
    print("  • Image size: 896x896 (multi-scale: 448-1344px)")
    print("  • Batch size: 3 (safe for 16GB GPU)")
    print("  • Augmentation: Multi-scale, Mixup, Mosaic (full training)")
    print("  • Device: GPU")
    print()
    
    # NOTE about warnings
    print("NOTE: You will see OpenCV TIFF warnings - this is normal.")
    print("      xView uses GeoTIFF format with GPS metadata.")
    print("      Images load correctly despite warnings.")
    print()
    print("="*70)
    print()
    
    try:
        # Initialize model
        print("Loading YOLO11s model...")
        model = YOLO("yolo11s.pt")
        print("✓ Model loaded (11.1M parameters - 3x more than nano)")
        print()
        
        # Training configuration
        data_path = "datasets/xview/xView.yaml"
        
        print("Starting training...")
        print("-"*70)
        
        results = model.train(
            data=data_path,
            epochs=300,
            imgsz=896,  # Reduced from 1024: multi_scale varies 0.5x-1.5x (448-1344px) to fit 16GB
            batch=3,  # Conservative: batch=4 uses 13.9GB but TaskAlignedAssigner OOMs during loss calc
            device=0,
            amp=True,
            patience=100,  # Increased from 50: allow more epochs without improvement before stopping
            save=True,
            save_period=10,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5.0,
            warmup_momentum=0.8,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.4,
            degrees=30.0,  # Increased from 15: containers appear at any angle in satellite imagery
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,  # Increased from 0: blend images for better generalization
            copy_paste=0.0,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            multi_scale=True,  # Train on multiple scales: better for varying container sizes
            iou=0.5,
            val=True,
            plots=True,
            verbose=True,
            close_mosaic=0,  # Keep mosaic augmentation throughout all epochs (was 10)
            cache=False,
            workers=4,
            project='runs/detect',
            name='train',
        )
        
        print()
        print("="*70)
        print("✓ Training completed successfully!")
        print("="*70)
        print()
        print("Model saved to:")
        print(f"  • Best model: runs/detect/train/weights/best.pt")
        print(f"  • Last model: runs/detect/train/weights/last.pt")
        print(f"  • Checkpoints: runs/detect/train/weights/epoch*.pt")
        print()
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
    except KeyboardInterrupt:
        print()
        print("="*70)
        print("⚠️  Training interrupted by user")
        print("="*70)
        print()
        print("Partial models may be saved in runs/detect/train/weights/")
        sys.exit(1)
        
    except Exception as e:
        print()
        print("="*70)
        print("✗ Training failed with error:")
        print("="*70)
        print(f"{type(e).__name__}: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
