#!/usr/bin/env python3
"""
Download DOTA-trained YOLOv8 model
Run this to eliminate the DOTA model warning
"""
import sys
from pathlib import Path

def download_dota_model():
    print("="*70)
    print("DOWNLOADING DOTA-TRAINED MODEL")
    print("="*70)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("\nAttempting to download from HuggingFace...")
    
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        
        print("  Downloading keremberke/yolov8m-dota-v8...")
        model_path = hf_hub_download(
            repo_id="keremberke/yolov8m-dota-v8",
            filename="best.pt",
            cache_dir="./models/cache"
        )
        
        # Copy to expected location
        target_paths = [
            models_dir / "yolov8n-dota.pt",
            models_dir / "yolov8m-dota.pt"
        ]
        
        for target in target_paths:
            shutil.copy(model_path, target)
            print(f"  ✓ Saved to: {target}")
        
        print("\n✓ DOTA model downloaded successfully!")
        print("\nThe warning will no longer appear.")
        return True
        
    except ImportError:
        print("\n⚠️  huggingface-hub not installed")
        print("\nInstall it with:")
        print("  pip install huggingface-hub")
        return False
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nAlternative: The system will use YOLOv8 with satellite optimizations")
        print("This still works, just with slightly lower accuracy.")
        return False

if __name__ == "__main__":
    success = download_dota_model()
    sys.exit(0 if success else 1)
