#!/usr/bin/env python3
"""
Quick install script for vLLM optimized sentiment scorer
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show progress"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - Failed: {e}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   Fast Sentiment Scorer - vLLM Installation Script              ║
║   10-20x faster than standard inference                         ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check CUDA
    print("\n[1/4] Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("✗ CUDA not available. vLLM requires GPU.")
            sys.exit(1)
    except ImportError:
        print("✗ PyTorch not installed. Installing...")
        if not run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", 
                          "Installing PyTorch with CUDA 11.8"):
            print("\n✗ Failed to install PyTorch. Please install manually:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            sys.exit(1)
    
    # Install vLLM
    print("\n[2/4] Installing vLLM...")
    if not run_command("pip install vllm", "Installing vLLM"):
        print("\n⚠ If installation failed due to compilation errors, try:")
        print("  pip install vllm --no-build-isolation")
        
        retry = input("\nRetry with --no-build-isolation? (y/n): ").strip().lower()
        if retry in ['y', 'yes']:
            if not run_command("pip install vllm --no-build-isolation", "Installing vLLM (no isolation)"):
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Install dependencies
    print("\n[3/4] Installing other dependencies...")
    run_command("pip install -r requirements.txt", "Installing dependencies")
    
    # Verify installation
    print("\n[4/4] Verifying installation...")
    try:
        from vllm import LLM
        print("✓ vLLM installed successfully")
    except ImportError:
        print("✗ vLLM import failed")
        sys.exit(1)
    
    # Success
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    Installation Complete! ✓                      ║
╚══════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
