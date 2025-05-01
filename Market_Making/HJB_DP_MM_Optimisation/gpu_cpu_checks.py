import subprocess
import sys
import os
from numba import config

config.CUDA_ENABLE_PYNVJITLINK = 1
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

def check_cuda():
    """Verify CUDA installation and compatibility"""
    try:
        # Check if nvidia-smi is available
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.PIPE)
        print("✅ NVIDIA GPU detected")
        
        # Attempt to import CUDA-enabled packages
        import numba
        from numba import cuda
        import cupy as cp
        
        # Test CUDA functionality
        @cuda.jit
        def test_kernel(x):
            i = cuda.grid(1)
            if i < x.shape[0]:
                x[i] *= 2
                
        # Simple array test
        import numpy as np
        test_array = np.ones(10, dtype=np.float32)
        d_test = cuda.to_device(test_array)
        test_kernel[1, 10](d_test)
        result = d_test.copy_to_host()
        
        if np.all(result == 2.0):
            print("✅ CUDA test successful")
            print(f"   Using CUDA {cp.cuda.runtime.runtimeGetVersion()/1000:.1f}")
            print(f"   Using NumPy {np.__version__}")
            print(f"   Using Numba {numba.__version__}")
            print(f"   Using CuPy {cp.__version__}")
            return True
        else:
            print("❌ CUDA test failed")
            return False
    
    except (subprocess.CalledProcessError, ImportError, Exception) as e:
        print(f"❌ CUDA configuration error: {str(e)}")
        return False

def validate_environment():
    """Validate the Python environment has all required packages"""
    required_packages = [
        "numpy", "pandas", "scipy", "matplotlib", "plotly", 
        "dash", "ccxt", "tabulate", "websocket", "numba"
    ]
    
    missing = []
    version_issues = []
    
    for package in required_packages:
        try:
            module = __import__(package)
            if package == "numpy" and not (1.22 <= float(module.__version__.split('.')[0] + '.' + module.__version__.split('.')[1]) < 1.29):
                version_issues.append(f"numpy=={module.__version__} (should be >=1.22.4 and <1.29.0)")
        except ImportError:
            missing.append(package)
    
    if missing or version_issues:
        print("❌ Environment validation failed")
        if missing:
            print(f"   Missing packages: {', '.join(missing)}")
        if version_issues:
            print(f"   Version issues: {', '.join(version_issues)}")
        return False
    
    print("✅ All required packages are installed")
    return True

if __name__ == "__main__":
    print("⭐ Validating HJB Market Making environment...")
    env_valid = validate_environment()
    
    if env_valid:
        cuda_valid = check_cuda()
        if cuda_valid:
            print("\n🚀 Environment is ready for HJB Market Making with GPU acceleration!")
        else:
            print("\n⚠️ Environment is ready for CPU-only mode (no GPU acceleration)")
    else:
        print("\n❌ Please fix environment issues before continuing")
        print("   Suggested fix: conda env create -f environment.yml")