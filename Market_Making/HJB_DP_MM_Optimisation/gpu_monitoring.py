import subprocess
import time
from tabulate import tabulate
import numpy as np

def monitor_gpu_utilization():
    """
    Monitor and return GPU utilization metrics.
    
    Returns:
        dict: Dictionary containing GPU utilization metrics or None if GPU monitoring failed
    """
    try:
        import pynvml
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                print("No NVIDIA GPUs detected with pynvml")
                raise ImportError("No GPUs detected")
            
            gpu_metrics = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_metrics.append({
                    'index': i,
                    'name': name.decode('utf-8') if isinstance(name, bytes) else name,
                    'gpu_util': util.gpu,
                    'memory_util': (memory.used / memory.total) * 100,
                    'memory_used_mb': memory.used / 1024 / 1024,
                    'memory_total_mb': memory.total / 1024 / 1024,
                    'temperature': temp
                })
            
            pynvml.nvmlShutdown()
            return {
                'method': 'pynvml',
                'gpu_count': device_count,
                'metrics': gpu_metrics
            }
            
        except Exception as e:
            print(f"pynvml error: {e}, falling back to nvidia-smi")
            pynvml.nvmlShutdown()
            raise ImportError("pynvml failed")
    
    except ImportError:
        # Fallback to nvidia-smi command
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpu_metrics = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpu_metrics.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'gpu_util': float(parts[2]),
                            'memory_used_mb': float(parts[3]),
                            'memory_total_mb': float(parts[4]),
                            'memory_util': (float(parts[3]) / float(parts[4])) * 100 if float(parts[4]) > 0 else 0,
                            'temperature': float(parts[5])
                        })
            
            return {
                'method': 'nvidia-smi',
                'gpu_count': len(gpu_metrics),
                'metrics': gpu_metrics
            }
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"nvidia-smi error: {e}")
            
            # Last resort - try GPUtil if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                
                gpu_metrics = []
                for i, gpu in enumerate(gpus):
                    gpu_metrics.append({
                        'index': i,
                        'name': gpu.name,
                        'gpu_util': gpu.load * 100,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_util': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature
                    })
                
                return {
                    'method': 'GPUtil',
                    'gpu_count': len(gpu_metrics),
                    'metrics': gpu_metrics
                }
                
            except ImportError:
                print("No GPU monitoring methods available")
                return None
    
    return None


def print_gpu_metrics():
    """
    Print formatted GPU utilization metrics to console
    """
    metrics = monitor_gpu_utilization()
    
    if metrics is None:
        print("GPU metrics unavailable - no monitoring method found")
        return
    
    if metrics['gpu_count'] == 0:
        print("No GPUs detected")
        return
    
    print("\n===== GPU Utilization Report =====")
    print(f"Method: {metrics['method']}")
    print(f"GPU Count: {metrics['gpu_count']}")
    
    headers = ["Index", "Name", "GPU Util%", "Memory Util%", "Memory Used (MB)", "Total Memory (MB)", "Temp (°C)"]
    table_data = []
    
    for gpu in metrics['metrics']:
        table_data.append([
            gpu['index'],
            gpu['name'],
            f"{gpu['gpu_util']:.1f}%",
            f"{gpu['memory_util']:.1f}%",
            f"{gpu['memory_used_mb']:.0f}",
            f"{gpu['memory_total_mb']:.0f}",
            f"{gpu['temperature']:.0f}"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("==================================\n")


def monitor_gpu_continuously(interval=5, duration=None):
    """
    Monitor GPU utilization continuously at specified intervals
    
    Args:
        interval: Time between updates in seconds
        duration: Total monitoring duration in seconds (None for indefinite)
    """
    start_time = time.time()
    iteration = 0
    
    try:
        while True:
            iteration += 1
            current_time = time.time() - start_time
            
            print(f"\nMonitoring iteration {iteration}, elapsed time: {current_time:.1f}s")
            print_gpu_metrics()
            
            # Check if we've reached the desired duration
            if duration is not None and current_time >= duration:
                break
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nGPU monitoring stopped by user")


if __name__ == "__main__":
    # Example usage
    print("GPU Monitoring Utility")
    print("----------------------")
    print("Single GPU metrics report:")
    print_gpu_metrics()
    
    print("\nStarting continuous monitoring (Ctrl+C to stop)...")
    print("Polling every 5 seconds")
    monitor_gpu_continuously(interval=5)