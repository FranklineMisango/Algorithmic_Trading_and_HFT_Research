"""
Performance benchmarking for CPU vs GPU HJB implementations
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from numba import config
import os

# Configure Numba CUDA
config.CUDA_ENABLE_PYNVJITLINK = 1
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
os.environ['NUMBA_CUDA_DRIVER'] = '/usr/lib/x86_64-linux-gnu/libcuda.so'

# Import after configuration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from hjb_cpu_modelling import HJBMarketMaker
from hjb_gpu_modelling import HJBGPUMarketMaker

class PerformanceBenchmark:
    """Benchmark CPU vs GPU performance for HJB solving"""

    def __init__(self):
        self.results = {}

    def benchmark_grid_sizes(self, grid_sizes=[(50, 50, 5), (100, 100, 11), (200, 200, 21)]):
        """Benchmark performance across different grid sizes"""
        print("🚀 Starting HJB Performance Benchmark")
        print("=" * 50)

        for n_S, n_t, n_I in grid_sizes:
            print(f"\n📊 Testing grid size: {n_S}x{n_t}x{n_I}")

            # CPU Benchmark
            cpu_times = []
            for trial in range(3):  # 3 trials for averaging
                try:
                    model_cpu = HJBMarketMaker(
                        sigma=0.3, gamma=0.1, k=1.5, c=1.0, T=1.0,
                        I_max=n_I//2, S_min=80, S_max=120, dS=(120-80)/(n_S-1), dt=1.0/(n_t-1)
                    )
                    start = time.time()
                    model_cpu.solve_pde()
                    end = time.time()
                    cpu_times.append(end - start)
                except Exception as e:
                    print(f"CPU trial {trial+1} failed: {e}")
                    cpu_times.append(float('inf'))

            cpu_avg = np.mean([t for t in cpu_times if t != float('inf')])
            cpu_std = np.std([t for t in cpu_times if t != float('inf')])

            # GPU Benchmark
            gpu_times = []
            for trial in range(3):
                try:
                    model_gpu = HJBGPUMarketMaker(
                        sigma=0.3, gamma=0.1, k=1.5, c=1.0, T=1.0,
                        S_min=80.0, S_max=120.0, n_S=n_S, n_t=n_t, I_max=n_I//2
                    )
                    start = time.time()
                    model_gpu.solve_pde()
                    end = time.time()
                    gpu_times.append(end - start)
                except Exception as e:
                    print(f"GPU trial {trial+1} failed: {e}")
                    gpu_times.append(float('inf'))

            gpu_avg = np.mean([t for t in gpu_times if t != float('inf')])
            gpu_std = np.std([t for t in gpu_times if t != float('inf')])

            # Calculate speedup
            if cpu_avg > 0 and gpu_avg > 0:
                speedup = cpu_avg / gpu_avg
            else:
                speedup = 0

            self.results[f"{n_S}x{n_t}x{n_I}"] = {
                'cpu_time': cpu_avg,
                'cpu_std': cpu_std,
                'gpu_time': gpu_avg,
                'gpu_std': gpu_std,
                'speedup': speedup
            }

            print(".2f")
            print(".2f")
            print(".1f")

        return self.results

    def plot_results(self, save_path=None):
        """Plot performance comparison results"""
        if not self.results:
            print("No results to plot. Run benchmark_grid_sizes() first.")
            return

        grid_sizes = list(self.results.keys())
        cpu_times = [self.results[k]['cpu_time'] for k in grid_sizes]
        gpu_times = [self.results[k]['gpu_time'] for k in grid_sizes]
        speedups = [self.results[k]['speedup'] for k in grid_sizes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Time comparison
        x = np.arange(len(grid_sizes))
        width = 0.35

        ax1.bar(x - width/2, cpu_times, width, label='CPU', alpha=0.8)
        ax1.bar(x + width/2, gpu_times, width, label='GPU', alpha=0.8)
        ax1.set_xlabel('Grid Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('CPU vs GPU Solve Time')
        ax1.set_xticks(x)
        ax1.set_xticklabels(grid_sizes, rotation=45)
        ax1.legend()
        ax1.set_yscale('log')

        # Speedup
        ax2.bar(grid_sizes, speedups, color='green', alpha=0.7)
        ax2.set_xlabel('Grid Size')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('GPU Speedup vs CPU')
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance plot saved to {save_path}")

        plt.show()

    def print_summary(self):
        """Print performance summary"""
        if not self.results:
            print("No results available. Run benchmark_grid_sizes() first.")
            return

        print("\n🎯 PERFORMANCE SUMMARY")
        print("=" * 60)
        print("<12")
        print("-" * 60)

        for grid_size, data in self.results.items():
            cpu_time = data['cpu_time']
            gpu_time = data['gpu_time']
            speedup = data['speedup']

            if cpu_time < float('inf') and gpu_time < float('inf'):
                print("<12")
            else:
                print("<12")

        # Overall statistics
        valid_speedups = [data['speedup'] for data in self.results.values() if data['speedup'] > 0]
        if valid_speedups:
            avg_speedup = np.mean(valid_speedups)
            max_speedup = np.max(valid_speedups)
            print("-" * 60)
            print("<12")

def main():
    """Run comprehensive performance benchmark"""
    benchmark = PerformanceBenchmark()

    # Test different grid sizes
    grid_sizes = [(50, 50, 5), (100, 100, 11), (150, 150, 15), (200, 200, 21)]

    results = benchmark.benchmark_grid_sizes(grid_sizes)
    benchmark.print_summary()
    benchmark.plot_results(save_path='hjb_performance_benchmark.png')

    return results

if __name__ == "__main__":
    main()