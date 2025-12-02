"""
Practical deployment analysis for HJB market making system
Addresses critique requirements for real-world deployment considerations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import psutil
import threading
import warnings

class DeploymentAnalyzer:
    """Analyze practical deployment considerations for HJB market making"""

    def __init__(self):
        self.deployment_metrics = {}

    def latency_analysis(self, grid_sizes=[(50, 50, 11), (100, 100, 21), (200, 200, 41)]):
        """
        Analyze solve latency for different grid sizes and requirements
        """
        print("Analyzing solve latency for different grid sizes...")

        latency_results = []

        for n_S, n_t, n_I in grid_sizes:
            print(f"Testing grid: {n_S}x{n_t}x{n_I}")

            # CPU implementation
            from hjb_cpu_modelling import HJBMarketMaker

            cpu_times = []
            for trial in range(5):  # Multiple trials for statistics
                model_cpu = HJBMarketMaker(
                    sigma=0.3, gamma=0.1, k=1.5, c=1.0, T=1.0,
                    I_max=(n_I-1)//2, S_min=80, S_max=120,
                    dS=(120-80)/(n_S-1), dt=1.0/(n_t-1)
                )

                start = time.perf_counter()
                model_cpu.solve_pde()
                end = time.perf_counter()
                cpu_times.append((end - start) * 1000)  # Convert to milliseconds

            cpu_avg = np.mean(cpu_times)
            cpu_std = np.std(cpu_times)
            cpu_95p = np.percentile(cpu_times, 95)

            # GPU implementation (if available)
            gpu_times = []
            try:
                from hjb_gpu_modelling import HJBGPUMarketMaker

                for trial in range(5):
                    model_gpu = HJBGPUMarketMaker(
                        sigma=0.3, gamma=0.1, k=1.5, c=1.0, T=1.0,
                        S_min=80.0, S_max=120.0, n_S=n_S, n_t=n_t, I_max=(n_I-1)//2
                    )

                    start = time.perf_counter()
                    model_gpu.solve_pde()
                    end = time.perf_counter()
                    gpu_times.append((end - start) * 1000)

                gpu_avg = np.mean(gpu_times)
                gpu_std = np.std(gpu_times)
                gpu_95p = np.percentile(gpu_times, 95)
                speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0

            except Exception as e:
                gpu_avg = gpu_std = gpu_95p = speedup = np.nan
                print(f"GPU test failed: {e}")

            latency_results.append({
                'grid_size': f"{n_S}x{n_t}x{n_I}",
                'n_S': n_S, 'n_t': n_t, 'n_I': n_I,
                'cpu_avg_ms': cpu_avg,
                'cpu_std_ms': cpu_std,
                'cpu_95p_ms': cpu_95p,
                'gpu_avg_ms': gpu_avg,
                'gpu_std_ms': gpu_std,
                'gpu_95p_ms': gpu_95p,
                'speedup': speedup
            })

        return pd.DataFrame(latency_results)

    def memory_usage_analysis(self, grid_sizes=[(50, 50, 11), (100, 100, 21), (200, 200, 41)]):
        """
        Analyze memory requirements for different grid sizes
        """
        print("Analyzing memory usage for different grid sizes...")

        memory_results = []

        for n_S, n_t, n_I in grid_sizes:
            print(f"Testing memory for grid: {n_S}x{n_t}x{n_I}")

            # Calculate theoretical memory requirements
            theta_size = n_t * n_S * n_I * 8  # 8 bytes per float64
            theta_size_mb = theta_size / (1024**2)

            # Additional arrays (grids, etc.)
            total_size_mb = theta_size_mb * 2  # Rough estimate

            # GPU memory (if available)
            gpu_memory_mb = np.nan
            try:
                import cupy as cp
                # CuPy arrays
                gpu_size = n_t * n_S * n_I * 4  # float32 on GPU
                gpu_memory_mb = gpu_size / (1024**2)
            except:
                pass

            memory_results.append({
                'grid_size': f"{n_S}x{n_t}x{n_I}",
                'cpu_memory_mb': total_size_mb,
                'gpu_memory_mb': gpu_memory_mb,
                'theta_array_mb': theta_size_mb
            })

        return pd.DataFrame(memory_results)

    def real_time_requirements_analysis(self):
        """
        Analyze requirements for real-time market making
        """
        print("Analyzing real-time requirements...")

        # Typical market making requirements
        requirements = {
            'crypto_hft': {
                'solve_frequency': 'Every 1-10 seconds',
                'max_solve_time': 100,  # ms
                'update_frequency': 'Every tick (microseconds)',
                'latency_requirement': 10,  # microseconds for co-located
                'data_frequency': 'Millisecond-level updates'
            },
            'traditional_hft': {
                'solve_frequency': 'Every 1-60 seconds',
                'max_solve_time': 500,  # ms
                'update_frequency': 'Every millisecond',
                'latency_requirement': 100,  # microseconds
                'data_frequency': 'Microsecond-level updates'
            },
            'retail_mm': {
                'solve_frequency': 'Every 1-5 minutes',
                'max_solve_time': 2000,  # ms
                'update_frequency': 'Every second',
                'latency_requirement': 1000,  # microseconds (1ms)
                'data_frequency': 'Second-level updates'
            }
        }

        return requirements

    def exchange_integration_analysis(self):
        """
        Analyze exchange-specific integration challenges
        """
        print("Analyzing exchange integration challenges...")

        exchanges = {
            'binance': {
                'api_rate_limit': '1200 requests/weight per minute',
                'order_types': ['LIMIT', 'MARKET', 'STOP_LOSS', 'OCO'],
                'fees': {'maker': 0.001, 'taker': 0.001},
                'latency': '~100-500ms typical',
                'websocket': 'Yes, real-time updates',
                'position_limits': 'Varies by asset',
                'maintenance_windows': 'Weekly maintenance'
            },
            'coinbase_pro': {
                'api_rate_limit': '10 requests/second (public), 5/s (private)',
                'order_types': ['LIMIT', 'MARKET', 'STOP'],
                'fees': {'maker': 0.005, 'taker': 0.005},
                'latency': '~200-800ms typical',
                'websocket': 'Yes, real-time updates',
                'position_limits': 'Varies by asset',
                'maintenance_windows': 'Scheduled maintenance'
            },
            'kraken': {
                'api_rate_limit': '15 requests/minute (starter), higher tiers available',
                'order_types': ['LIMIT', 'MARKET', 'STOP_LOSS', 'TRAILING_STOP'],
                'fees': {'maker': 0.0016, 'taker': 0.0026},
                'latency': '~300-1000ms typical',
                'websocket': 'Yes, real-time updates',
                'position_limits': 'Varies by asset',
                'maintenance_windows': 'Scheduled maintenance'
            }
        }

        return exchanges

    def risk_management_framework(self):
        """
        Define comprehensive risk management framework
        """
        print("Defining risk management framework...")

        risk_framework = {
            'inventory_limits': {
                'max_inventory': '5-10% of ADV (Average Daily Volume)',
                'rebalancing_threshold': '2-3% of max inventory',
                'emergency_stop': '10% of max inventory'
            },
            'pnl_limits': {
                'daily_pnl_limit': '2-5% of capital',
                'hourly_pnl_limit': '0.5-1% of daily limit',
                'drawdown_limit': '10-20% of capital'
            },
            'execution_risks': {
                'slippage_tolerance': '0.1-0.5% of spread',
                'stale_quote_timeout': '1-5 seconds',
                'order_cancel_delay': '100-500ms'
            },
            'market_risks': {
                'volatility_filter': 'Pause if realized vol > 2x average',
                'gap_detection': 'Pause if price gap > 5%',
                'liquidity_filter': 'Pause if spread > 2% or depth < threshold'
            },
            'system_risks': {
                'heartbeat_timeout': '30 seconds',
                'cpu_usage_limit': '80%',
                'memory_usage_limit': '90%',
                'network_timeout': '5 seconds'
            }
        }

        return risk_framework

    def scalability_analysis(self, n_assets_range=(1, 50)):
        """
        Analyze scalability with multiple assets
        """
        print("Analyzing scalability for multiple assets...")

        scalability_results = []

        for n_assets in range(n_assets_range[0], n_assets_range[1] + 1, 5):
            # Assume independent PDE solves per asset
            single_solve_time = 50  # ms (conservative estimate)
            total_solve_time = n_assets * single_solve_time

            # Memory scaling
            single_memory = 100  # MB per asset
            total_memory = n_assets * single_memory

            # CPU cores needed (rough estimate)
            cores_needed = min(n_assets, psutil.cpu_count())

            scalability_results.append({
                'n_assets': n_assets,
                'total_solve_time_ms': total_solve_time,
                'total_memory_gb': total_memory / 1024,
                'cores_needed': cores_needed,
                'solve_frequency_limit': 1000 / total_solve_time,  # Hz
                'feasible': total_solve_time < 1000  # Can solve within 1 second
            })

        return pd.DataFrame(scalability_results)

    def deployment_architecture_design(self):
        """
        Design practical deployment architecture
        """
        architecture = {
            'components': {
                'pde_solver': {
                    'technology': 'GPU-accelerated Python (CuPy/Numba)',
                    'responsibility': 'Solve HJB PDE in real-time',
                    'scaling': 'Horizontal scaling with multiple GPUs',
                    'redundancy': 'Hot standby solver instances'
                },
                'market_data_feed': {
                    'technology': 'WebSocket + REST APIs',
                    'responsibility': 'Real-time price and order book data',
                    'scaling': 'Multiple exchange connections',
                    'redundancy': 'Multiple data providers'
                },
                'order_management': {
                    'technology': 'Exchange-specific APIs (CCXT)',
                    'responsibility': 'Order placement and execution',
                    'scaling': 'Rate-limited request queues',
                    'redundancy': 'Circuit breaker patterns'
                },
                'risk_management': {
                    'technology': 'Real-time monitoring and alerting',
                    'responsibility': 'Position limits and emergency stops',
                    'scaling': 'Distributed monitoring',
                    'redundancy': 'Multiple risk checks'
                },
                'monitoring_dashboard': {
                    'technology': 'Dash/Plotly + WebSocket',
                    'responsibility': 'Real-time performance monitoring',
                    'scaling': 'Load-balanced web servers',
                    'redundancy': 'Read replicas'
                }
            },
            'infrastructure': {
                'compute': 'GPU instances (AWS P3/V100 or equivalent)',
                'network': 'Low-latency network (< 100μs round trip)',
                'storage': 'High-speed SSD for logging, Redis for state',
                'location': 'Co-located with exchange data centers'
            },
            'operational': {
                'deployment': 'Docker containers with Kubernetes orchestration',
                'monitoring': 'Prometheus + Grafana stack',
                'logging': 'Structured logging with ELK stack',
                'backup': 'Real-time state replication'
            }
        }

        return architecture

    def regulatory_compliance_checklist(self):
        """
        Checklist for regulatory compliance in crypto market making
        """
        compliance = {
            'sec_requirements': {
                'registration': 'May require broker-dealer registration for HFT',
                'reporting': 'Trade reporting requirements (if applicable)',
                'capital_requirements': 'Sufficient capital for market making activities',
                'risk_management': 'Comprehensive risk management framework'
            },
            'crypto_specific': {
                'kyc_aml': 'Customer identification for large positions',
                'geographic_restrictions': 'Compliance with local crypto regulations',
                'stablecoin_considerations': 'Additional scrutiny for stablecoin pairs',
                'wash_trading': 'Avoid artificial price manipulation'
            },
            'exchange_requirements': {
                'api_terms': 'Compliance with exchange API terms of service',
                'position_limits': 'Adherence to exchange position limits',
                'trading_rules': 'Compliance with exchange-specific trading rules',
                'fee_structure': 'Understanding of maker/taker fee implications'
            },
            'operational_compliance': {
                'audit_trail': 'Complete record of all trading decisions',
                'error_handling': 'Proper handling of failed orders and executions',
                'business_continuity': 'Disaster recovery and business continuity plans',
                'documentation': 'Comprehensive system documentation'
            }
        }

        return compliance

    def plot_deployment_analysis(self, latency_results, memory_results, scalability_results, save_path=None):
        """
        Create comprehensive deployment analysis plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('HJB Market Making: Practical Deployment Analysis', fontsize=16)

        # Latency analysis
        if latency_results is not None and not latency_results.empty:
            grid_sizes = latency_results['grid_size']

            axes[0,0].bar(range(len(grid_sizes)), latency_results['cpu_avg_ms'],
                          label='CPU', alpha=0.7)
            if 'gpu_avg_ms' in latency_results.columns and not latency_results['gpu_avg_ms'].isna().all():
                axes[0,0].bar(range(len(grid_sizes)), latency_results['gpu_avg_ms'],
                              label='GPU', alpha=0.7)

            axes[0,0].set_xticks(range(len(grid_sizes)))
            axes[0,0].set_xticklabels(grid_sizes, rotation=45)
            axes[0,0].set_ylabel('Solve Time (ms)')
            axes[0,0].set_title('Solve Latency by Grid Size')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

            # Add latency requirement lines
            axes[0,0].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='HFT Requirement')
            axes[0,0].axhline(y=1000, color='orange', linestyle='--', alpha=0.7, label='Retail Requirement')

        # Memory usage
        if memory_results is not None and not memory_results.empty:
            grid_sizes = memory_results['grid_size']

            axes[0,1].bar(range(len(grid_sizes)), memory_results['cpu_memory_mb'],
                          label='CPU Memory', alpha=0.7)
            if 'gpu_memory_mb' in memory_results.columns and not memory_results['gpu_memory_mb'].isna().all():
                axes[0,1].bar(range(len(grid_sizes)), memory_results['gpu_memory_mb'],
                              label='GPU Memory', alpha=0.7, bottom=memory_results['cpu_memory_mb'])

            axes[0,1].set_xticks(range(len(grid_sizes)))
            axes[0,1].set_xticklabels(grid_sizes, rotation=45)
            axes[0,1].set_ylabel('Memory Usage (MB)')
            axes[0,1].set_title('Memory Requirements')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)

        # Scalability analysis
        if scalability_results is not None and not scalability_results.empty:
            axes[0,2].plot(scalability_results['n_assets'], scalability_results['total_solve_time_ms'],
                          'bo-', label='Solve Time')
            axes[0,2].axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='1s Limit')
            axes[0,2].set_xlabel('Number of Assets')
            axes[0,2].set_ylabel('Total Solve Time (ms)')
            axes[0,2].set_title('Multi-Asset Scalability')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)

        # Real-time requirements comparison
        requirements = self.real_time_requirements_analysis()
        req_names = list(requirements.keys())
        req_latencies = [req['latency_requirement'] for req in requirements.values()]

        axes[1,0].bar(req_names, req_latencies)
        axes[1,0].set_ylabel('Latency Requirement (μs)')
        axes[1,0].set_title('Real-Time Requirements by Use Case')
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True, alpha=0.3)

        # Exchange comparison
        exchanges = self.exchange_integration_analysis()
        exch_names = list(exchanges.keys())
        exch_latencies = []

        for exch in exchanges.values():
            # Extract numeric latency
            latency_str = exch['latency']
            if 'ms' in latency_str:
                latency = float(latency_str.split('-')[0].strip('~ '))
            else:
                latency = 500  # default
            exch_latencies.append(latency)

        axes[1,1].bar(exch_names, exch_latencies)
        axes[1,1].set_ylabel('Typical Latency (ms)')
        axes[1,1].set_title('Exchange Latency Comparison')
        axes[1,1].grid(True, alpha=0.3)

        # Risk management visualization
        risk_framework = self.risk_management_framework()
        risk_categories = list(risk_framework.keys())

        # Count items per category
        risk_counts = [len(risk_framework[cat]) for cat in risk_categories]

        axes[1,2].bar(risk_categories, risk_counts)
        axes[1,2].set_ylabel('Number of Risk Controls')
        axes[1,2].set_title('Risk Management Framework Coverage')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Deployment analysis plots saved to {save_path}")

        plt.show()

    def run_full_deployment_analysis(self, save_plots=True):
        """
        Run complete deployment analysis suite
        """
        print("Starting comprehensive deployment analysis...")

        # Performance analysis
        latency_results = self.latency_analysis()
        memory_results = self.memory_usage_analysis()

        # Scalability analysis
        scalability_results = self.scalability_analysis()

        # Generate plots
        plot_path = "/Users/misango/codechest/Algorithmic_Trading_and_HFT_Research/Market_Making/HJB_DP_MM_Optimisation/hjb_deployment_analysis.png" if save_plots else None
        self.plot_deployment_analysis(latency_results, memory_results, scalability_results, plot_path)

        # Get other analyses
        requirements = self.real_time_requirements_analysis()
        exchanges = self.exchange_integration_analysis()
        risk_framework = self.risk_management_framework()
        architecture = self.deployment_architecture_design()
        compliance = self.regulatory_compliance_checklist()

        # Print summary
        print("\n" + "="*60)
        print("DEPLOYMENT ANALYSIS SUMMARY")
        print("="*60)

        print("Performance Analysis:")
        if not latency_results.empty:
            fastest_cpu = latency_results.loc[latency_results['cpu_avg_ms'].idxmin()]
            print(".1f")
            if not np.isnan(fastest_cpu['speedup']):
                print(".1f")

        print("\nScalability:")
        if not scalability_results.empty:
            max_feasible = scalability_results[scalability_results['feasible']].tail(1)
            if not max_feasible.empty:
                print(f"Max feasible assets: {max_feasible['n_assets'].values[0]}")

        print("\nReal-time Requirements:")
        for use_case, reqs in requirements.items():
            print(f"  {use_case}: {reqs['max_solve_time']}ms solve time required")

        print("\nKey Deployment Challenges:")
        print("  • Sub-millisecond latency requirements for HFT")
        print("  • Exchange API rate limits and reliability")
        print("  • Real-time risk management and position monitoring")
        print("  • Co-location requirements for low latency")
        print("  • Regulatory compliance for automated trading")

        print("="*60)

        return {
            'latency_analysis': latency_results,
            'memory_analysis': memory_results,
            'scalability_analysis': scalability_results,
            'requirements': requirements,
            'exchanges': exchanges,
            'risk_framework': risk_framework,
            'architecture': architecture,
            'compliance': compliance
        }


if __name__ == "__main__":
    analyzer = DeploymentAnalyzer()

    try:
        results = analyzer.run_full_deployment_analysis()
        print("Deployment analysis completed successfully!")
    except Exception as e:
        print(f"Deployment analysis failed: {e}")
        import traceback
        traceback.print_exc()