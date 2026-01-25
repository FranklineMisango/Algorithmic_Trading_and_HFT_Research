"""
Main Orchestration Script for Copy Congress Strategy

Coordinates the complete strategy pipeline:
1. Data acquisition
2. Signal generation
3. Portfolio construction
4. Backtesting
5. Performance analysis
"""

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_acquisition import CongressionalDataAcquisition
from signal_generator import SignalGenerator
from portfolio_constructor import PortfolioConstructor
from backtester import CongressBacktester


class CopyCongressStrategy:
    """Main strategy orchestrator."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize strategy.
        
        Args:
            config_path: Path to configuration file
        """
        print("Initializing Copy Congress Strategy...")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_acq = CongressionalDataAcquisition(self.config)
        self.signal_gen = SignalGenerator(self.config)
        self.portfolio_constructor = PortfolioConstructor(self.config)
        self.backtester = CongressBacktester(self.config)
        
        # Data storage
        self.congressional_trades = None
        self.prices = None
        self.volumes = None
        self.market_caps = None
        self.volatility = None
        self.signals_history = None
        self.portfolio_history = None
        self.results = None
        
        print("Initialization complete")
    
    def load_data(self):
        """Load all required data."""
        print("\n" + "="*60)
        print("STEP 1: DATA ACQUISITION")
        print("="*60)
        
        (self.congressional_trades, 
         self.prices, 
         self.volumes, 
         self.market_caps, 
         self.volatility) = self.data_acq.get_full_dataset()
        
        print("\nData loaded successfully")
    
    def generate_signals(self):
        """Generate trading signals."""
        print("\n" + "="*60)
        print("STEP 2: SIGNAL GENERATION")
        print("="*60)
        
        # Create rebalance dates
        rebalance_freq = self.config['portfolio']['rebalance_frequency']
        rebalance_dates = pd.date_range(
            start=self.prices.index[0],
            end=self.prices.index[-1],
            freq=rebalance_freq
        )
        
        print(f"Rebalance frequency: {rebalance_freq}")
        print(f"Number of rebalance dates: {len(rebalance_dates)}")
        
        # Generate signals
        self.signals_history = self.signal_gen.generate_signals_timeseries(
            self.congressional_trades,
            self.prices,
            rebalance_dates.tolist()
        )
        
        # Analyze signal quality
        signal_quality = self.signal_gen.analyze_signal_quality(self.signals_history)
        
        print("\nSignals generated successfully")
    
    def construct_portfolios(self):
        """Construct portfolio allocations."""
        print("\n" + "="*60)
        print("STEP 3: PORTFOLIO CONSTRUCTION")
        print("="*60)
        
        # Create rebalance dates
        rebalance_freq = self.config['portfolio']['rebalance_frequency']
        rebalance_dates = pd.date_range(
            start=self.prices.index[0],
            end=self.prices.index[-1],
            freq=rebalance_freq
        )
        
        # Construct portfolios
        self.portfolio_history = self.portfolio_constructor.generate_portfolio_timeseries(
            self.signals_history,
            self.volatility,
            rebalance_dates.tolist()
        )
        
        # Analyze characteristics
        portfolio_chars = self.portfolio_constructor.analyze_portfolio_characteristics(
            self.portfolio_history
        )
        
        print("\nPortfolios constructed successfully")
    
    def run_backtest(self):
        """Run backtest simulation."""
        print("\n" + "="*60)
        print("STEP 4: BACKTESTING")
        print("="*60)
        
        # Simulate portfolio
        self.results = self.backtester.simulate_portfolio(
            self.portfolio_history,
            self.prices
        )
        
        # Calculate performance metrics
        metrics = self.backtester.calculate_performance_metrics(self.results)
        self.backtester.print_performance_summary(metrics)
        
        # Compare to benchmark
        benchmark_ticker = self.config['backtest']['benchmark']
        benchmark_results = self.backtester.calculate_benchmark_returns(
            benchmark_ticker,
            self.prices
        )
        
        if len(benchmark_results) > 0:
            comparison = self.backtester.compare_to_benchmark(
                self.results,
                benchmark_results
            )
            
            print("\n" + "="*60)
            print("BENCHMARK COMPARISON")
            print("="*60)
            print(f"Strategy Return:       {comparison['strategy_return']:>10.2%}")
            print(f"Benchmark Return:      {comparison['benchmark_return']:>10.2%}")
            print(f"Excess Return:         {comparison['excess_return']:>10.2%}")
            print(f"Information Ratio:     {comparison['information_ratio']:>10.2f}")
            print(f"Tracking Error:        {comparison['tracking_error']:>10.2%}")
            print("="*60)
        
        print("\nBacktest complete")
    
    def generate_visualizations(self, output_dir: str = 'output'):
        """Generate performance visualizations."""
        print("\n" + "="*60)
        print("STEP 5: VISUALIZATION")
        print("="*60)
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Cumulative returns
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.results.index, self.results['cumulative_returns'] * 100, 
                label='Copy Congress Strategy', linewidth=2)
        
        benchmark_ticker = self.config['backtest']['benchmark']
        if benchmark_ticker in self.prices.columns:
            benchmark_returns = self.prices[benchmark_ticker].pct_change()
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
            ax.plot(benchmark_cumulative.index, benchmark_cumulative * 100,
                   label=f'{benchmark_ticker} Benchmark', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title('Copy Congress Strategy: Cumulative Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cumulative_returns.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/cumulative_returns.png")
        plt.close()
        
        # 2. Drawdown
        fig, ax = plt.subplots(figsize=(12, 6))
        cumulative = (1 + self.results['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Copy Congress Strategy: Drawdown')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/drawdown.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/drawdown.png")
        plt.close()
        
        # 3. Rolling Sharpe ratio
        fig, ax = plt.subplots(figsize=(12, 6))
        rolling_returns = self.results['returns'].rolling(window=252)
        rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)
        ax.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Rolling Sharpe Ratio')
        ax.set_title('Copy Congress Strategy: 1-Year Rolling Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/rolling_sharpe.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/rolling_sharpe.png")
        plt.close()
        
        # 4. Number of holdings over time
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.results.index, self.results['n_positions'], linewidth=1.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Holdings')
        ax.set_title('Copy Congress Strategy: Portfolio Holdings Over Time')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/holdings_timeseries.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/holdings_timeseries.png")
        plt.close()
        
        # 5. Monthly returns heatmap
        monthly_returns = self.results['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot = monthly_returns.to_frame('returns')
        monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
        heatmap_data = monthly_returns_pivot.pivot(index='year', columns='month', values='returns') * 100
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        ax.set_title('Copy Congress Strategy: Monthly Returns Heatmap')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/monthly_returns_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/monthly_returns_heatmap.png")
        plt.close()
        
        print("\nVisualization complete")
    
    def run_full_strategy(self):
        """Execute complete strategy pipeline."""
        print("\n" + "="*60)
        print("COPY CONGRESS TRADING STRATEGY")
        print("="*60)
        print(f"Start Date: {self.config['data']['start_date']}")
        print(f"End Date: {self.config['data']['end_date']}")
        print(f"Initial Capital: ${self.config['backtest']['initial_capital']:,.0f}")
        print("="*60)
        
        # Execute pipeline
        self.load_data()
        self.generate_signals()
        self.construct_portfolios()
        self.run_backtest()
        self.generate_visualizations()
        
        print("\n" + "="*60)
        print("STRATEGY EXECUTION COMPLETE")
        print("="*60)
        print("\nCheck the 'output' directory for visualizations")
        print("Run Jupyter notebooks for detailed analysis")


if __name__ == "__main__":
    # Run the strategy
    strategy = CopyCongressStrategy()
    strategy.run_full_strategy()
