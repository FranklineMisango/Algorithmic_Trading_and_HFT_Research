"""
Main execution script for Statistical Arbitrage RL Strategy

Orchestrates the complete pipeline: data acquisition, pair selection,
RL training, and backtesting.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from data_acquisition import DataAcquisition
from pair_selection import PairSelector
from feature_engineering import FeatureEngineer
from rl_agent import DQNAgent, PairsTradingEnv
from backtester import Backtester


class StatArbRLPipeline:
    """Complete pipeline for Statistical Arbitrage RL strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_acq = DataAcquisition(config_path)
        self.pair_selector = PairSelector(config_path)
        self.feature_eng = FeatureEngineer(config_path)
        self.backtester = Backtester(config_path)
        
        self.agent = None
        self.selected_pairs = None
        
    def run_data_acquisition(self):
        """Step 1: Fetch and prepare data."""
        print("\n" + "="*50)
        print("STEP 1: DATA ACQUISITION")
        print("="*50)
        
        dataset = self.data_acq.fetch_full_dataset()
        
        print(f"\nLoaded {dataset['metadata']['total_tickers']} stocks")
        print(f"Date range: {dataset['metadata']['date_range']}")
        
        return dataset
    
    def run_pair_selection(self, dataset):
        """Step 2: Select optimal pairs using grid search."""
        print("\n" + "="*50)
        print("STEP 2: PAIR SELECTION (GRID SEARCH)")
        print("="*50)
        
        # Split data
        train_prices, test_prices = self.data_acq.split_train_test(dataset['prices'])
        
        # Run selection on training data
        selection_results = self.pair_selector.run_selection(
            train_prices,
            dataset['constituents']
        )
        
        self.selected_pairs = selection_results['selected_pairs']
        
        # Save selected pairs
        self.selected_pairs.to_csv('selected_pairs.csv', index=False)
        print(f"\nSelected pairs saved to selected_pairs.csv")
        
        return train_prices, test_prices
    
    def train_rl_agent(self, train_prices):
        """Step 3: Train RL agent on selected pairs."""
        print("\n" + "="*50)
        print("STEP 3: RL AGENT TRAINING")
        print("="*50)
        
        if self.selected_pairs is None or len(self.selected_pairs) == 0:
            raise ValueError("No pairs selected. Run pair selection first.")
        
        # Train on first pair (can extend to all pairs)
        first_pair = self.selected_pairs.iloc[0]
        ticker1 = first_pair['ticker1']
        ticker2 = first_pair['ticker2']
        
        print(f"\nTraining on pair: {ticker1}-{ticker2}")
        
        # Create features
        states = self.feature_eng.create_state_vector(
            train_prices[ticker1],
            train_prices[ticker2]
        )
        states = self.feature_eng.normalize_features(states)
        
        # Create environment
        env = PairsTradingEnv(
            states,
            train_prices[ticker1],
            train_prices[ticker2]
        )
        
        # Initialize agent
        self.agent = DQNAgent()
        self.agent.initialize_networks(states.shape[1])
        
        # Training loop
        episodes = self.config['rl_agent']['episodes']
        target_update = self.config['rl_agent']['target_update_frequency']
        
        episode_rewards = []
        episode_values = []
        
        for episode in tqdm(range(episodes), desc="Training"):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Environment step
                next_state, reward, done, info = env.step(action)
                
                # Store in replay buffer
                self.agent.memory.push(state, action, reward, next_state, done)
                
                # Train agent
                self.agent.train_step()
                
                episode_reward += reward
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_values.append(info['portfolio_value'])
            
            # Decay exploration
            self.agent.decay_epsilon()
            
            # Update target network
            if episode % target_update == 0:
                self.agent.update_target_network()
            
            # Log progress
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_value = np.mean(episode_values[-50:])
                print(f"\nEpisode {episode+1}/{episodes}")
                print(f"  Avg Reward: {avg_reward:.4f}")
                print(f"  Avg Portfolio Value: ${avg_value:,.2f}")
                print(f"  Epsilon: {self.agent.epsilon:.4f}")
        
        # Save trained agent
        self.agent.save('trained_agent.pth')
        print("\nTrained agent saved to trained_agent.pth")
        
        # Save training metrics
        training_df = pd.DataFrame({
            'episode': range(episodes),
            'reward': episode_rewards,
            'portfolio_value': episode_values
        })
        training_df.to_csv('training_metrics.csv', index=False)
        
        return training_df
    
    def run_backtest(self, test_prices):
        """Step 4: Backtest on out-of-sample data."""
        print("\n" + "="*50)
        print("STEP 4: BACKTESTING")
        print("="*50)
        
        if self.agent is None:
            raise ValueError("Agent not trained. Run training first.")
        
        # Prepare pairs for backtesting
        pairs_data = []
        
        for _, pair in self.selected_pairs.iterrows():
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            
            if ticker1 not in test_prices.columns or ticker2 not in test_prices.columns:
                continue
            
            # Create features
            states = self.feature_eng.create_state_vector(
                test_prices[ticker1],
                test_prices[ticker2]
            )
            states = self.feature_eng.normalize_features(states)
            
            pairs_data.append({
                'pair_name': f"{ticker1}_{ticker2}",
                'states': states,
                'price1': test_prices[ticker1],
                'price2': test_prices[ticker2]
            })
        
        # Run backtest
        results = self.backtester.run_multi_pair_backtest(self.agent, pairs_data)
        
        # Print summary
        print("\n" + "="*50)
        print("BACKTEST SUMMARY")
        print("="*50)
        print(f"Number of Pairs:        {results['num_pairs']}")
        print(f"Total Return:           {results['total_return_pct']:.2f}%")
        print(f"Avg Sharpe Ratio:       {results['avg_sharpe_ratio']:.2f}")
        print(f"Avg Max Drawdown:       {results['avg_max_drawdown']*100:.2f}%")
        print(f"Total Trades:           {results['total_trades']}")
        print(f"Avg Win Rate:           {results['avg_win_rate']*100:.1f}%")
        
        # Save detailed results
        results_df = pd.DataFrame([
            {
                'pair': r['pair_name'],
                'return_pct': r['total_return_pct'],
                'sharpe': r['sharpe_ratio'],
                'max_dd': r['max_drawdown'],
                'num_trades': r['num_trades'],
                'win_rate': r['win_rate']
            }
            for r in results['individual_results']
        ])
        results_df.to_csv('backtest_results.csv', index=False)
        print("\nDetailed results saved to backtest_results.csv")
        
        return results
    
    def run_full_pipeline(self):
        """Execute complete pipeline."""
        print("\n" + "="*60)
        print("STATISTICAL ARBITRAGE WITH REINFORCEMENT LEARNING")
        print("="*60)
        
        # Step 1: Data
        dataset = self.run_data_acquisition()
        
        # Step 2: Pair Selection
        train_prices, test_prices = self.run_pair_selection(dataset)
        
        # Step 3: Training
        training_metrics = self.train_rl_agent(train_prices)
        
        # Step 4: Backtesting
        backtest_results = self.run_backtest(test_prices)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        return {
            'selected_pairs': self.selected_pairs,
            'training_metrics': training_metrics,
            'backtest_results': backtest_results
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Statistical Arbitrage RL Strategy')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train', 'backtest'],
                       help='Execution mode')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    pipeline = StatArbRLPipeline(args.config)
    
    if args.mode == 'full':
        pipeline.run_full_pipeline()
    elif args.mode == 'train':
        dataset = pipeline.run_data_acquisition()
        train_prices, _ = pipeline.run_pair_selection(dataset)
        pipeline.train_rl_agent(train_prices)
    elif args.mode == 'backtest':
        # Load trained agent
        dataset = pipeline.run_data_acquisition()
        _, test_prices = pipeline.run_pair_selection(dataset)
        
        # Load selected pairs
        pipeline.selected_pairs = pd.read_csv('selected_pairs.csv')
        
        # Load trained agent
        pipeline.agent = DQNAgent(args.config)
        states = pipeline.feature_eng.create_state_vector(
            test_prices.iloc[:, 0],
            test_prices.iloc[:, 1]
        )
        pipeline.agent.initialize_networks(states.shape[1])
        pipeline.agent.load('trained_agent.pth')
        
        pipeline.run_backtest(test_prices)


if __name__ == "__main__":
    main()
