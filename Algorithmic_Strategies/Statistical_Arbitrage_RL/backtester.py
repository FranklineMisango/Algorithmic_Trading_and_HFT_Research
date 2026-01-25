"""
Backtester for Statistical Arbitrage RL Strategy

Evaluates trained RL agent on out-of-sample data with realistic costs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yaml

from rl_agent import DQNAgent, PairsTradingEnv
from feature_engineering import FeatureEngineer


class Backtester:
    """Backtest RL pairs trading strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.execution_config = self.config['execution']
        self.risk_config = self.config['risk_management']
        self.portfolio_config = self.config['portfolio']
        
        self.slippage_bps = self.execution_config['slippage_bps']
        self.commission_bps = self.execution_config['commission_bps']
        
    def calculate_transaction_costs(self, 
                                     trade_value: float) -> float:
        """
        Calculate realistic transaction costs.
        
        Args:
            trade_value: Dollar value of trade
            
        Returns:
            Total cost in dollars
        """
        slippage = trade_value * (self.slippage_bps / 10000)
        commission = trade_value * (self.commission_bps / 10000)
        
        return slippage + commission
    
    def run_backtest(self,
                     agent: DQNAgent,
                     states: pd.DataFrame,
                     price1: pd.Series,
                     price2: pd.Series,
                     pair_name: str = "Pair") -> Dict:
        """
        Run backtest for single pair using trained agent.
        
        Args:
            agent: Trained DQNAgent
            states: State features
            price1, price2: Price series for the pair
            pair_name: Identifier for the pair
            
        Returns:
            Dictionary with backtest results
        """
        # Create environment
        env = PairsTradingEnv(
            states,
            price1,
            price2,
            initial_capital=self.portfolio_config['initial_capital']
        )
        
        # Run episode
        state = env.reset()
        done = False
        
        actions_taken = []
        timestamps = states.index.tolist()
        
        while not done:
            # Agent selects action (no exploration in backtest)
            action = agent.select_action(state, training=False)
            actions_taken.append(action)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        # Calculate performance metrics
        portfolio_values = np.array(env.portfolio_values)
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Adjust for transaction costs
        num_trades = len(env.trades)
        avg_trade_size = env.initial_capital * 0.2  # 10% per leg * 2 legs
        total_costs = num_trades * self.calculate_transaction_costs(avg_trade_size)
        
        final_value = portfolio_values[-1] - total_costs
        total_return = (final_value - env.initial_capital) / env.initial_capital
        
        # Risk metrics
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Trade analysis
        trade_pnls = [t['pnl'] for t in env.trades]
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p < 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        
        profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if len(losing_trades) > 0 else np.inf
        
        results = {
            'pair_name': pair_name,
            'initial_capital': env.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'transaction_costs': total_costs,
            'portfolio_values': portfolio_values,
            'returns': returns,
            'trades': env.trades,
            'actions': actions_taken,
            'timestamps': timestamps
        }
        
        return results
    
    def run_multi_pair_backtest(self,
                                  agent: DQNAgent,
                                  pairs_data: List[Dict]) -> Dict:
        """
        Run backtest across multiple pairs.
        
        Args:
            agent: Trained DQNAgent
            pairs_data: List of dicts with {pair_name, states, price1, price2}
            
        Returns:
            Aggregate results across all pairs
        """
        all_results = []
        
        for pair_data in pairs_data:
            print(f"Backtesting {pair_data['pair_name']}...")
            
            results = self.run_backtest(
                agent,
                pair_data['states'],
                pair_data['price1'],
                pair_data['price2'],
                pair_data['pair_name']
            )
            
            all_results.append(results)
        
        # Aggregate metrics
        total_return = np.mean([r['total_return'] for r in all_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
        avg_max_dd = np.mean([r['max_drawdown'] for r in all_results])
        total_trades = sum([r['num_trades'] for r in all_results])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        
        aggregate = {
            'num_pairs': len(pairs_data),
            'total_return_pct': total_return * 100,
            'avg_sharpe_ratio': avg_sharpe,
            'avg_max_drawdown': avg_max_dd,
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'individual_results': all_results
        }
        
        return aggregate
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate formatted backtest report.
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Formatted report string
        """
        report = f"""
=== BACKTEST REPORT ===

Pair: {results['pair_name']}

PERFORMANCE
-----------
Initial Capital:    ${results['initial_capital']:,.2f}
Final Value:        ${results['final_value']:,.2f}
Total Return:       {results['total_return_pct']:.2f}%
Sharpe Ratio:       {results['sharpe_ratio']:.2f}
Max Drawdown:       {results['max_drawdown']*100:.2f}%

TRADING ACTIVITY
----------------
Number of Trades:   {results['num_trades']}
Win Rate:           {results['win_rate']*100:.1f}%
Average Win:        ${results['avg_win']:,.2f}
Average Loss:       ${results['avg_loss']:,.2f}
Profit Factor:      {results['profit_factor']:.2f}

COSTS
-----
Transaction Costs:  ${results['transaction_costs']:,.2f}
Cost/Trade:         ${results['transaction_costs']/results['num_trades']:,.2f}

"""
        return report


if __name__ == "__main__":
    # Test backtester
    from data_acquisition import DataAcquisition
    
    # Fetch data
    data_acq = DataAcquisition()
    dataset = data_acq.fetch_full_dataset()
    _, test_prices = data_acq.split_train_test(dataset['prices'])
    
    # Create features
    feature_eng = FeatureEngineer()
    states = feature_eng.create_state_vector(
        test_prices['MSFT'],
        test_prices['GOOGL']
    )
    states = feature_eng.normalize_features(states)
    
    # Create untrained agent (for testing)
    agent = DQNAgent()
    agent.initialize_networks(states.shape[1])
    
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest(
        agent,
        states,
        test_prices['MSFT'],
        test_prices['GOOGL'],
        "MSFT_GOOGL"
    )
    
    # Print report
    print(backtester.generate_report(results))
