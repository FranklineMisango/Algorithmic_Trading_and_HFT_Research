"""
DRL Agent for Portfolio Allocation

Implements PPO agent using Stable-Baselines3.
"""

import numpy as np
import pandas as pd
import yaml
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("Warning: stable-baselines3 not installed")


class DRLAgent:
    """
    DRL agent wrapper for portfolio allocation.
    Supports PPO, A2C, and SAC algorithms.
    """
    
    def __init__(
        self,
        env,
        config_path: str = "config.yaml",
        algorithm: str = "ppo"
    ):
        """
        Initialize DRL agent.
        
        Args:
            env: Gymnasium environment
            config_path: Path to config file
            algorithm: 'ppo', 'a2c', or 'sac'
        """
        self.env = env
        self.algorithm_name = algorithm.lower()
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        agent_config = self.config['agent']
        
        # Prepare policy kwargs
        policy_config = agent_config['policy']
        policy_kwargs = {
            'net_arch': {
                'pi': policy_config['net_arch']['pi'],
                'vf': policy_config['net_arch']['vf']
            },
            'activation_fn': self._get_activation_fn(policy_config['activation'])
        }
        
        # Get algorithm hyperparameters
        if self.algorithm_name == 'ppo':
            algo_params = agent_config['ppo']
            self.model = PPO(
                policy=policy_config['policy_class'],
                env=env,
                learning_rate=algo_params['learning_rate'],
                n_steps=algo_params['n_steps'],
                batch_size=algo_params['batch_size'],
                n_epochs=algo_params['n_epochs'],
                gamma=algo_params['gamma'],
                gae_lambda=algo_params['gae_lambda'],
                clip_range=algo_params['clip_range'],
                ent_coef=algo_params['ent_coef'],
                vf_coef=algo_params['vf_coef'],
                max_grad_norm=algo_params['max_grad_norm'],
                policy_kwargs=policy_kwargs,
                verbose=agent_config['training']['verbose']
            )
        
        elif self.algorithm_name == 'a2c':
            self.model = A2C(
                policy='MlpPolicy',
                env=env,
                policy_kwargs=policy_kwargs,
                verbose=1
            )
        
        elif self.algorithm_name == 'sac':
            self.model = SAC(
                policy='MlpPolicy',
                env=env,
                policy_kwargs=policy_kwargs,
                verbose=1
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.training_config = agent_config['training']
    
    def _get_activation_fn(self, activation_name: str):
        """Get activation function from name."""
        import torch.nn as nn
        
        activations = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'elu': nn.ELU
        }
        
        return activations.get(activation_name, nn.Tanh)
    
    def train(
        self,
        eval_env: Optional[object] = None,
        save_path: str = "models/best_model"
    ) -> Dict:
        """
        Train the agent.
        
        Args:
            eval_env: Evaluation environment
            save_path: Path to save best model
            
        Returns:
            Training info dict
        """
        callbacks = []
        
        # Evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path='logs/',
                eval_freq=self.training_config['eval_freq'],
                n_eval_episodes=self.training_config['n_eval_episodes'],
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.training_config['save_freq'],
            save_path='models/checkpoints/',
            name_prefix='portfolio_model'
        )
        callbacks.append(checkpoint_callback)
        
        # Train
        print(f"\nTraining {self.algorithm_name.upper()} agent...")
        print(f"Total timesteps: {self.training_config['total_timesteps']:,}")
        
        self.model.learn(
            total_timesteps=self.training_config['total_timesteps'],
            callback=callbacks,
            log_interval=self.training_config['log_interval']
        )
        
        return {'status': 'completed'}
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Predict action from observation.
        
        Args:
            observation: State vector
            deterministic: Use deterministic policy
            
        Returns:
            Action (portfolio weights)
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def evaluate(
        self,
        eval_env,
        n_episodes: int = 10
    ) -> Dict:
        """
        Evaluate agent performance.
        
        Args:
            eval_env: Evaluation environment
            n_episodes: Number of episodes
            
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_returns = []
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            initial_value = 1.0
            
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            final_value = info['portfolio_value']
            total_return = (final_value - initial_value) / initial_value
            
            episode_rewards.append(episode_reward)
            episode_returns.append(total_return)
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_return = np.mean(episode_returns)
        
        # Sharpe ratio approximation
        sharpe = (mean_return * np.sqrt(252)) / (np.std(episode_returns) * np.sqrt(252) + 1e-8)
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_return': mean_return,
            'sharpe_ratio': sharpe,
            'n_episodes': n_episodes
        }
    
    def save(self, path: str):
        """Save model."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model."""
        if self.algorithm_name == 'ppo':
            self.model = PPO.load(path, env=self.env)
        elif self.algorithm_name == 'a2c':
            self.model = A2C.load(path, env=self.env)
        elif self.algorithm_name == 'sac':
            self.model = SAC.load(path, env=self.env)
        
        print(f"Model loaded from {path}")


# Test code
if __name__ == "__main__":
    from data_acquisition import DataAcquisition
    from portfolio_env import PortfolioEnv
    
    data_acq = DataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset()
    
    # Create environments
    train_env = PortfolioEnv(
        prices=dataset['train']['prices'],
        returns=dataset['train']['returns']
    )
    
    # Create agent
    agent = DRLAgent(train_env, algorithm='ppo')
    
    print(f"Agent: {agent.algorithm_name.upper()}")
    print(f"Model: {agent.model.policy}")
    
    # Test prediction
    obs, _ = train_env.reset()
    action = agent.predict(obs)
    print(f"\nSample action (weights): {action}")
    print(f"Sum: {np.sum(action):.4f}")
