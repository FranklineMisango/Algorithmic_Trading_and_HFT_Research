"""
Regime-Based ML Strategy for Portfolio Allocation

Instead of predicting returns, this strategy:
1. Uses ML to predict market regimes
2. Applies regime-specific allocation rules
3. Tests multiple models: Gradient Boosting, Neural Networks, Ensemble
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class RegimeMLStrategy:
    """ML-based regime prediction and allocation strategy."""
    
    def __init__(self, config: Dict):
        """Initialize the strategy."""
        self.config = config
        self.scaler = StandardScaler()
        self.models = {}
        self.regime_allocations = self._define_regime_allocations()
        
    def _define_regime_allocations(self) -> Dict:
        """
        Define fixed allocations for each regime.
        
        Regimes:
        0 = Defensive (high vol, bear market, inverted curve)
        1 = Neutral (normal conditions)
        2 = Aggressive (low vol, bull market, steep curve)
        """
        assets = []
        for asset in self.config['assets']['traditional']:
            assets.append(asset['ticker'])
        for asset in self.config['assets']['alternative']:
            assets.append(asset['ticker'])
        
        # Define allocations for each regime
        allocations = {
            0: {  # Defensive: bonds + gold, minimal equity/crypto
                'SPY': 0.15,
                'TLT': 0.30,
                'BTC-USD': 0.00,
                'GLD': 0.25,
                'SHY': 0.20,
                'IEF': 0.10
            },
            1: {  # Neutral: balanced 60/40 style
                'SPY': 0.40,
                'TLT': 0.25,
                'BTC-USD': 0.05,
                'GLD': 0.10,
                'SHY': 0.10,
                'IEF': 0.10
            },
            2: {  # Aggressive: equity heavy with crypto
                'SPY': 0.60,
                'TLT': 0.10,
                'BTC-USD': 0.15,
                'GLD': 0.05,
                'SHY': 0.05,
                'IEF': 0.05
            }
        }
        
        return allocations
    
    def create_regime_labels(self, 
                            vix: pd.Series,
                            prices: pd.Series,
                            yield_spread: pd.Series,
                            returns: pd.Series) -> pd.Series:
        """
        Create regime labels based on market conditions.
        
        Uses forward-looking returns to label regimes (for training only).
        """
        # Calculate forward 3-month return
        forward_return = returns.rolling(3).sum().shift(-3)
        
        # Volatility regime
        vix_high = vix > vix.rolling(60).quantile(0.67)
        vix_low = vix < vix.rolling(60).quantile(0.33)
        
        # Trend regime
        ma_60 = prices.rolling(60).mean()
        price_above_ma = prices > ma_60 * 1.02
        price_below_ma = prices < ma_60 * 0.98
        
        # Yield curve
        curve_inverted = yield_spread < 0
        curve_steep = yield_spread > 1.5
        
        # Forward return
        return_positive = forward_return > 0.05
        return_negative = forward_return < -0.05
        
        # Combine signals
        regime = pd.Series(1, index=vix.index)  # Default: neutral
        
        # Defensive: high vol OR bear trend OR negative forward returns
        defensive = (vix_high | price_below_ma | return_negative | curve_inverted)
        regime[defensive] = 0
        
        # Aggressive: low vol AND bull trend AND positive forward returns
        aggressive = (vix_low & price_above_ma & return_positive & ~curve_inverted)
        regime[aggressive] = 2
        
        return regime
    
    def create_features(self,
                       indicators: pd.DataFrame,
                       prices: pd.DataFrame,
                       returns: pd.DataFrame) -> pd.DataFrame:
        """Create features for regime prediction."""
        features = pd.DataFrame(index=indicators.index)
        
        # VIX features
        if 'VIX' in indicators.columns:
            features['vix'] = indicators['VIX']
            features['vix_ma_20'] = indicators['VIX'].rolling(20).mean()
            features['vix_ma_60'] = indicators['VIX'].rolling(60).mean()
            features['vix_std'] = indicators['VIX'].rolling(20).std()
            features['vix_change'] = indicators['VIX'].pct_change()
            features['vix_zscore'] = (indicators['VIX'] - indicators['VIX'].rolling(60).mean()) / indicators['VIX'].rolling(60).std()
        
        # Yield spread features
        if 'Yield_Spread' in indicators.columns:
            features['yield_spread'] = indicators['Yield_Spread']
            features['yield_spread_ma'] = indicators['Yield_Spread'].rolling(20).mean()
            features['yield_spread_change'] = indicators['Yield_Spread'].diff()
            features['yield_inverted'] = (indicators['Yield_Spread'] < 0).astype(int)
        
        # Interest rate features
        if 'Interest_Rate' in indicators.columns:
            features['interest_rate'] = indicators['Interest_Rate']
            features['rate_change'] = indicators['Interest_Rate'].diff()
            features['rate_ma'] = indicators['Interest_Rate'].rolling(20).mean()
        
        # SPY price features
        if 'SPY' in prices.columns:
            spy = prices['SPY']
            features['spy_return_1m'] = returns['SPY']
            features['spy_return_3m'] = returns['SPY'].rolling(3).sum()
            features['spy_return_6m'] = returns['SPY'].rolling(6).sum()
            features['spy_ma_20'] = spy.rolling(20).mean()
            features['spy_ma_60'] = spy.rolling(60).mean()
            features['spy_ma_200'] = spy.rolling(200).mean()
            features['spy_vs_ma20'] = (spy - features['spy_ma_20']) / features['spy_ma_20']
            features['spy_vs_ma60'] = (spy - features['spy_ma_60']) / features['spy_ma_60']
            features['spy_vol'] = returns['SPY'].rolling(20).std()
            
            # RSI
            delta = spy.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            features['spy_rsi'] = 100 - (100 / (1 + rs))
        
        # TLT features
        if 'TLT' in prices.columns:
            features['tlt_return_1m'] = returns['TLT']
            features['tlt_return_3m'] = returns['TLT'].rolling(3).sum()
            features['tlt_vol'] = returns['TLT'].rolling(20).std()
        
        # Stock-bond correlation
        if 'SPY' in returns.columns and 'TLT' in returns.columns:
            features['spy_tlt_corr'] = returns['SPY'].rolling(60).corr(returns['TLT'])
        
        # Gold features
        if 'GLD' in returns.columns:
            features['gld_return_3m'] = returns['GLD'].rolling(3).sum()
        
        # Bitcoin features
        if 'BTC-USD' in returns.columns:
            features['btc_return_3m'] = returns['BTC-USD'].rolling(3).sum()
            features['btc_vol'] = returns['BTC-USD'].rolling(20).std()
        
        # Cross-asset momentum
        if 'SPY' in returns.columns:
            features['momentum_score'] = 0
            for col in returns.columns:
                if col in returns.columns:
                    features['momentum_score'] += returns[col].rolling(3).sum()
            features['momentum_score'] = features['momentum_score'] / len(returns.columns)
        
        return features.fillna(method='ffill').fillna(0)
    
    def train_models(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series) -> Dict:
        """
        Train multiple models for regime prediction.
        
        Returns:
            Dictionary of trained models
        """
        print("\nTraining regime prediction models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        models = {}
        
        # 1. Gradient Boosting
        print("  - Gradient Boosting Classifier...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        models['gradient_boosting'] = gb_model
        
        # 2. Random Forest
        print("  - Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        models['random_forest'] = rf_model
        
        # 3. Neural Network
        print("  - Neural Network (MLP)...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=False  # Disable early stopping for small datasets
        )
        nn_model.fit(X_train_scaled, y_train)
        models['neural_network'] = nn_model
        
        # 4. Ensemble (Voting)
        print("  - Ensemble (Voting)...")
        ensemble = VotingClassifier(
            estimators=[
                ('gb', gb_model),
                ('rf', rf_model),
                ('nn', nn_model)
            ],
            voting='soft'
        )
        ensemble.fit(X_train_scaled, y_train)
        models['ensemble'] = ensemble
        
        self.models = models
        return models
    
    def evaluate_models(self,
                       X_test: pd.DataFrame,
                       y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all models on test set."""
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Predictions': y_pred
            })
            
            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {accuracy:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=['Defensive', 'Neutral', 'Aggressive']))
        
        return pd.DataFrame(results)
    
    def predict_regime(self,
                      X: pd.DataFrame,
                      model_name: str = 'ensemble') -> pd.Series:
        """Predict regime using specified model."""
        X_scaled = self.scaler.transform(X)
        model = self.models[model_name]
        predictions = model.predict(X_scaled)
        return pd.Series(predictions, index=X.index)
    
    def get_allocations_for_regime(self,
                                   regimes: pd.Series) -> pd.DataFrame:
        """Convert regime predictions to portfolio allocations."""
        allocations = pd.DataFrame(index=regimes.index)
        
        for asset in self.regime_allocations[0].keys():
            allocations[asset] = 0.0
        
        for idx in regimes.index:
            regime = regimes.loc[idx]
            for asset, weight in self.regime_allocations[regime].items():
                allocations.loc[idx, asset] = weight
        
        return allocations
    
    def get_feature_importance(self, model_name: str = 'gradient_boosting') -> pd.Series:
        """Get feature importance from tree-based model."""
        if model_name not in ['gradient_boosting', 'random_forest']:
            print(f"Feature importance not available for {model_name}")
            return None
        
        model = self.models[model_name]
        # Note: feature names are lost after scaling, need to track them
        return pd.Series(model.feature_importances_)
