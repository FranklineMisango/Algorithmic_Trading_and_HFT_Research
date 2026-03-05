"""
Regime-Based ML Strategy V2 - Improved Version

Key improvements:
1. Calibrated regime definitions for better bull market detection
2. Dynamic allocation optimization within regimes
3. More forward-looking indicators
4. No look-ahead bias in training
5. Longer historical data (back to 2000)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class RegimeMLStrategyV2:
    """Improved ML-based regime prediction and allocation strategy."""
    
    def __init__(self, config: Dict):
        """Initialize the strategy."""
        self.config = config
        self.scaler = StandardScaler()
        self.models = {}
        self.regime_allocations = self._define_regime_allocations()
        
    def _define_regime_allocations(self) -> Dict:
        """
        Define BASE allocations for each regime (will be optimized dynamically).
        
        Regimes:
        0 = Defensive (crisis, high vol, bear market)
        1 = Neutral (normal conditions)
        2 = Aggressive (low vol, bull market, strong momentum)
        """
        # Base allocations - will be adjusted dynamically
        allocations = {
            0: {  # Defensive: safety first
                'SPY': 0.10,
                'TLT': 0.35,
                'BTC-USD': 0.00,
                'GLD': 0.30,
                'SHY': 0.15,
                'IEF': 0.10
            },
            1: {  # Neutral: balanced
                'SPY': 0.45,
                'TLT': 0.25,
                'BTC-USD': 0.05,
                'GLD': 0.10,
                'SHY': 0.08,
                'IEF': 0.07
            },
            2: {  # Aggressive: growth focused (CALIBRATED TO TRIGGER MORE EASILY)
                'SPY': 0.65,
                'TLT': 0.08,
                'BTC-USD': 0.12,
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
                            returns: pd.Series,
                            use_forward_looking: bool = True) -> pd.Series:
        """
        Create regime labels - CALIBRATED for better bull market detection.
        
        Args:
            use_forward_looking: If True, uses forward returns (training only)
                                If False, uses only current state (prediction)
        """
        # Volatility regime (RECALIBRATED)
        vix_high = vix > vix.rolling(60).quantile(0.75)  # Top 25% = high vol
        vix_low = vix < vix.rolling(60).quantile(0.40)   # Bottom 40% = low vol (easier to trigger)
        
        # Trend regime (RECALIBRATED)
        ma_60 = prices.rolling(60).mean()
        ma_200 = prices.rolling(200).mean()
        price_above_ma = prices > ma_60 * 1.00  # At or above MA (easier)
        price_below_ma = prices < ma_60 * 0.97  # 3% below MA
        strong_uptrend = (prices > ma_60) & (ma_60 > ma_200)  # Golden cross
        
        # Momentum (NEW - forward-looking)
        momentum_3m = returns.rolling(3).sum()
        momentum_6m = returns.rolling(6).sum()
        strong_momentum = (momentum_3m > 0.05) & (momentum_6m > 0.08)  # 5% and 8% gains
        weak_momentum = (momentum_3m < -0.03) | (momentum_6m < -0.05)
        
        # Yield curve
        curve_inverted = yield_spread < -0.2  # Strongly inverted
        curve_steep = yield_spread > 1.2  # Steep curve (easier to trigger)
        
        # Initialize regime
        regime = pd.Series(1, index=vix.index)  # Default: neutral
        
        # DEFENSIVE: Crisis conditions
        # High vol OR bear trend OR inverted curve OR weak momentum
        defensive = (vix_high | price_below_ma | curve_inverted | weak_momentum)
        regime[defensive] = 0
        
        # AGGRESSIVE: Bull market conditions (RECALIBRATED - easier to trigger)
        # Low vol AND (uptrend OR strong momentum) AND not inverted
        aggressive = (vix_low & (price_above_ma | strong_momentum | strong_uptrend) & ~curve_inverted)
        regime[aggressive] = 2
        
        # Optional: Use forward returns for training validation
        if use_forward_looking:
            forward_return = returns.rolling(3).sum().shift(-3)
            # Override if forward returns are very strong/weak
            regime[forward_return > 0.10] = 2  # Strong future gains = aggressive
            regime[forward_return < -0.08] = 0  # Strong future losses = defensive
        
        return regime
    
    def create_features(self,
                       indicators: pd.DataFrame,
                       prices: pd.DataFrame,
                       returns: pd.DataFrame) -> pd.DataFrame:
        """Create features with MORE forward-looking indicators."""
        features = pd.DataFrame(index=indicators.index)
        
        # VIX features (current state + trend)
        if 'VIX' in indicators.columns:
            features['vix'] = indicators['VIX']
            features['vix_ma_20'] = indicators['VIX'].rolling(20).mean()
            features['vix_ma_60'] = indicators['VIX'].rolling(60).mean()
            features['vix_trend'] = indicators['VIX'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            features['vix_change'] = indicators['VIX'].pct_change()
            features['vix_zscore'] = (indicators['VIX'] - indicators['VIX'].rolling(60).mean()) / indicators['VIX'].rolling(60).std()
            features['vix_acceleration'] = features['vix_change'].diff()  # Rate of change
        
        # Yield spread features (forward-looking)
        if 'Yield_Spread' in indicators.columns:
            features['yield_spread'] = indicators['Yield_Spread']
            features['yield_spread_ma'] = indicators['Yield_Spread'].rolling(20).mean()
            features['yield_spread_trend'] = indicators['Yield_Spread'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            features['yield_inverted'] = (indicators['Yield_Spread'] < 0).astype(int)
            features['yield_steepening'] = indicators['Yield_Spread'].diff()  # Steepening/flattening
        
        # Interest rate features (forward-looking)
        if 'Interest_Rate' in indicators.columns:
            features['interest_rate'] = indicators['Interest_Rate']
            features['rate_change'] = indicators['Interest_Rate'].diff()
            features['rate_trend'] = indicators['Interest_Rate'].rolling(12).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # SPY features (momentum + trend)
        if 'SPY' in prices.columns:
            spy = prices['SPY']
            features['spy_return_1m'] = returns['SPY']
            features['spy_return_3m'] = returns['SPY'].rolling(3).sum()
            features['spy_return_6m'] = returns['SPY'].rolling(6).sum()
            features['spy_return_12m'] = returns['SPY'].rolling(12).sum()
            
            # Moving averages
            features['spy_ma_20'] = spy.rolling(20).mean()
            features['spy_ma_60'] = spy.rolling(60).mean()
            features['spy_ma_200'] = spy.rolling(200).mean()
            features['spy_vs_ma20'] = (spy - features['spy_ma_20']) / features['spy_ma_20']
            features['spy_vs_ma60'] = (spy - features['spy_ma_60']) / features['spy_ma_60']
            features['spy_vs_ma200'] = (spy - features['spy_ma_200']) / features['spy_ma_200']
            
            # Golden/Death cross
            features['spy_golden_cross'] = (features['spy_ma_60'] > features['spy_ma_200']).astype(int)
            
            # Volatility
            features['spy_vol'] = returns['SPY'].rolling(20).std()
            features['spy_vol_trend'] = features['spy_vol'].diff()
            
            # RSI
            delta = spy.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            features['spy_rsi'] = 100 - (100 / (1 + rs))
            
            # Momentum score
            features['spy_momentum_score'] = (
                features['spy_return_3m'] * 0.5 +
                features['spy_return_6m'] * 0.3 +
                features['spy_return_12m'] * 0.2
            )
        
        # TLT features
        if 'TLT' in returns.columns:
            features['tlt_return_1m'] = returns['TLT']
            features['tlt_return_3m'] = returns['TLT'].rolling(3).sum()
            features['tlt_vol'] = returns['TLT'].rolling(20).std()
        
        # Stock-bond correlation (regime indicator)
        if 'SPY' in returns.columns and 'TLT' in returns.columns:
            features['spy_tlt_corr'] = returns['SPY'].rolling(60).corr(returns['TLT'])
            features['spy_tlt_corr_change'] = features['spy_tlt_corr'].diff()
        
        # Gold features (safe haven)
        if 'GLD' in returns.columns:
            features['gld_return_3m'] = returns['GLD'].rolling(3).sum()
            features['gld_momentum'] = returns['GLD'].rolling(6).sum()
        
        # Bitcoin features
        if 'BTC-USD' in returns.columns:
            features['btc_return_3m'] = returns['BTC-USD'].rolling(3).sum()
            features['btc_vol'] = returns['BTC-USD'].rolling(20).std()
        
        # Cross-asset momentum (forward-looking)
        momentum_cols = []
        for col in ['SPY', 'TLT', 'GLD']:
            if col in returns.columns:
                momentum_cols.append(returns[col].rolling(3).sum())
        if momentum_cols:
            features['cross_asset_momentum'] = pd.concat(momentum_cols, axis=1).mean(axis=1)
        
        # Market breadth (using available assets)
        if 'SPY' in returns.columns:
            positive_returns = (returns > 0).sum(axis=1) / len(returns.columns)
            features['market_breadth'] = positive_returns
        
        return features.fillna(method='ffill').fillna(0)
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train multiple models for regime prediction."""
        print("\nTraining regime prediction models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        models = {}
        
        # 1. Gradient Boosting
        print("  - Gradient Boosting Classifier...")
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        models['gradient_boosting'] = gb_model
        
        # 2. Random Forest
        print("  - Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
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
            early_stopping=False
        )
        nn_model.fit(X_train_scaled, y_train)
        models['neural_network'] = nn_model
        
        # 4. Ensemble
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
    
    def predict_regime(self, X: pd.DataFrame, model_name: str = 'ensemble') -> pd.Series:
        """Predict regime using specified model."""
        X_scaled = self.scaler.transform(X)
        model = self.models[model_name]
        predictions = model.predict(X_scaled)
        return pd.Series(predictions, index=X.index)
    
    def optimize_allocations_for_regime(self,
                                       regime: int,
                                       returns: pd.DataFrame,
                                       lookback: int = 60) -> Dict:
        """
        Dynamically optimize allocations within a regime using recent data.
        
        Args:
            regime: Regime number (0, 1, 2)
            returns: Historical returns
            lookback: Lookback period for optimization
            
        Returns:
            Dictionary of optimized allocations
        """
        # Get base allocations for regime
        base_alloc = self.regime_allocations[regime]
        assets = list(base_alloc.keys())
        
        # Use recent returns for optimization
        recent_returns = returns[assets].tail(lookback)
        
        # Calculate covariance matrix
        cov_matrix = recent_returns.cov() * 12  # Annualized
        
        # Expected returns (simple momentum)
        expected_returns = recent_returns.mean() * 12  # Annualized
        
        # Regime-specific constraints
        if regime == 0:  # Defensive
            bounds = [(0.0, 0.25), (0.2, 0.5), (0.0, 0.0), (0.15, 0.4), (0.1, 0.3), (0.05, 0.2)]
            risk_aversion = 5.0  # High risk aversion
        elif regime == 1:  # Neutral
            bounds = [(0.3, 0.6), (0.15, 0.35), (0.0, 0.1), (0.05, 0.15), (0.05, 0.15), (0.05, 0.15)]
            risk_aversion = 2.0  # Moderate
        else:  # Aggressive
            bounds = [(0.5, 0.75), (0.05, 0.15), (0.05, 0.2), (0.0, 0.1), (0.0, 0.1), (0.0, 0.1)]
            risk_aversion = 1.0  # Low risk aversion
        
        # Optimization objective: maximize Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Negative Sharpe (for minimization)
            return -(portfolio_return - 0.02) / (portfolio_vol + 1e-6)
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Initial guess: base allocations
        x0 = np.array([base_alloc[asset] for asset in assets])
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimized_alloc = {asset: weight for asset, weight in zip(assets, result.x)}
        else:
            # Fall back to base allocations
            optimized_alloc = base_alloc
        
        return optimized_alloc
    
    def get_allocations_for_regime(self,
                                   regimes: pd.Series,
                                   returns: pd.DataFrame = None,
                                   use_dynamic_optimization: bool = True) -> pd.DataFrame:
        """
        Convert regime predictions to portfolio allocations.
        
        Args:
            regimes: Predicted regimes
            returns: Historical returns (for dynamic optimization)
            use_dynamic_optimization: If True, optimize within regimes
        """
        allocations = pd.DataFrame(index=regimes.index)
        
        # Initialize columns
        for asset in self.regime_allocations[0].keys():
            allocations[asset] = 0.0
        
        for idx in regimes.index:
            regime = regimes.loc[idx]
            
            if use_dynamic_optimization and returns is not None:
                # Get historical data up to this point
                hist_returns = returns.loc[:idx]
                if len(hist_returns) >= 60:
                    alloc = self.optimize_allocations_for_regime(regime, hist_returns, lookback=60)
                else:
                    alloc = self.regime_allocations[regime]
            else:
                alloc = self.regime_allocations[regime]
            
            for asset, weight in alloc.items():
                allocations.loc[idx, asset] = weight
        
        return allocations
