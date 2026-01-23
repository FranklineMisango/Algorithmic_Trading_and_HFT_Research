"""
Model Trainer
=============

Comprehensive model training with proper cross-validation, evaluation, and SHAP analysis.
"""

import numpy as np
import pandas as pd
import logging
import joblib
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

# Scikit-learn
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, mean_squared_error, mean_absolute_error, r2_score
)

# ML Libraries
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelTrainer:
    """
    Train and evaluate ML models with proper validation and metrics.
    
    Uses time-series cross-validation to prevent data leakage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("FuturesPrediction.ModelTrainer")
        self.model_config = config.get("models", {})
        self.cv_config = config.get("cross_validation", {})
        self.eval_config = config.get("evaluation", {})
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        
        # Setup MLflow if enabled
        self.mlflow_enabled = config.get("mlflow", {}).get("enabled", False)
        if self.mlflow_enabled:
            try:
                import mlflow
                self.mlflow = mlflow
                mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "futures_prediction"))
                self.logger.info("MLflow tracking enabled")
            except ImportError:
                self.logger.warning("MLflow not available. Disabling tracking.")
                self.mlflow_enabled = False
    
    def train_xgboost_classifier(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: Optional[pd.DataFrame] = None,
                                 y_val: Optional[pd.Series] = None) -> xgb.XGBClassifier:
        """
        Train XGBoost classifier with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Best trained model
        """
        self.logger.info("Training XGBoost Classifier...")
        
        xgb_config = self.model_config.get("xgboost", {}).get("classifier", {})
        
        # Handle class imbalance
        scale_pos_weight = None
        if xgb_config.get("use_scale_pos_weight", True):
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            self.logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
        
        # Base model
        base_model = xgb.XGBClassifier(
            objective=xgb_config.get("objective", "binary:logistic"),
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.get("random_seed", 42),
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Parameter grid
        param_grid = {
            'max_depth': xgb_config.get("max_depth", [3, 5]),
            'learning_rate': xgb_config.get("learning_rate", [0.01, 0.1]),
            'n_estimators': xgb_config.get("n_estimators", [100, 200])
        }
        
        # Cross-validation
        cv = self._get_cv_splitter()
        scoring = self.cv_config.get("scoring_clf", "roc_auc")
        
        grid_search = GridSearchCV(
            base_model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        self.logger.info(f"Best params: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.models['xgb_classifier'] = best_model
        return best_model
    
    def train_xgboost_regressor(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
        """
        Train XGBoost regressor with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Best trained model
        """
        self.logger.info("Training XGBoost Regressor...")
        
        xgb_config = self.model_config.get("xgboost", {}).get("regressor", {})
        
        base_model = xgb.XGBRegressor(
            objective=xgb_config.get("objective", "reg:squarederror"),
            random_state=self.config.get("random_seed", 42)
        )
        
        param_grid = {
            'max_depth': xgb_config.get("max_depth", [3, 5]),
            'learning_rate': xgb_config.get("learning_rate", [0.01, 0.1]),
            'n_estimators': xgb_config.get("n_estimators", [100, 200])
        }
        
        cv = self._get_cv_splitter()
        scoring = self.cv_config.get("scoring_reg", "neg_mean_squared_error")
        
        grid_search = GridSearchCV(
            base_model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        self.logger.info(f"Best params: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.models['xgb_regressor'] = best_model
        return best_model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train Random Forest classifier."""
        self.logger.info("Training Random Forest Classifier...")
        
        rf_config = self.model_config.get("random_forest", {})
        
        model = RandomForestClassifier(
            n_estimators=rf_config.get("n_estimators", 100),
            class_weight=rf_config.get("class_weight", "balanced"),
            random_state=self.config.get("random_seed", 42),
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> VotingClassifier:
        """Train voting ensemble of all available classifiers."""
        self.logger.info("Training Voting Ensemble...")
        
        estimators = []
        
        if 'xgb_classifier' in self.models:
            estimators.append(('xgb', self.models['xgb_classifier']))
        
        if 'random_forest' in self.models:
            estimators.append(('rf', self.models['random_forest']))
        
        if len(estimators) < 2:
            self.logger.warning("Not enough models for ensemble. Training individual models first.")
            return None
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        self.models['ensemble'] = ensemble
        return ensemble
    
    def build_lstm_model(self, input_shape: Tuple, lstm_config: Optional[Dict] = None) -> keras.Model:
        """
        Build LSTM model for time series regression.
        
        Args:
            input_shape: Input shape (sequence_length, n_features)
            lstm_config: LSTM configuration. If None, uses default config.
            
        Returns:
            Compiled LSTM model
        """
        if lstm_config is None:
            lstm_config = self.model_config.get("lstm", {})
        
        units = lstm_config.get("units", [50])[0]  # Use first value as default
        dropout_rate = lstm_config.get("dropout_rate", [0.2])[0]
        learning_rate = lstm_config.get("learning_rate", [0.001])[0]
        
        # Enable mixed precision if configured
        if lstm_config.get("use_mixed_precision", False):
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                self.logger.info("Mixed precision enabled for LSTM")
            except Exception as e:
                self.logger.warning(f"Could not enable mixed precision: {e}")
        
        model = keras.Sequential([
            layers.LSTM(units, input_shape=input_shape, return_sequences=False),
            layers.Dropout(dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1)
        ], name="LSTM_Regressor")
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def evaluate_classification(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                               model_name: str = "Model") -> Dict[str, float]:
        """
        Comprehensive classification evaluation with multiple metrics.
        
        Args:
            model: Trained classifier
            X_test: Test features
            y_test: Test labels
            model_name: Name for logging
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Evaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        self.logger.info(f"{model_name} Metrics:")
        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value:.4f}")
        
        # Classification report
        self.logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_regression(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                           model_name: str = "Model") -> Dict[str, float]:
        """
        Comprehensive regression evaluation.
        
        Args:
            model: Trained regressor
            X_test: Test features
            y_test: Test target
            model_name: Name for logging
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Evaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'directional_accuracy': self._directional_accuracy(y_test, y_pred)
        }
        
        self.logger.info(f"{model_name} Metrics:")
        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value:.6f}")
        
        self.results[model_name] = metrics
        return metrics
    
    def explain_model_shap(self, model, X_test: pd.DataFrame, model_name: str = "Model",
                          sample_size: Optional[int] = None):
        """
        Generate SHAP explanations for model interpretability.
        
        Args:
            model: Trained model
            X_test: Test features
            model_name: Model name for logging
            sample_size: Number of samples for SHAP. If None, uses config.
        """
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available. Skipping explanations.")
            return
        
        if not self.eval_config.get("use_shap", False):
            return
        
        if sample_size is None:
            sample_size = self.eval_config.get("shap_sample_size", 100)
        
        self.logger.info(f"Generating SHAP explanations for {model_name}...")
        
        # Sample data to avoid memory issues
        X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
        
        try:
            if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model.predict, X_sample)
            
            shap_values = explainer(X_sample)
            
            # Summary plot
            shap.summary_plot(shap_values, X_sample, show=False)
            
            self.logger.info("SHAP explanations generated successfully")
        except Exception as e:
            self.logger.warning(f"SHAP explanation failed: {e}")
    
    def save_model(self, model, model_name: str, directory: Optional[str] = None):
        """
        Save trained model to disk.
        
        Args:
            model: Model to save
            model_name: Name for the saved file
            directory: Save directory. If None, uses config.
        """
        if directory is None:
            directory = self.config.get("model_saving", {}).get("directory", "models")
        
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = save_dir / f"{model_name}.pkl"
        joblib.dump(model, file_path)
        
        self.logger.info(f"Model saved: {file_path}")
    
    def _get_cv_splitter(self):
        """Get cross-validation splitter based on config."""
        method = self.cv_config.get("method", "TimeSeriesSplit")
        n_splits = self.cv_config.get("n_splits", 5)
        
        if method == "TimeSeriesSplit":
            return TimeSeriesSplit(n_splits=n_splits)
        else:
            from sklearn.model_selection import KFold
            return KFold(n_splits=n_splits, shuffle=False)
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy."""
        return np.mean((np.sign(y_true) == np.sign(y_pred)).astype(int))
