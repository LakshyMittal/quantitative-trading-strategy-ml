"""
Machine Learning models for trade filtering.
Includes Logistic Regression, XGBoost, and LSTM implementations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)


class MLTradeFilterBase:
    """Base class for ML trade filter models."""
    
    def __init__(self, name: str):
        """Initialize base class."""
        self.name = name
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        logger.info(f"Initialized {name}")
    
    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input data."""
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        if X.shape[0] == 0:
            raise ValueError("Empty dataset provided")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLTradeFilterBase':
        """Fit model on training data."""
        self._validate_data(X, y)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"{self.name} fitted on {len(X)} samples")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_proba)
        }
        logger.info(f"{self.name} metrics: {metrics}")
        return metrics


class LogisticRegressionFilter(MLTradeFilterBase):
    """Logistic Regression trade filter."""
    
    def __init__(self, max_iter: int = 1000):
        """Initialize Logistic Regression."""
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(max_iter=max_iter, random_state=42)


class XGBoostFilter(MLTradeFilterBase):
    """XGBoost trade filter."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 5):
        """Initialize XGBoost."""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        super().__init__("XGBoost")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbosity=0
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostFilter':
        """Fit XGBoost with early stopping."""
        self._validate_data(X, y)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"{self.name} fitted on {len(X)} samples")
        return self
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        importance = self.model.feature_importances_
        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        return df_importance


class LSTMFilter:
    """LSTM-based trade filter for sequence prediction."""
    
    def __init__(self, sequence_length: int = 10, lstm_units: int = 64, dropout: float = 0.2):
        """
        Initialize LSTM filter.
        
        Args:
            sequence_length: Number of time steps
            lstm_units: Number of LSTM units
            dropout: Dropout rate
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        logger.info(f"Initialized LSTM (seq_len={sequence_length}, units={lstm_units})")
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from flat data.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        for i in range(len(X) - self.sequence_length):
            seq = X[i:i + self.sequence_length]
            sequences.append(seq)
            targets.append(y[i + self.sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def build_model(self, input_features: int) -> None:
        """
        Build LSTM model.
        
        Args:
            input_features: Number of input features
        """
        try:
            from tensorflow import keras
            layers = keras.layers
            
            self.model = keras.Sequential([
                layers.LSTM(self.lstm_units, activation='relu', input_shape=(self.sequence_length, input_features)),
                layers.Dropout(self.dropout),
                layers.Dense(32, activation='relu'),
                layers.Dropout(self.dropout),
                layers.Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            logger.info("LSTM model built successfully")
        except ImportError as e:
            logger.warning(f"TensorFlow not available: {e}. LSTM will use sklearn fallback.")
            self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32) -> None:
        """
        Fit LSTM model.
        
        Args:
            X: Feature array
            y: Target array
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        self.model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
        self.is_fitted = True
        logger.info(f"LSTM fitted on {len(X_seq)} sequences")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using LSTM.
        
        Args:
            X: Feature array
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model is None:
            raise ValueError("LSTM model not available. TensorFlow may not be installed.")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X)))
        
        predictions = self.model.predict(X_seq, verbose=0)
        return (predictions > 0.5).astype(int).flatten()