"""
HMM Regime Detection Module
"""
import numpy as np
import pandas as pd
import logging

try:
    from hmmlearn import hmm
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

logger = logging.getLogger(__name__)

class HMMRegimeDetector:
    """
    HMM-based regime detection wrapper.
    """
    def __init__(self, n_states=3, n_iter=100, random_state=42):
        if not HAS_HMM:
            raise ImportError("hmmlearn not installed. Run: pip install hmmlearn")
            
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type="full", 
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )
        self.is_fitted = False
        self.feature_cols = None
        
    def fit(self, df: pd.DataFrame, feature_cols: list, train_ratio: float = 0.7):
        """Fit the HMM model."""
        self.feature_cols = feature_cols
        
        # Prepare data - fill NaNs to avoid errors
        X = df[feature_cols].copy()
        X = X.fillna(method='bfill').fillna(method='ffill').fillna(0)
        X_values = X.values
        
        # Split train/test
        train_size = int(len(X_values) * train_ratio)
        X_train = X_values[:train_size]
        
        self.model.fit(X_train)
        self.is_fitted = True
        logger.info(f"HMM fitted with {self.n_states} states")
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regimes."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        X = df[self.feature_cols].copy()
        X = X.fillna(method='bfill').fillna(method='ffill').fillna(0)
        return self.model.predict(X.values)
        
    def get_transition_matrix(self) -> pd.DataFrame:
        """Get state transition matrix."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame(self.model.transmat_)