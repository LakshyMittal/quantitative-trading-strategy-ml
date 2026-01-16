"""
Regime detection using Hidden Markov Model (HMM).
Classifies market into 3 regimes: Uptrend (+1), Downtrend (-1), Sideways (0).
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """Hidden Markov Model for market regime detection."""
    
    def __init__(self, n_states: int = 3, n_iter: int = 100, random_state: int = 42):
        """
        Initialize HMM regime detector.
        
        Args:
            n_states: Number of regimes (typically 3)
            n_iter: Number of iterations for training
            random_state: Random seed
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(n_components=n_states, n_iter=n_iter, random_state=random_state)
        self.feature_columns = None
        self.is_fitted = False
        logger.info(f"Initialized HMM with {n_states} states")
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for HMM.
        
        Args:
            df: Input dataframe
            feature_cols: List of feature column names
            
        Returns:
            Tuple of (feature_array, feature_names)
        """
        X = df[feature_cols].values
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        # Standardize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        self.feature_columns = feature_cols
        logger.info(f"Prepared features: {len(feature_cols)} columns, shape {X.shape}")
        return X
    
    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        train_ratio: float = 0.7
    ) -> 'HMMRegimeDetector':
        """
        Fit HMM on training data.
        
        Args:
            df: Input dataframe
            feature_cols: Feature column names
            train_ratio: Proportion for training
            
        Returns:
            Self for chaining
        """
        # Prepare features
        X = self.prepare_features(df, feature_cols)
        
        # Train/test split
        split_idx = int(len(X) * train_ratio)
        X_train = X[:split_idx]
        
        # Fit model
        self.model.fit(X_train)
        self.is_fitted = True
        logger.info(f"Fitted HMM on {len(X_train)} samples")
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regimes for all samples.
        
        Args:
            df: Input dataframe
            
        Returns:
            Array of regime labels (+1, -1, 0)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.prepare_features(df, self.feature_columns)
        states = self.model.predict(X)
        
        # Map HMM states to trading regimes dynamically
        regimes = self._map_states_to_regimes(df, states)
        logger.info(f"Predicted regimes: {np.bincount(regimes)}")
        return regimes
    
    def _map_states_to_regimes(self, df: pd.DataFrame, states: np.ndarray) -> np.ndarray:
        """
        Map HMM states to trading regimes based on mean returns.
        States are mapped dynamically based on actual return characteristics.
        
        Args:
            df: Input dataframe with log returns
            states: HMM state labels (0, 1, 2)
            
        Returns:
            Regime labels: +1 (uptrend), -1 (downtrend), 0 (sideways)
        """
        # Calculate mean returns for each state
        state_returns = {}
        for state in range(self.n_states):
            mask = states == state
            if mask.sum() > 0 and "log_return" in df.columns:
                state_returns[state] = df.loc[mask, "log_return"].mean()
            else:
                state_returns[state] = 0
        
        # Sort states by mean return
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1], reverse=True)
        
        # Map: highest return → +1, lowest return → -1, middle → 0
        regime_map = {}
        if len(sorted_states) == 3:
            regime_map[sorted_states[0][0]] = 1   # Uptrend (highest return)
            regime_map[sorted_states[1][0]] = 0   # Sideways (middle return)
            regime_map[sorted_states[2][0]] = -1  # Downtrend (lowest return)
        elif len(sorted_states) == 2:
            regime_map[sorted_states[0][0]] = 1
            regime_map[sorted_states[1][0]] = -1
        else:
            regime_map = {0: 1, 1: -1, 2: 0}
        
        logger.info(f"State-to-regime mapping: {regime_map}")
        return np.array([regime_map.get(s, 0) for s in states])
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get HMM transition matrix as DataFrame.
        
        Returns:
            Transition matrix
        """
        trans_matrix = pd.DataFrame(
            self.model.transmat_,
            columns=[f"To_State_{i}" for i in range(self.n_states)],
            index=[f"From_State_{i}" for i in range(self.n_states)]
        )
        return trans_matrix
    
    def get_state_statistics(self, df: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
        """
        Get statistics for each HMM state.
        
        Args:
            df: Input dataframe with returns
            states: Predicted states
            
        Returns:
            Statistics dataframe
        """
        stats_list = []
        for state in range(self.n_states):
            mask = states == state
            if mask.sum() > 0:
                state_data = df[mask]
                stats = {
                    "State": state,
                    "Count": mask.sum(),
                    "Mean_Return": state_data.get("log_return", pd.Series([0])).mean(),
                    "Volatility": state_data.get("rolling_vol", pd.Series([0])).mean()
                }
                stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
