"""
Trading strategy implementation.
Includes EMA-based strategy with regime filter.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TradeSignal:
    """Represents a single trade signal."""
    
    def __init__(self, timestamp: pd.Timestamp, signal_type: int, price: float, confidence: float = 1.0):
        """
        Initialize trade signal.
        
        Args:
            timestamp: Time of signal
            signal_type: 1 (long), -1 (short), 0 (no signal)
            price: Price at signal
            confidence: Confidence score (0-1)
        """
        self.timestamp = timestamp
        self.signal_type = signal_type
        self.price = price
        self.confidence = confidence


class EMA5_15Strategy:
    """5/15 EMA crossover strategy with regime filter."""
    
    def __init__(self, regime_filter: bool = True):
        """
        Initialize strategy.
        
        Args:
            regime_filter: Whether to use regime filter
        """
        self.regime_filter = regime_filter
        self.signals = []
        logger.info(f"Initialized EMA5_15Strategy (regime_filter={regime_filter})")
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        ema_fast_col: str = "ema_fast",
        ema_slow_col: str = "ema_slow",
        regime_col: str = "market_regime"
    ) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            df: Input dataframe with EMA columns
            ema_fast_col: Column name for fast EMA
            ema_slow_col: Column name for slow EMA
            regime_col: Column name for regime (optional)
            
        Returns:
            DataFrame with signal column added
        """
        df = df.copy()
        df["signal"] = 0
        
        # Detect crossovers
        ema_diff = df[ema_fast_col] - df[ema_slow_col]
        ema_diff_prev = ema_diff.shift(1)
        
        # Bullish crossover (fast crosses above slow)
        bullish = (ema_diff > 0) & (ema_diff_prev <= 0)
        # Bearish crossover (fast crosses below slow)
        bearish = (ema_diff < 0) & (ema_diff_prev >= 0)
        
        # Apply regime filter if enabled
        if self.regime_filter and regime_col in df.columns:
            bullish = bullish & (df[regime_col] == "bull_trend_high_vol")
            bearish = bearish & (df[regime_col] == "bear_trend_high_vol")
        
        df.loc[bullish, "signal"] = 1   # Long signal
        df.loc[bearish, "signal"] = -1  # Short signal
        
        logger.info(f"Generated {(df['signal'] != 0).sum()} signals")
        return df
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, signal_col: str = "signal") -> pd.DataFrame:
        """
        Calculate strategy returns.
        
        Args:
            df: Input dataframe with signal column
            signal_col: Signal column name
            
        Returns:
            DataFrame with return columns
        """
        df = df.copy()
        
        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        
        # Shift signal to avoid look-ahead bias
        df["signal_shifted"] = df[signal_col].shift(1)
        
        # Strategy returns
        df["strategy_return"] = df["signal_shifted"] * df["log_return"]
        
        return df


class MLEnhancedStrategy(EMA5_15Strategy):
    """EMA strategy enhanced with ML trade filter."""
    
    def __init__(self, ml_model, confidence_threshold: float = 0.5):
        """
        Initialize ML-enhanced strategy.
        
        Args:
            ml_model: Fitted ML model with predict_proba method
            confidence_threshold: Minimum confidence to take trade
        """
        super().__init__(regime_filter=True)
        self.ml_model = ml_model
        self.confidence_threshold = confidence_threshold
        logger.info(f"Initialized MLEnhancedStrategy (threshold={confidence_threshold})")
    
    def filter_signals_with_ml(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        signal_col: str = "signal"
    ) -> pd.DataFrame:
        """
        Filter signals using ML model.
        
        Args:
            df: Input dataframe with signals and features
            feature_cols: List of feature columns
            signal_col: Signal column name
            
        Returns:
            DataFrame with ML-filtered signals
        """
        df = df.copy()
        df["ml_confidence"] = 0.0
        df["ml_signal"] = 0
        
        # Get predictions for rows with signals
        signal_mask = df[signal_col] != 0
        if signal_mask.sum() > 0:
            X = df.loc[signal_mask, feature_cols].values
            probs = self.ml_model.predict_proba(X)
            
            df.loc[signal_mask, "ml_confidence"] = probs[:, 1]
            
            # Filter by confidence
            confident_mask = signal_mask & (df["ml_confidence"] >= self.confidence_threshold)
            df.loc[confident_mask, "ml_signal"] = df.loc[confident_mask, signal_col]
        
        logger.info(f"ML filtered signals: {(df['ml_signal'] != 0).sum()} / {(df[signal_col] != 0).sum()}")
        return df