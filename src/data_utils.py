"""
Data utilities for quantitative trading strategy.
Handles data loading, validation, and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate datasets from processed data folder."""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to processed data directory
        """
        self.data_path = Path(data_path)
        self._validate_path()
    
    def _validate_path(self) -> None:
        """Validate that data path exists."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        logger.info(f"Data path validated: {self.data_path}")
    
    def load_features(self, filename: str = "features_1y.csv") -> pd.DataFrame:
        """Load engineered features dataset."""
        filepath = self.data_path / filename
        df = pd.read_csv(filepath, parse_dates=["date"])
        logger.info(f"Loaded {filename}: shape {df.shape}")
        return df
    
    def load_spot_features(self, filename: str = "spot_features_1y.csv") -> pd.DataFrame:
        """Load spot data with engineered features."""
        filepath = self.data_path / filename
        df = pd.read_csv(filepath, parse_dates=["date"])
        logger.info(f"Loaded {filename}: shape {df.shape}")
        return df
    
    @staticmethod
    def train_test_split(
        df: pd.DataFrame, 
        train_ratio: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (time-based).
        
        Args:
            df: Input dataframe
            train_ratio: Proportion for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        logger.info(f"Split data: train {len(train_df)}, test {len(test_df)}")
        return train_df, test_df


class DataValidator:
    """Validate data quality and handle missing values."""
    
    @staticmethod
    def check_missing(df: pd.DataFrame) -> dict:
        """Check for missing values in dataframe."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        result = {col: pct for col, pct in missing_pct.items() if pct > 0}
        if result:
            logger.warning(f"Missing values found: {result}")
        return result
    
    @staticmethod
    def handle_missing(df: pd.DataFrame, method: str = "dropna") -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            df: Input dataframe
            method: 'dropna' or 'forward_fill'
            
        Returns:
            Cleaned dataframe
        """
        if method == "dropna":
            df_clean = df.dropna()
        elif method == "forward_fill":
            df_clean = df.fillna(method="ffill").dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        logger.info(f"Handled missing values: {len(df)} -> {len(df_clean)} rows")
        return df_clean


def align_dataframes(
    spot_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    on: str = "date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align spot and futures dataframes on timestamp.
    
    Args:
        spot_df: Spot price dataframe
        futures_df: Futures price dataframe
        on: Column to align on
        
    Returns:
        Tuple of aligned (spot_df, futures_df)
    """
    # Inner join on date
    merged = spot_df.merge(futures_df, on=on, how="inner", suffixes=("_spot", "_fut"))
    
    # Extract aligned versions
    spot_cols = ["date", "close_spot", "ema_fast_spot", "ema_slow_spot"]
    futures_cols = ["date", "close_fut", "ema_fast_fut", "ema_slow_fut"]
    
    logger.info(f"Aligned dataframes: {len(spot_df)} + {len(futures_df)} -> {len(merged)}")
    return merged