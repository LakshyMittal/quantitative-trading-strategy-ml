"""
Data Processing Pipeline
Downloads market data and generates technical features.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Ensure directory exists
DATA_PATH.mkdir(parents=True, exist_ok=True)

print("="*70)
print("DATA PROCESSING PIPELINE")
print("="*70)

print("\n[1/4] Downloading data (NIFTY 50)...")
# Using NIFTY 50 symbol
ticker = "^NSEI" 
try:
    # Download 2 years to ensure enough data for 1y analysis after rolling windows
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    
    # Handle multi-index columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "date", "close": "close", "adj close": "adj_close"})
    
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
        
    print(f" Downloaded {len(df)} rows")
    
except Exception as e:
    print(f" Error downloading data: {e}")
    print(" Please ensure you have yfinance installed: pip install yfinance")
    sys.exit(1)

print("\n[2/4] Calculating Technical Indicators...")

# 1. EMAs
df["ema_fast"] = df["close"].ewm(span=5, adjust=False).mean()
df["ema_slow"] = df["close"].ewm(span=15, adjust=False).mean()
df["ema_spread"] = df["ema_fast"] - df["ema_slow"]

# 2. Volatility
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["rolling_vol"] = df["log_return"].rolling(window=20).std() * np.sqrt(252)

# 3. High Volatility Regime (Binary)
# Define high volatility as being above the 75th percentile of the rolling volatility
vol_threshold = df["rolling_vol"].quantile(0.75)
df["high_vol"] = (df["rolling_vol"] > vol_threshold).astype(int)

print(f" Features calculated: ema_fast, ema_slow, rolling_vol, high_vol")

print("\n[3/4] Cleaning data...")
# Drop NaN values created by rolling windows
initial_len = len(df)
df = df.dropna()

# Filter to last 1 year
one_year_ago = df["date"].max() - pd.DateOffset(years=1)
df_1y = df[df["date"] >= one_year_ago].copy()

print("\n[4/4] Saving data...")
output_path = DATA_PATH / "features_1y.csv"
df_1y.to_csv(output_path, index=False)
print(f" Saved to {output_path}")

print("\n" + "="*70)
print(" DATA PROCESSING COMPLETED SUCCESSFULLY!")
print("="*70)