"""
Strategy Parameter Optimization
Tests different EMA combinations to find the best Sharpe Ratio.
"""

import pandas as pd
import numpy as np
import sys
import itertools
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy import EMA5_15Strategy
from backtest import Backtester

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = PROJECT_ROOT / "results"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

print("="*70)
print("STRATEGY PARAMETER OPTIMIZATION")
print("="*70)

# Load data
features_path = DATA_PATH / "features_1y.csv"
if not features_path.exists():
    print("Error: Data file not found. Run run_data_processing.py first.")
    sys.exit(1)

df_raw = pd.read_csv(features_path, parse_dates=["date"])
print(f"Loaded {len(df_raw)} rows")

# Define parameter grid
fast_emas = [5, 10, 20]
slow_emas = [15, 30, 50, 100]

results = []

print(f"\nTesting {len(fast_emas) * len(slow_emas)} combinations...")
print("-" * 60)
print(f"{'Fast':<6} {'Slow':<6} {'Return':<10} {'Sharpe':<10} {'Trades':<8}")
print("-" * 60)

for fast, slow in itertools.product(fast_emas, slow_emas):
    if fast >= slow:
        continue
        
    # Calculate temporary EMAs
    df = df_raw.copy()
    col_fast = f"ema_{fast}"
    col_slow = f"ema_{slow}"
    
    df[col_fast] = df["close"].ewm(span=fast, adjust=False).mean()
    df[col_slow] = df["close"].ewm(span=slow, adjust=False).mean()
    
    # Run Strategy (Regime Filtered)
    # Note: We use the existing 'regime' column if it exists, otherwise it defaults to no filter
    # or we can disable it to find the best raw EMAs first.
    strategy = EMA5_15Strategy(regime_filter=True)
    
    # If regime column is missing, add a dummy one or disable filter
    if "regime" not in df.columns:
        df["regime"] = 0 # Sideways default
        
    df = strategy.generate_signals(df, col_fast, col_slow, "regime")
    df = strategy.calculate_returns(df, "signal")
    
    # Backtest
    backtester = Backtester(commission=0.001)
    metrics = backtester.run(df, "signal", "close")
    
    res = {
        "Fast": fast,
        "Slow": slow,
        "Return": metrics["strategy"]["total_return"],
        "Sharpe": metrics["sharpe_ratio"],
        "Trades": len(df[df["signal"] != 0])
    }
    results.append(res)
    print(f"{fast:<6} {slow:<6} {res['Return']:<10.4f} {res['Sharpe']:<10.2f} {res['Trades']:<8}")

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH / "optimization_results.csv", index=False)
print(f"\nResults saved to {RESULTS_PATH / 'optimization_results.csv'}")

print("-" * 60)
best_result = max(results, key=lambda x: x["Sharpe"])
print(f"\nBEST PARAMETERS: EMA {best_result['Fast']} / {best_result['Slow']}")
print(f"Sharpe Ratio: {best_result['Sharpe']:.2f}")
print(f"Total Return: {best_result['Return']:.2%}")
