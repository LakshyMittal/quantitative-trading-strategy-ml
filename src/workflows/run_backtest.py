"""
Complete Backtesting Pipeline
Tests baseline, regime-filtered, and ML-enhanced strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy import EMA5_15Strategy
from backtest import Backtester

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = PROJECT_ROOT / "results"
PLOTS_PATH = PROJECT_ROOT / "plots"
MODELS_PATH = PROJECT_ROOT / "models"

for path in [RESULTS_PATH, PLOTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print("="*70)
print("COMPLETE BACKTESTING PIPELINE")
print("="*70)

print("\n[1/5] Loading data...")
features_path = DATA_PATH / "features_1y.csv"
df = pd.read_csv(features_path, parse_dates=["date"])
print(f" Loaded {len(df)} samples")

print("\n[2/5] Testing baseline strategy (EMA without filter)...")
strategy_baseline = EMA5_15Strategy(regime_filter=False)
df_baseline = strategy_baseline.generate_signals(df.copy(), "ema_fast", "ema_slow")
df_baseline = strategy_baseline.calculate_returns(df_baseline, "signal")

backtester_baseline = Backtester(commission=0.001)
metrics_baseline = backtester_baseline.run(df_baseline, "signal", "close")

print(f"   Total Return: {metrics_baseline['strategy']['total_return']:.4f}")
print(f"   Sharpe Ratio: {metrics_baseline['sharpe_ratio']:.2f}")
print(f"   Max Drawdown: {metrics_baseline['max_drawdown']:.2%}")

print("\n[3/5] Testing regime-filtered strategy...")

try:
    regime_path = DATA_PATH / "features_with_regime.csv"
    df_regime = pd.read_csv(regime_path, parse_dates=["date"])
except:
    print("   Regime data not found, using basic regime filter")
    df_regime = df.copy()

strategy_regime = EMA5_15Strategy(regime_filter=True)
df_regime = strategy_regime.generate_signals(df_regime.copy(), "ema_fast", "ema_slow", "regime")
df_regime = strategy_regime.calculate_returns(df_regime, "signal")

backtester_regime = Backtester(commission=0.001)
metrics_regime = backtester_regime.run(df_regime, "signal", "close")

print(f"   Total Return: {metrics_regime['strategy']['total_return']:.4f}")
print(f"   Sharpe Ratio: {metrics_regime['sharpe_ratio']:.2f}")
print(f"   Max Drawdown: {metrics_regime['max_drawdown']:.2%}")

if metrics_baseline['sharpe_ratio'] < 0:
    improvement = (metrics_regime['sharpe_ratio'] - metrics_baseline['sharpe_ratio']) / abs(metrics_baseline['sharpe_ratio']) * 100
else:
    improvement = ((metrics_regime['sharpe_ratio'] / metrics_baseline['sharpe_ratio']) - 1) * 100

print(f"   Sharpe improvement: {improvement:.1f}%")

print("\n[4/5] Testing ML-enhanced strategy (XGBoost)...")

has_ml = False
try:
    with open(MODELS_PATH / "xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(MODELS_PATH / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    
    df_ml = df_baseline.copy()
    
    X_features = df_ml[feature_columns].fillna(0).values
    df_ml["ml_confidence"] = xgb_model.predict_proba(X_features)[:, 1]
    
    df_ml["signal_filtered"] = 0
    ml_signal_mask = (df_ml["signal"] != 0) & (df_ml["ml_confidence"] >= 0.6)
    df_ml.loc[ml_signal_mask, "signal_filtered"] = df_ml.loc[ml_signal_mask, "signal"]
    
    df_ml["log_return"] = np.log(df_ml["close"] / df_ml["close"].shift(1))
    df_ml["signal_shifted"] = df_ml["signal_filtered"].shift(1)
    df_ml["strategy_return"] = df_ml["signal_shifted"] * df_ml["log_return"]
    
    backtester_ml = Backtester(commission=0.001)
    metrics_ml = backtester_ml.run(df_ml, "signal_filtered", "close")
    
    print(f"   Total Return: {metrics_ml['strategy']['total_return']:.4f}")
    print(f"   Sharpe Ratio: {metrics_ml['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics_ml['max_drawdown']:.2%}")
    
    if metrics_baseline['sharpe_ratio'] < 0:
        ml_improvement = (metrics_ml['sharpe_ratio'] - metrics_baseline['sharpe_ratio']) / abs(metrics_baseline['sharpe_ratio']) * 100
    else:
        ml_improvement = ((metrics_ml['sharpe_ratio'] / metrics_baseline['sharpe_ratio']) - 1) * 100
        
    print(f"   Sharpe improvement: {ml_improvement:.1f}%")
    
    has_ml = True
except Exception as e:
    print(f"   ML model not available: {e}")

print("\n[5/5] Saving results...")

comparison_data = {
    "Baseline": [
        metrics_baseline["strategy"]["total_return"],
        metrics_baseline["strategy"]["annual_return"],
        metrics_baseline["strategy"]["annual_volatility"],
        metrics_baseline["sharpe_ratio"],
        metrics_baseline["sortino_ratio"],
        metrics_baseline["max_drawdown"],
        metrics_baseline["calmar_ratio"]
    ],
    "Regime-Filtered": [
        metrics_regime["strategy"]["total_return"],
        metrics_regime["strategy"]["annual_return"],
        metrics_regime["strategy"]["annual_volatility"],
        metrics_regime["sharpe_ratio"],
        metrics_regime["sortino_ratio"],
        metrics_regime["max_drawdown"],
        metrics_regime["calmar_ratio"]
    ]
}

if has_ml:
    comparison_data["ML-Enhanced"] = [
        metrics_ml["strategy"]["total_return"],
        metrics_ml["strategy"]["annual_return"],
        metrics_ml["strategy"]["annual_volatility"],
        metrics_ml["sharpe_ratio"],
        metrics_ml["sortino_ratio"],
        metrics_ml["max_drawdown"],
        metrics_ml["calmar_ratio"]
    ]

comparison_df = pd.DataFrame(
    comparison_data,
    index=["Total Return", "Annual Return", "Annual Volatility", "Sharpe Ratio", 
           "Sortino Ratio", "Max Drawdown", "Calmar Ratio"]
)

print("\n" + comparison_df.round(4).to_string())

comparison_df.to_csv(RESULTS_PATH / "strategy_comparison.csv")
print("\n Results saved")

print("\n[BONUS] Creating visualizations...")

fig, ax = plt.subplots(figsize=(15, 6))

df_baseline["cum_return_baseline"] = df_baseline["strategy_return"].cumsum()
df_baseline["cum_return_market"] = df_baseline["log_return"].cumsum()
df_regime["cum_return_regime"] = df_regime["strategy_return"].cumsum()

ax.plot(df_baseline["date"], df_baseline["cum_return_market"], label="Market", 
        linewidth=2, color="gray", linestyle="--")
ax.plot(df_baseline["date"], df_baseline["cum_return_baseline"], label="Baseline EMA", 
        linewidth=2, color="#1f77b4")
ax.plot(df_regime["date"], df_regime["cum_return_regime"], label="Regime-Filtered", 
        linewidth=2, color="#2ca02c")

if has_ml:
    df_ml["cum_return_ml"] = df_ml["strategy_return"].cumsum()
    ax.plot(df_ml["date"], df_ml["cum_return_ml"], label="ML-Enhanced (XGBoost)", 
            linewidth=2, color="#ff7f0e")

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Cumulative Log Return", fontsize=12)
ax.set_title("Strategy Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "07_cumulative_returns_comparison.png", dpi=300, bbox_inches="tight")
print("   Cumulative returns plot saved")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))

strategies = ["Baseline", "Regime-Filtered"]
sharpes = [metrics_baseline["sharpe_ratio"], metrics_regime["sharpe_ratio"]]
colors = ["#1f77b4", "#2ca02c"]

if has_ml:
    strategies.append("ML-Enhanced")
    sharpes.append(metrics_ml["sharpe_ratio"])
    colors.append("#ff7f0e")

bars = ax.bar(strategies, sharpes, color=colors)

for bar, sharpe in zip(bars, sharpes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{sharpe:.2f}",
            ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.set_ylabel("Sharpe Ratio", fontsize=12)
ax.set_title("Sharpe Ratio Comparison", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS_PATH / "08_sharpe_ratio_comparison.png", dpi=300, bbox_inches="tight")
print("   Sharpe ratio comparison saved")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))

drawdowns = [metrics_baseline["max_drawdown"], metrics_regime["max_drawdown"]]

if has_ml:
    drawdowns.append(metrics_ml["max_drawdown"])

bars = ax.bar(strategies, drawdowns, color=colors)

for bar, dd in zip(bars, drawdowns):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{dd:.1%}",
            ha="center", va="top" if height < 0 else "bottom", fontweight="bold", fontsize=11)

ax.set_ylabel("Maximum Drawdown", fontsize=12)
ax.set_title("Maximum Drawdown Comparison", fontsize=14, fontweight="bold")
ax.set_ylim([min(drawdowns) * 1.2, 0])
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS_PATH / "09_drawdown_comparison.png", dpi=300, bbox_inches="tight")
print("   Drawdown comparison saved")
plt.close()

print("\n" + "="*70)
print(" BACKTESTING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nOutputs saved to:")
print(f"  Results: {RESULTS_PATH}")
print(f"  Plots: {PLOTS_PATH}")
