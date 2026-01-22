"""
HMM Regime Detection Pipeline
Detects market regimes using Hidden Markov Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from regime import HMMRegimeDetector

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = PROJECT_ROOT / "results"
PLOTS_PATH = PROJECT_ROOT / "plots"

for path in [RESULTS_PATH, PLOTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print("="*70)
print("HMM REGIME DETECTION PIPELINE")
print("="*70)

print("\n[1/5] Loading data...")
features_path = DATA_PATH / "features_1y.csv"
df = pd.read_csv(features_path, parse_dates=["date"])
df["log_return"] = np.log(df["close"] / df["close"].shift(1))

# Ensure ema_spread exists
if "ema_spread" not in df.columns and "ema_fast" in df.columns and "ema_slow" in df.columns:
    df["ema_spread"] = df["ema_fast"] - df["ema_slow"]

# Validate features exist before proceeding
if "ema_spread" not in df.columns:
    print("Error: 'ema_spread' could not be calculated. Check if 'ema_fast' and 'ema_slow' exist in input data.")
    sys.exit(1)

print(f" Loaded {len(df)} samples")
print(f" Date range: {df['date'].min()} to {df['date'].max()}")

print("\n[2/5] Preparing HMM features...")
hmm_features = ["log_return", "rolling_vol", "ema_spread", "high_vol"]
print(f" HMM features: {hmm_features}")

print("\n[3/5] Training HMM (3 states)...")
hmm = HMMRegimeDetector(n_states=3, n_iter=100, random_state=42)
hmm.fit(df, hmm_features, train_ratio=0.7)
print(" HMM model trained on 70% of data")

print("\n[4/5] Predicting regimes for full dataset...")
regimes = hmm.predict(df)
df["regime"] = regimes

regime_names = {1: "Uptrend", -1: "Downtrend", 0: "Sideways"}
df["regime_name"] = df["regime"].map(regime_names)

print("\nRegime distribution:")
print(df["regime_name"].value_counts())
print(f"\nPercentages:")
print(df["regime_name"].value_counts(normalize=True) * 100)

print("\n[5/5] Calculating regime statistics...")

regime_stats = []
for regime in [1, -1, 0]:
    mask = df["regime"] == regime
    if mask.sum() > 0:
        regime_data = df[mask]
        volatility = regime_data["log_return"].std()
        mean_return = regime_data["log_return"].mean()
        sharpe = (mean_return / volatility * np.sqrt(252*78)) if volatility > 0 else 0
        
        stats = {
            "Regime": regime_names[regime],
            "Count": mask.sum(),
            "Percentage": f"{(mask.sum() / len(df) * 100):.1f}%",
            "Avg_Return": f"{mean_return:.6f}",
            "Volatility": f"{volatility:.6f}",
            "Sharpe_Ratio": f"{sharpe:.2f}",
            "Avg_EMA_Spread": f"{regime_data['ema_spread'].mean():.2f}"
        }
        regime_stats.append(stats)

regime_stats_df = pd.DataFrame(regime_stats)
print("\n", regime_stats_df)

regime_stats_df.to_csv(RESULTS_PATH / "regime_statistics.csv", index=False)
print("\n Regime statistics saved")

print("\nGenerating transition matrix...")
trans_matrix = hmm.get_transition_matrix()
print("\n", trans_matrix.round(4))

trans_matrix.to_csv(RESULTS_PATH / "regime_transition_matrix.csv")

output_path = DATA_PATH / "features_with_regime.csv"
df.to_csv(output_path, index=False)
print(f"\n Data with regimes saved to: {output_path}")

print("\n[BONUS] Creating visualizations...")

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df["date"], df["close"], label="NIFTY 50", linewidth=1.5, color="black", zorder=1)

colors = {1: "#2ca02c", -1: "#d62728", 0: "#ff7f0e"}
for regime, color in colors.items():
    mask = df["regime"] == regime
    ax.scatter(df[mask]["date"], df[mask]["close"], c=color, s=15, alpha=0.4, 
               label=regime_names[regime], zorder=2)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Close Price", fontsize=12)
ax.set_title("NIFTY 50 Price with HMM Regimes", fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "03_price_with_regimes.png", dpi=300, bbox_inches="tight")
print("   Price chart with regimes saved")
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(trans_matrix, annot=True, fmt=".3f", cmap="RdYlGn", cbar_kws={"label": "Probability"},
            ax=ax, linewidths=0.5)
ax.set_title("HMM State Transition Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_PATH / "04_transition_matrix.png", dpi=300, bbox_inches="tight")
print("   Transition matrix heatmap saved")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
volatilities = []
regime_list = []
for regime in [1, -1, 0]:
    mask = df["regime"] == regime
    volatilities.append(df[mask]["rolling_vol"].dropna().values)
    regime_list.append(regime_names[regime])

bp = ax.boxplot(volatilities, tick_labels=regime_list, patch_artist=True)
for patch, regime in zip(bp["boxes"], [1, -1, 0]):
    patch.set_facecolor(colors[regime])
    patch.set_alpha(0.7)

ax.set_ylabel("Volatility", fontsize=12)
ax.set_title("Volatility Distribution by Regime", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS_PATH / "05_volatility_by_regime.png", dpi=300, bbox_inches="tight")
print("   Volatility boxplot saved")
plt.close()

print("\n" + "="*70)
print(" HMM REGIME DETECTION COMPLETED SUCCESSFULLY!")
print("="*70)
