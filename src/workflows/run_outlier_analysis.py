"""
High-Performance Trade Analysis & Outlier Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy import EMA5_15Strategy

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = PROJECT_ROOT / "results"
PLOTS_PATH = PROJECT_ROOT / "plots"

for path in [RESULTS_PATH, PLOTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print("="*70)
print("HIGH-PERFORMANCE TRADE ANALYSIS & OUTLIER DETECTION")
print("="*70)

print("\n[1/5] Loading and preparing trade data...")
features_path = DATA_PATH / "features_1y.csv"
df = pd.read_csv(features_path, parse_dates=["date"])

strategy = EMA5_15Strategy(regime_filter=False)
df = strategy.generate_signals(df, "ema_fast", "ema_slow")
df = strategy.calculate_returns(df, "signal")

print(f" Loaded {len(df)} candles")

print("\n[2/5] Extracting individual trades...")

trades = []
in_trade = False
entry_price = 0
entry_date = None
entry_idx = 0
entry_signal = 0

for idx, row in df.iterrows():
    if row["signal"] != 0 and not in_trade:
        in_trade = True
        entry_price = row["close"]
        entry_date = row["date"]
        entry_idx = idx
        entry_signal = row["signal"]
    
    elif row["signal"] != entry_signal and in_trade:
        exit_price = row["close"]
        exit_date = row["date"]
        exit_idx = idx
        
        pnl = (exit_price - entry_price) * entry_signal
        pnl_pct = pnl / entry_price
        duration = exit_idx - entry_idx
        
        trade = {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "duration": duration,
            "signal_type": "LONG" if entry_signal == 1 else "SHORT",
            "avg_vol": df.iloc[entry_idx:exit_idx+1]["rolling_vol"].mean(),
            "avg_ema_spread": df.iloc[entry_idx:exit_idx+1]["ema_spread"].mean(),
        }
        trades.append(trade)
        
        in_trade = False

trades_df = pd.DataFrame(trades)

if len(trades_df) == 0:
    print("   No trades found, exiting")
    import sys
    sys.exit(1)

print(f" Extracted {len(trades_df)} trades")
print(f" Winning trades: {(trades_df['pnl'] > 0).sum()}")
print(f" Win rate: {(trades_df['pnl'] > 0).sum() / len(trades_df):.1%}")

print("\n[3/5] Detecting outliers (Z-score > 3)...")

trades_df["pnl_zscore"] = np.abs(stats.zscore(trades_df["pnl"]))
trades_df["duration_zscore"] = np.abs(stats.zscore(trades_df["duration"]))

trades_df["is_outlier"] = (trades_df["pnl_zscore"] > 3) | (trades_df["duration_zscore"] > 3)
trades_df["is_profitable"] = trades_df["pnl"] > 0

outlier_count = trades_df["is_outlier"].sum()
print(f" Outliers detected: {outlier_count}")
print(f" Outlier percentage: {outlier_count / len(trades_df):.1%}")

print("\n[4/5] Analyzing outlier characteristics...")

normal_profitable = trades_df[(~trades_df["is_outlier"]) & (trades_df["is_profitable"])]
outlier_profitable = trades_df[(trades_df["is_outlier"]) & (trades_df["is_profitable"])]

stats_comparison = {
    "Metric": [
        "Count",
        "Avg PnL",
        "Avg PnL %",
        "Avg Duration",
        "Avg Volatility",
        "Win Rate"
    ],
    "Normal Profitable": [
        len(normal_profitable),
        f"{normal_profitable['pnl'].mean():.2f}" if len(normal_profitable) > 0 else "N/A",
        f"{normal_profitable['pnl_pct'].mean():.2%}" if len(normal_profitable) > 0 else "N/A",
        f"{normal_profitable['duration'].mean():.0f}" if len(normal_profitable) > 0 else "N/A",
        f"{normal_profitable['avg_vol'].mean():.4f}" if len(normal_profitable) > 0 else "N/A",
        f"{(normal_profitable['pnl'] > 0).sum() / len(normal_profitable):.1%}" if len(normal_profitable) > 0 else "N/A"
    ],
    "Outlier Profitable": [
        len(outlier_profitable),
        f"{outlier_profitable['pnl'].mean():.2f}" if len(outlier_profitable) > 0 else "N/A",
        f"{outlier_profitable['pnl_pct'].mean():.2%}" if len(outlier_profitable) > 0 else "N/A",
        f"{outlier_profitable['duration'].mean():.0f}" if len(outlier_profitable) > 0 else "N/A",
        f"{outlier_profitable['avg_vol'].mean():.4f}" if len(outlier_profitable) > 0 else "N/A",
        f"{(outlier_profitable['pnl'] > 0).sum() / len(outlier_profitable):.1%}" if len(outlier_profitable) > 0 else "N/A"
    ]
}

stats_df = pd.DataFrame(stats_comparison)
print("\n", stats_df)

stats_df.to_csv(RESULTS_PATH / "outlier_analysis_comparison.csv", index=False)

print("\n[5/5] Saving trade data...")

trades_df.to_csv(RESULTS_PATH / "trade_analysis.csv", index=False)
outlier_profitable.to_csv(RESULTS_PATH / "outlier_profitable_trades.csv", index=False)
normal_profitable.to_csv(RESULTS_PATH / "normal_profitable_trades.csv", index=False)

print(" Trade data saved")

print("\n[BONUS] Creating visualizations...")

fig, ax = plt.subplots(figsize=(12, 6))

normal = trades_df[~trades_df["is_outlier"]]
outliers = trades_df[trades_df["is_outlier"]]

ax.scatter(normal["duration"], normal["pnl"], alpha=0.5, s=50, 
          c="#1f77b4", label="Normal Trades", edgecolors="black", linewidth=0.5)
ax.scatter(outliers["duration"], outliers["pnl"], alpha=0.8, s=100, 
          c="#d62728", label="Outlier Trades", marker="*", edgecolors="black", linewidth=1)

ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Trade Duration (bars)", fontsize=12)
ax.set_ylabel("PnL", fontsize=12)
ax.set_title("PnL vs Trade Duration", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "10_pnl_vs_duration.png", dpi=300, bbox_inches="tight")
print("   PnL vs duration scatter saved")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(trades_df["pnl"], bins=30, edgecolor="black", alpha=0.7, color="#1f77b4")
axes[0].axvline(trades_df["pnl"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {trades_df['pnl'].mean():.2f}")
axes[0].set_xlabel("PnL", fontsize=11)
axes[0].set_ylabel("Frequency", fontsize=11)
axes[0].set_title("PnL Distribution (All Trades)", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].hist(trades_df["pnl_pct"] * 100, bins=30, edgecolor="black", alpha=0.7, color="#2ca02c")
axes[1].axvline(trades_df["pnl_pct"].mean() * 100, color="red", linestyle="--", linewidth=2, 
               label=f"Mean: {trades_df['pnl_pct'].mean()*100:.2f}%")
axes[1].set_xlabel("PnL %", fontsize=11)
axes[1].set_ylabel("Frequency", fontsize=11)
axes[1].set_title("PnL % Distribution", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PLOTS_PATH / "11_pnl_distribution.png", dpi=300, bbox_inches="tight")
print("   PnL distribution saved")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

features_to_compare = [
    ("avg_vol", "Volatility"),
    ("avg_ema_spread", "EMA Spread"),
    ("duration", "Trade Duration"),
    ("pnl", "PnL")
]

for idx, (feature, title) in enumerate(features_to_compare):
    ax = axes[idx // 2, idx % 2]
    
    data_to_plot = [
        normal[feature].dropna(),
        outliers[feature].dropna()
    ]
    
    bp = ax.boxplot(data_to_plot, tick_labels=["Normal", "Outlier"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#1f77b4")
    bp["boxes"][1].set_facecolor("#d62728")
    
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(f"{title} by Trade Type", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PLOTS_PATH / "12_feature_comparison_boxplot.png", dpi=300, bbox_inches="tight")
print("   Feature comparison boxplot saved")
plt.close()

trades_df["hour"] = trades_df["entry_date"].dt.hour

hourly_performance = trades_df.groupby("hour").agg({
    "pnl": ["mean", "count"],
    "is_profitable": "sum"
}).round(4)

hourly_performance.columns = ["Avg_PnL", "Count", "Wins"]
hourly_performance["Win_Rate"] = hourly_performance["Wins"] / hourly_performance["Count"]

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(hourly_performance.index, hourly_performance["Avg_PnL"], 
      color=["#2ca02c" if x > 0 else "#d62728" for x in hourly_performance["Avg_PnL"]],
      edgecolor="black", alpha=0.7)

ax.set_xlabel("Hour of Day", fontsize=12)
ax.set_ylabel("Average PnL", fontsize=12)
ax.set_title("Trade Performance by Hour of Day", fontsize=14, fontweight="bold")
ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PLOTS_PATH / "13_time_of_day_analysis.png", dpi=300, bbox_inches="tight")
print("   Time of day analysis saved")
plt.close()

correlation_features = ["pnl", "duration", "avg_vol", "avg_ema_spread"]
corr_matrix = trades_df[correlation_features].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0,
           cbar_kws={"label": "Correlation"}, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_PATH / "14_correlation_heatmap.png", dpi=300, bbox_inches="tight")
print("   Correlation heatmap saved")
plt.close()

print("\n" + "="*70)
print(" OUTLIER ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nKey Findings:")
print(f"   Total trades: {len(trades_df)}")
print(f"   Outliers detected: {outlier_count} ({outlier_count/len(trades_df):.1%})")
