"""
ML Model Training Pipeline
Trains Logistic Regression, XGBoost, and LSTM models
Saves models and generates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models import LogisticRegressionFilter, XGBoostFilter, LSTMFilter
from data_utils import DataLoader, DataValidator

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = PROJECT_ROOT / "results"
PLOTS_PATH = PROJECT_ROOT / "plots"
MODELS_PATH = PROJECT_ROOT / "models"

# Create directories
for path in [RESULTS_PATH, PLOTS_PATH, MODELS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ML MODEL TRAINING PIPELINE")
print("="*70)

# ===== LOAD DATA =====
print("\n[1/5] Loading data...")
loader = DataLoader(str(DATA_PATH))
df = loader.load_features("features_1y.csv")
print(f" Loaded {len(df)} samples with {len(df.columns)} columns")

# ===== CREATE TARGET =====
print("\n[2/5] Creating target variable...")
df["future_close"] = df["close"].shift(-1)
df["target"] = (df["future_close"] > df["close"]).astype(int)
df = df.dropna().reset_index(drop=True)

print(f" Target distribution: {df['target'].value_counts().to_dict()}")
print(f" Positive class: {df['target'].mean():.2%}")

# ===== PREPARE FEATURES =====
print("\n[3/5] Preparing features...")
exclude_cols = ["date", "close", "future_close", "target"]
feature_columns = [col for col in df.columns if col not in exclude_cols]

X = df[feature_columns].fillna(0).values
y = df["target"].values

# Time-series split
split_idx = int(len(df) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f" Features: {len(feature_columns)}")
print(f" Train: {len(X_train)}, Test: {len(X_test)}")

# ===== TRAIN MODELS =====
print("\n[4/5] Training models...")

results = {}

# Logistic Regression
print("\n  Training Logistic Regression...")
lr_model = LogisticRegressionFilter(max_iter=1000)
lr_model.fit(X_train, y_train)
results["Logistic Regression"] = lr_model.evaluate(X_test, y_test)
print(f"   Accuracy: {results['Logistic Regression']['accuracy']:.4f}")

# XGBoost
print("\n  Training XGBoost...")
try:
    xgb_model = XGBoostFilter(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X_train, y_train)
    results["XGBoost"] = xgb_model.evaluate(X_test, y_test)
    print(f"   Accuracy: {results['XGBoost']['accuracy']:.4f}")
    
    xgb_importance = xgb_model.get_feature_importance(feature_columns)
    print(f"   Top feature: {xgb_importance.iloc[0]['feature']}")
except Exception as e:
    print(f"   XGBoost error: {e}")

# LSTM
print("\n  Training LSTM...")
try:
    lstm_model = LSTMFilter(sequence_length=10, lstm_units=64, dropout=0.2)
    lstm_model.build_model(input_features=X_train.shape[1])
    
    if lstm_model.model is not None:
        lstm_model.fit(X_train, y_train, epochs=5, batch_size=32)
        print(f"   LSTM model trained")
    else:
        print(f"   LSTM not available (TensorFlow)")
except Exception as e:
    print(f"   LSTM error: {e}")

# ===== SAVE MODELS & RESULTS =====
print("\n[5/5] Saving models and results...")

with open(MODELS_PATH / "logistic_regression_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)
print("   Logistic Regression model saved")

try:
    with open(MODELS_PATH / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    print("   XGBoost model saved")
    
    xgb_importance.to_csv(RESULTS_PATH / "xgboost_feature_importance.csv", index=False)
    print("   Feature importance saved")
except:
    pass

with open(MODELS_PATH / "feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)
print("   Feature columns reference saved")

results_df = pd.DataFrame(results).T
results_df.to_csv(RESULTS_PATH / "ml_model_comparison.csv")
print("   Model comparison results saved")

print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)
print("\n", results_df.round(4))

print("\n[BONUS] Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = list(results.keys())
accuracies = [results[m]["accuracy"] for m in models]
auc_scores = [results[m]["auc_roc"] for m in models]

colors = ["#1f77b4", "#ff7f0e"]

axes[0].bar(models, accuracies, color=colors[:len(models)])
axes[0].set_ylabel("Accuracy", fontsize=12)
axes[0].set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
axes[0].set_ylim([0, 1])
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(models, auc_scores, color=colors[:len(models)])
axes[1].set_ylabel("ROC AUC Score", fontsize=12)
axes[1].set_title("Model ROC AUC Comparison", fontsize=14, fontweight="bold")
axes[1].set_ylim([0, 1])
for i, v in enumerate(auc_scores):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PLOTS_PATH / "01_model_comparison.png", dpi=300, bbox_inches="tight")
print("   Model comparison plot saved")
plt.close()

try:
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = xgb_importance.head(15)
    ax.barh(range(len(top_features)), top_features["importance"].values, color="#ff7f0e")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values, fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("XGBoost Feature Importance (Top 15)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "02_xgboost_feature_importance.png", dpi=300, bbox_inches="tight")
    print("   Feature importance plot saved")
    plt.close()
except:
    print("   Feature importance plot skipped")

print("\n" + "="*70)
print(" ML TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nOutputs saved to:")
print(f"  Models: {MODELS_PATH}")
print(f"  Results: {RESULTS_PATH}")
print(f"  Plots: {PLOTS_PATH}")
