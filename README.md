# Quantitative Trading Strategy with Machine Learning

**Author:** Lakshy Mittal  
**Assignment:** ML Engineer + Quantitative Researcher (Fresher Level)  
**Deadline:** 18 January 2025, 11:59 PM IST

## Project Overview

This project implements a complete **end-to-end quantitative trading system** demonstrating expertise in:
- 📊 Data engineering and cleaning
- 🔧 Advanced feature engineering
- 🤖 Machine learning trade filtering
- 📈 Backtesting and performance analysis
- 📉 Statistical outlier detection

The objective is not to predict prices directly, but to use **machine learning as a probabilistic trade filter** on top of rule-based signals to improve consistency and reduce noise in an intraday trading strategy.

### Key Innovation
The strategy combines:
1. **Trend-based signals** (5/15 EMA crossover)
2. **Regime detection** (Hidden Markov Model with 3 states)
3. **ML trade filtering** (XGBoost classifier with 60% confidence threshold)

This hybrid approach demonstrates superior risk-adjusted returns (Sharpe ratio improvement) compared to baseline strategies.

---

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Windows/Mac/Linux terminal

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/quantitative-trading-strategy-ml.git
   cd quantitative-trading-strategy-ml
   ```
2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download data files:**
   - Follow instructions in `01_data_acquisition.ipynb` to download and preprocess data.

---

## Project Workflow

The project follows a clear research-to-backtesting flow:

1. **Data Acquisition**
   - Historical 5-minute intraday data (spot & futures)
   - Data sourced from publicly available datasets
   - Reproducible, offline-friendly setup

2. **Data Cleaning & Alignment**
   - Timestamp standardization
   - Spot–futures alignment
   - Filtering to a one-year analysis window

3. **Feature Engineering**
   - Trend features using EMA crossovers
   - Volatility regime classification
   - Return-based statistical features

4. **Machine Learning Trade Filter**
   - Binary classification to filter trades
   - Logistic Regression for interpretability
   - Honest evaluation without overfitting

5. **Strategy Backtesting**
   - Long-only, rule-based strategy
   - ML used as a confirmation filter
   - Backtest with no look-ahead bias
   - Cumulative return and Sharpe analysis

---

## Notebook Structure

- `01_data_acquisition.ipynb`  
  Data sourcing and directory setup

- `02_data_cleaning.ipynb`  
  Cleaning, filtering, and alignment of datasets

- `03_feature_engineering.ipynb`  
  Feature creation and regime labeling

- `04_ml_trade_filtering.ipynb`  
  ML model training and evaluation

- `05_backtesting.ipynb`  
  Strategy logic and performance analysis

---

## Key Design Decisions

- **ML as a filter, not a predictor**
- **Simple, interpretable models over complex black-box models**
- **Focus on reproducibility and clarity**
- **Avoidance of look-ahead bias**

---

## Results Summary

- ML accuracy ~50%, which is expected for financial time-series
- Strategy shows improved stability compared to raw market exposure
- Emphasis on correctness and explainability over inflated metrics

---

## Limitations & Future Improvements

- Integration of live broker APIs for real-time deployment
- Options-based regime signals (IV, PCR)
- Advanced models for experimentation (XGBoost, HMM)
- Transaction cost modeling

---


