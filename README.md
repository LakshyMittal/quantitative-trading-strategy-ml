# quantitative-trading-strategy-ml
## Author
Lakshy Mittal
# Quantitative Trading Strategy using Machine Learning

## Overview
This project implements an end-to-end quantitative research pipeline for
intraday market data using Python and machine learning.

The objective is not to predict prices directly, but to use ML as a
**trade filter** to improve consistency and reduce noise in a
rule-based trading strategy.

This project is developed as part of a technical screening task.

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


