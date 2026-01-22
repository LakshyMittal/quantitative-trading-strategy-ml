"""
Backtesting Engine
"""
import pandas as pd
import numpy as np

class Backtester:
    """Simple vector-based backtester."""
    
    def __init__(self, commission=0.001, initial_capital=10000.0):
        self.commission = commission
        self.initial_capital = initial_capital
        
    def run(self, df: pd.DataFrame, signal_col: str, price_col: str) -> dict:
        """Run backtest and return metrics."""
        df = df.copy()
        
        # Calculate returns
        # Log return of the asset
        df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Strategy return: signal from previous day * today's return
        df['strategy_return'] = df[signal_col].shift(1) * df['log_return']
        
        # Apply commission on trades
        trades = df[signal_col].diff().abs().fillna(0)
        df['strategy_return'] -= trades * self.commission
        
        # Calculate Metrics
        total_return = df['strategy_return'].sum()
        annual_return = df['strategy_return'].mean() * 252
        annual_volatility = df['strategy_return'].std() * np.sqrt(252)
        
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Drawdown
        cum_returns = df['strategy_return'].cumsum()
        peak = cum_returns.cummax()
        drawdown = cum_returns - peak
        max_drawdown = drawdown.min()
        
        return {
            "strategy": {
                "total_return": total_return,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility
            },
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            "sortino_ratio": 0.0 # Simplified
        }