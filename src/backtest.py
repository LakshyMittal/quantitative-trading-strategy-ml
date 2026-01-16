"""
Backtesting engine for trading strategies.
Calculates performance metrics and generates backtest reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class BacktestMetrics:
    """Calculate backtest performance metrics."""
    
    @staticmethod
    def calculate_returns_metrics(returns: np.ndarray, risk_free_rate: float = 0.065) -> Dict[str, float]:
        """
        Calculate return-based metrics.
        
        Args:
            returns: Array of strategy returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of metrics
        """
        total_return = np.sum(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Annualize (assuming 252 trading days, 78 5-min bars per day)
        annual_factor = 252 * 78
        annual_return = mean_return * annual_factor
        annual_volatility = std_return * np.sqrt(annual_factor)
        
        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_mean": mean_return,
            "daily_std": std_return,
            "annual_volatility": annual_volatility
        }
        return metrics
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.065, annual_factor: float = 19656) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            annual_factor: Annualization factor (252*78 for 5-min bars)
            
        Returns:
            Sharpe Ratio
        """
        excess_return = np.mean(returns) - (risk_free_rate / annual_factor)
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0
        
        sharpe = excess_return / volatility * np.sqrt(annual_factor)
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.065, annual_factor: float = 19656) -> float:
        """
        Calculate Sortino Ratio (downside volatility only).
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            annual_factor: Annualization factor
            
        Returns:
            Sortino Ratio
        """
        excess_return = np.mean(returns) - (risk_free_rate / annual_factor)
        downside_returns = np.minimum(returns, 0)
        downside_volatility = np.std(downside_returns)
        
        if downside_volatility == 0:
            return 0
        
        sortino = excess_return / downside_volatility * np.sqrt(annual_factor)
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown (negative value)
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, annual_factor: float = 19656) -> float:
        """
        Calculate Calmar Ratio.
        
        Args:
            returns: Array of returns
            annual_factor: Annualization factor
            
        Returns:
            Calmar Ratio
        """
        annual_return = np.mean(returns) * annual_factor
        max_dd = BacktestMetrics.calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return 0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trade-based metrics.
        
        Args:
            trades: DataFrame of individual trades with PnL
            
        Returns:
            Dictionary of metrics
        """
        if len(trades) == 0:
            return {}
        
        profitable = trades[trades["pnl"] > 0]
        losing = trades[trades["pnl"] <= 0]
        
        metrics = {
            "total_trades": len(trades),
            "winning_trades": len(profitable),
            "losing_trades": len(losing),
            "win_rate": len(profitable) / len(trades) if len(trades) > 0 else 0,
            "avg_win": profitable["pnl"].mean() if len(profitable) > 0 else 0,
            "avg_loss": losing["pnl"].mean() if len(losing) > 0 else 0,
            "profit_factor": profitable["pnl"].sum() / abs(losing["pnl"].sum()) if len(losing) > 0 else np.inf,
            "avg_trade_duration": trades["duration"].mean() if "duration" in trades.columns else 0
        }
        return metrics


class Backtester:
    """Main backtesting engine."""
    
    def __init__(self, commission: float = 0.001):
        """
        Initialize backtester.
        
        Args:
            commission: Commission per trade as fraction
        """
        self.commission = commission
        self.metrics = {}
        logger.info(f"Initialized Backtester (commission={commission*100}%)")
    
    def run(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        price_col: str = "close"
    ) -> Dict[str, float]:
        """
        Run backtest.
        
        Args:
            df: Input dataframe with signals
            signal_col: Signal column name
            price_col: Price column name
            
        Returns:
            Dictionary of backtest metrics
        """
        df = df.copy()
        
        # Calculate returns
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
        df["signal_shifted"] = df[signal_col].shift(1)
        df["strategy_return"] = df["signal_shifted"] * df["log_return"]
        
        # Remove commission
        df.loc[df[signal_col] != 0, "strategy_return"] -= self.commission
        
        # Get return arrays
        strategy_returns = df["strategy_return"].dropna().values
        market_returns = df["log_return"].dropna().values
        
        # Calculate all metrics
        self.metrics = {
            "strategy": BacktestMetrics.calculate_returns_metrics(strategy_returns),
            "market": BacktestMetrics.calculate_returns_metrics(market_returns),
            "sharpe_ratio": BacktestMetrics.calculate_sharpe_ratio(strategy_returns),
            "sortino_ratio": BacktestMetrics.calculate_sortino_ratio(strategy_returns),
            "max_drawdown": BacktestMetrics.calculate_max_drawdown(strategy_returns),
            "calmar_ratio": BacktestMetrics.calculate_calmar_ratio(strategy_returns)
        }
        
        logger.info(f"Backtest completed. Sharpe: {self.metrics['sharpe_ratio']:.2f}")
        return self.metrics
    
    def get_report(self) -> pd.DataFrame:
        """Get backtest report as DataFrame."""
        if not self.metrics:
            return pd.DataFrame()
        
        report_data = {
            "Strategy Total Return": self.metrics["strategy"]["total_return"],
            "Market Total Return": self.metrics["market"]["total_return"],
            "Annual Return": self.metrics["strategy"]["annual_return"],
            "Annual Volatility": self.metrics["strategy"]["annual_volatility"],
            "Sharpe Ratio": self.metrics["sharpe_ratio"],
            "Sortino Ratio": self.metrics["sortino_ratio"],
            "Max Drawdown": self.metrics["max_drawdown"],
            "Calmar Ratio": self.metrics["calmar_ratio"]
        }
        
        return pd.DataFrame(list(report_data.items()), columns=["Metric", "Value"])