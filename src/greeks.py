"""
Options Greeks calculation using Black-Scholes model.
Calculates Delta, Gamma, Theta, Vega, and Rho for options.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class BlackScholesCalculator:
    """Calculate options Greeks using Black-Scholes model."""
    
    def __init__(self, risk_free_rate: float = 0.065):
        """
        Initialize Black-Scholes calculator.
        
        Args:
            risk_free_rate: Risk-free rate (default 6.5%)
        """
        self.r = risk_free_rate
        logger.info(f"Initialized BS Calculator with r={risk_free_rate*100}%")
    
    def _d1_d2(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> Tuple[float, float]:
        """
        Calculate d1 and d2 from Black-Scholes formula.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            sigma: Volatility (annualized)
            
        Returns:
            Tuple of (d1, d2)
        """
        if T <= 0 or sigma <= 0:
            return 0, 0
        
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate call option delta."""
        d1, _ = self._d1_d2(S, K, T, sigma)
        return norm.cdf(d1)
    
    def put_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate put option delta."""
        d1, _ = self._d1_d2(S, K, T, sigma)
        return norm.cdf(d1) - 1
    
    def gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate option gamma (same for calls and puts)."""
        d1, _ = self._d1_d2(S, K, T, sigma)
        if T <= 0 or sigma <= 0:
            return 0
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate option vega (same for calls and puts). Per 1% change in IV."""
        d1, _ = self._d1_d2(S, K, T, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% IV change
    
    def theta(self, S: float, K: float, T: float, sigma: float, option_type: str = "call") -> float:
        """
        Calculate option theta. Per day decay.
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Theta per day
        """
        d1, d2 = self._d1_d2(S, K, T, sigma)
        
        if T <= 0:
            return 0
        
        if option_type == "call":
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
            )
        else:  # put
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
            )
        
        return theta / 365  # Convert to per-day theta
    
    def rho(self, S: float, K: float, T: float, sigma: float, option_type: str = "call") -> float:
        """
        Calculate option rho. Per 1% change in interest rate.
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Rho per 1% rate change
        """
        _, d2 = self._d1_d2(S, K, T, sigma)
        
        if T <= 0:
            return 0
        
        if option_type == "call":
            rho = K * T * np.exp(-self.r * T) * norm.cdf(d2) / 100
        else:  # put
            rho = -K * T * np.exp(-self.r * T) * norm.cdf(-d2) / 100
        
        return rho
    
    def get_all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = "call"
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with all Greeks
        """
        greeks = {
            "delta": self.call_delta(S, K, T, sigma) if option_type == "call" else self.put_delta(S, K, T, sigma),
            "gamma": self.gamma(S, K, T, sigma),
            "theta": self.theta(S, K, T, sigma, option_type),
            "vega": self.vega(S, K, T, sigma),
            "rho": self.rho(S, K, T, sigma, option_type)
        }
        return greeks