"""Base Strategy interface and implementations for fast idea testing."""

from abc import ABC, abstractmethod
import numpy as np


class Strategy(ABC):
    """Abstract base class for execution strategies.
    
    All strategies must implement get_execution_schedule(T, Q) which returns
    a (T, Q+1) array where u[t, q] is the quantity to execute at time t 
    given remaining inventory q.
    """
    
    @abstractmethod
    def get_execution_schedule(self, T: int, Q0: int) -> np.ndarray:
        """
        Args:
            T: number of time steps
            Q0: initial inventory
            
        Returns:
            u_star: (T, Q0+1) array of optimal execution quantities
        """
        pass
    
    def __repr__(self) -> str:
        return self.__class__.__name__


# Wrapper implementations of existing policies

class OptimalDPStrategy(Strategy):
    """Dynamic programming optimal execution strategy."""
    
    def __init__(self, sigma: float, eta: float, risk_lambda: float):
        self.sigma = sigma
        self.eta = eta
        self.risk_lambda = risk_lambda
    
    def get_execution_schedule(self, T: int, Q0: int) -> np.ndarray:
        """Solve optimal execution via backward induction."""
        from .policies import solve_optimal_execution_dp
        u_star, _ = solve_optimal_execution_dp(
            T=T, Q0=Q0, sigma=self.sigma, eta=self.eta, risk_lambda=self.risk_lambda
        )
        return u_star


class TWAPStrategy(Strategy):
    """Time-weighted average price strategy."""
    
    def get_execution_schedule(self, T: int, Q0: int) -> np.ndarray:
        from .policies import twap_policy
        return twap_policy(T=T, Q0=Q0)


class ImmediateStrategy(Strategy):
    """Execute entire position immediately."""
    
    def get_execution_schedule(self, T: int, Q0: int) -> np.ndarray:
        from .policies import immediate_policy
        return immediate_policy(T=T, Q0=Q0)


class POVStrategy(Strategy):
    """Percentage of volume strategy."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
    
    def get_execution_schedule(self, T: int, Q0: int) -> np.ndarray:
        from .policies import pov_policy
        return pov_policy(T=T, Q0=Q0, alpha=self.alpha)


class LinearVWAPStrategy(Strategy):
    """Simple linear execution that mimics VWAP."""
    
    def get_execution_schedule(self, T: int, Q0: int) -> np.ndarray:
        u = np.zeros((T, Q0 + 1), dtype=int)
        per_period = Q0 // T
        remainder = Q0 % T
        
        for t in range(T):
            for q in range(Q0 + 1):
                if q <= per_period * (T - t):
                    u[t, q] = per_period + (1 if t < remainder else 0)
                else:
                    u[t, q] = q
        return u


class RandomStrategy(Strategy):
    """Random valid execution schedule (for testing/sanity checks)."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def get_execution_schedule(self, T: int, Q0: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        u = np.zeros((T, Q0 + 1), dtype=int)
        
        for q in range(Q0 + 1):
            remaining = q
            for t in range(T):
                if remaining == 0:
                    break
                # Randomly decide how much to execute
                max_trade = remaining if t == T - 1 else rng.integers(0, remaining + 1)
                u[t, q] = max_trade
                remaining -= max_trade
        
        return u
