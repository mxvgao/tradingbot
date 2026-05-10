"""Data loading utilities for backtesting with historical data."""

import numpy as np
import pandas as pd
from pathlib import Path


class DataLoader:
    """Load and provide access to historical price data."""
    
    def __init__(self, csv_path: str | Path):
        """
        Args:
            csv_path: Path to OHLCV CSV file. Must have 'close' column at minimum.
        """
        # Try to detect if it's a MultiIndex format
        with open(csv_path, 'r') as f:
            lines = [f.readline().strip() for _ in range(3)]
        
        if 'Price' in lines[0] and 'Ticker' in lines[1]:
            # Yahoo Finance MultiIndex format
            self.df = pd.read_csv(csv_path, header=[0, 1])
            # Skip any rows where Date column is empty
            self.df = self.df[self.df.index.notna()]
            # Find the Close column
            close_cols = [col for col in self.df.columns if col[0] == 'Close']
            if close_cols:
                self.prices = self.df[close_cols[0]].values
            else:
                raise ValueError("Could not find Close column in Yahoo Finance format")
        else:
            # Standard format
            self.df = pd.read_csv(csv_path)
            if 'close' in self.df.columns:
                self.prices = self.df['close'].values
            elif 'Close' in self.df.columns:
                self.prices = self.df['Close'].values
            else:
                raise ValueError("Could not find close/Close column")
        
        # Filter out any NaN prices
        valid_mask = ~np.isnan(self.prices)
        self.prices = self.prices[valid_mask]
        if hasattr(self.df, 'index') and len(self.df.index) == len(valid_mask):
            self.df = self.df[valid_mask]
        
        self.dates = self.df.index if hasattr(self.df.index, 'name') and self.df.index.name == 'Date' else None
        
    def get_price_segment(self, start_idx: int, length: int) -> np.ndarray:
        """Extract a price path segment.
        
        Args:
            start_idx: Starting index
            length: Number of periods
            
        Returns:
            Array of prices
        """
        end_idx = min(start_idx + length, len(self.prices))
        return self.prices[start_idx:end_idx]
    
    def get_returns(self) -> np.ndarray:
        """Calculate log returns.
        
        Returns:
            Array of log returns between consecutive prices
        """
        return np.diff(np.log(self.prices))
    
    def estimate_volatility(self, lookback: int = 30) -> float:
        """Estimate annualized volatility from returns.
        
        Args:
            lookback: Number of periods for volatility estimation
            
        Returns:
            Daily volatility estimate
        """
        returns = self.get_returns()
        if len(returns) < lookback:
            lookback = len(returns)
        return float(np.std(returns[-lookback:], ddof=1))
    
    def __len__(self) -> int:
        return len(self.prices)
    
    def __getitem__(self, idx: int) -> float:
        return self.prices[idx]


class SyntheticDataGenerator:
    """Generate synthetic price paths for testing."""
    
    @staticmethod
    def geometric_brownian_motion(
        S0: float, 
        mu: float, 
        sigma: float, 
        T: int, 
        n_paths: int = 1,
        seed: int = None
    ) -> np.ndarray:
        """Generate GBM price paths.
        
        Args:
            S0: Initial price
            mu: Drift
            sigma: Volatility
            T: Number of steps
            n_paths: Number of paths to generate
            seed: Random seed
            
        Returns:
            Array of shape (n_paths, T+1) with price paths
        """
        rng = np.random.default_rng(seed)
        dt = 1.0
        
        paths = np.zeros((n_paths, T + 1))
        paths[:, 0] = S0
        
        for t in range(T):
            dW = rng.standard_normal(n_paths)
            paths[:, t + 1] = paths[:, t] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW)
        
        return paths
    
    @staticmethod
    def random_walk(
        S0: float,
        sigma: float,
        T: int,
        n_paths: int = 1,
        seed: int = None
    ) -> np.ndarray:
        """Generate random walk price paths.
        
        Args:
            S0: Initial price
            sigma: Daily volatility
            T: Number of steps
            n_paths: Number of paths
            seed: Random seed
            
        Returns:
            Array of shape (n_paths, T+1)
        """
        rng = np.random.default_rng(seed)
        paths = np.zeros((n_paths, T + 1))
        paths[:, 0] = S0
        
        for t in range(T):
            shocks = rng.standard_normal(n_paths)
            paths[:, t + 1] = paths[:, t] + sigma * shocks
        
        return paths
