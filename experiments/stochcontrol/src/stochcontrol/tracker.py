"""Position and PnL tracking during execution."""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Trade:
    """Single trade record."""
    t: int  # Time step
    u: int  # Quantity executed
    price: float  # Execution price
    impact: float  # Market impact cost
    pnl: float = 0.0  # Realized PnL from this trade
    
    def __post_init__(self):
        if self.pnl == 0.0:
            # PnL is negative (cost) if we sell below initial price
            self.pnl = -self.impact


class TradeTracker:
    """Track execution, inventory, and PnL throughout simulation."""
    
    def __init__(self, S0: float, Q0: int):
        """
        Args:
            S0: Initial price benchmark
            Q0: Initial inventory
        """
        self.S0 = S0
        self.Q0 = Q0
        
        self.trades: List[Trade] = []
        self.inventory_path = [Q0]  # inventory at each step
        self.price_path = [S0]      # price at each step
        self.pnl_path = [0.0]       # cumulative PnL at each step
        self.cash_path = [0.0]      # cumulative cash collected
        
    def record_step(
        self, 
        t: int,
        u: int,
        price: float,
        market_impact: float,
        remaining_inventory: int
    ):
        """Record a trading step.
        
        Args:
            t: Time step
            u: Quantity executed
            price: Execution price (after impact)
            market_impact: Market impact cost
            remaining_inventory: Inventory after trade
        """
        if u > 0:
            trade = Trade(
                t=t,
                u=u,
                price=price,
                impact=market_impact,
                pnl=-market_impact  # Negative: this is a cost
            )
            self.trades.append(trade)
            
            # Update cash (proceeds from selling)
            proceeds = u * price
            cash = self.cash_path[-1] + proceeds
            
            # Update PnL (cost basis was S0, actually sold at price)
            pnl = cash - (self.Q0 - remaining_inventory) * self.S0
            
            self.inventory_path.append(remaining_inventory)
            self.price_path.append(price)
            self.pnl_path.append(pnl)
            self.cash_path.append(cash)
    
    def final_cost(self) -> float:
        """Total execution cost (Q0*S0 - final_proceeds)."""
        if self.Q0 == 0:
            return 0.0
        return self.Q0 * self.S0 - self.cash_path[-1]
    
    def total_impact_costs(self) -> float:
        """Sum of all market impact costs."""
        return sum(t.impact for t in self.trades)
    
    def num_trades(self) -> int:
        """Number of trades executed."""
        return len(self.trades)
    
    def avg_execution_price(self) -> float:
        """Volume-weighted average execution price."""
        if self.num_trades() == 0:
            return self.S0
        total_qty = sum(t.u for t in self.trades)
        if total_qty == 0:
            return self.S0
        total_value = sum(t.u * t.price for t in self.trades)
        return total_value / total_qty
    
    def max_inventory(self) -> int:
        """Peak inventory during execution."""
        return max(self.inventory_path) if self.inventory_path else 0
    
    def summary(self) -> dict:
        """Generate summary statistics."""
        return {
            "num_trades": self.num_trades(),
            "total_cost": float(self.final_cost()),
            "impact_costs": float(self.total_impact_costs()),
            "avg_execution_price": float(self.avg_execution_price()),
            "initial_price": float(self.S0),
            "price_slippage": float(self.avg_execution_price() - self.S0),
            "max_inventory_held": self.max_inventory(),
        }


class MultiPathTracker:
    """Track statistics across multiple simulated paths."""
    
    def __init__(self):
        self.trackers: List[TradeTracker] = []
        
    def add_path(self, tracker: TradeTracker):
        """Add a completed path."""
        self.trackers.append(tracker)
    
    def costs(self) -> np.ndarray:
        """Get all execution costs."""
        return np.array([t.final_cost() for t in self.trackers])
    
    def avg_prices(self) -> np.ndarray:
        """Get all VWAP prices."""
        return np.array([t.avg_execution_price() for t in self.trackers])
    
    def total_impacts(self) -> np.ndarray:
        """Get all total impact costs."""
        return np.array([t.total_impact_costs() for t in self.trackers])
    
    def num_trades_per_path(self) -> np.ndarray:
        """Get number of trades in each path."""
        return np.array([t.num_trades() for t in self.trackers])
