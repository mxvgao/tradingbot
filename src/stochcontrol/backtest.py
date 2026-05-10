"""Unified backtesting interface for fast iteration."""

import numpy as np
from .strategy import Strategy
from .tracker import TradeTracker, MultiPathTracker
from .metrics import summarize_costs


def run_single_path(
    strategy: Strategy,
    T: int,
    Q0: int,
    S0: float,
    price_path: np.ndarray,
    eta: float,
    gamma: float = 0.0,
    tracker: TradeTracker = None
) -> float:
    """Execute strategy on a single price path with tracking.
    
    Args:
        strategy: Strategy instance
        T: Number of time steps
        Q0: Initial inventory
        S0: Initial price (reference)
        price_path: Price at each time step, shape (T+1,)
        eta: Market impact parameter: impact = eta * u
        gamma: Price pressure parameter: dS = gamma * u
        tracker: Optional TradeTracker to record details
        
    Returns:
        Total execution cost
    """
    u_star = strategy.get_execution_schedule(T, Q0)
    
    Q = Q0
    proceeds = 0.0
    S = price_path[0]
    
    for t in range(T):
        if Q == 0:
            break
        
        u = int(u_star[t, Q])
        
        # Execution price (after market impact)
        market_impact = eta * u
        p_exec = S - market_impact
        
        # Record trade if tracker provided
        if tracker is not None:
            tracker.record_step(t, u, p_exec, market_impact, Q - u)
        
        # Update proceeds
        proceeds += u * p_exec
        
        # Price evolution: drift + stochastic shock - price pressure
        if t + 1 < len(price_path):
            S = price_path[t + 1]
        
        Q -= u
    
    cost = Q0 * S0 - proceeds
    return cost


import numpy as np
from .strategy import Strategy
from .tracker import TradeTracker, MultiPathTracker
from .metrics import summarize_costs
from .simulator import PriceGenerator


def run_single_path(
    strategy: Strategy,
    T: int,
    Q0: int,
    S0: float,
    price_path: np.ndarray,
    eta: float,
    gamma: float = 0.0,
    tracker: TradeTracker = None
) -> float:
    """Execute strategy on a single price path with tracking."""
    u_star = strategy.get_execution_schedule(T, Q0)
    Q = Q0
    proceeds = 0.0
    S = price_path[0]

    for t in range(T):
        if Q == 0:
            break

        u = int(u_star[t, Q])
        market_impact = eta * u
        p_exec = S - market_impact

        if tracker is not None:
            tracker.record_step(t, u, p_exec, market_impact, Q - u)

        proceeds += u * p_exec
        Q -= u

        if t + 1 < len(price_path):
            S = price_path[t + 1] - gamma * u

    return Q0 * S0 - proceeds


def backtest(
    strategy: Strategy,
    T: int,
    Q0: int,
    S0: float,
    sigma: float,
    eta: float,
    gamma: float = 0.0,
    n_paths: int = 1000,
    seed: int = None,
    use_price_shocks: bool = True,
    track_details: bool = False
) -> dict:
    """Run backtest with given strategy and configuration."""
    rng = np.random.default_rng(seed)
    costs = np.zeros(n_paths)
    multi_tracker = MultiPathTracker() if track_details else None

    for k in range(n_paths):
        path_seed = None if seed is None else seed + k
        if use_price_shocks:
            price_path = PriceGenerator.gbm_path(S0, sigma, T, seed=path_seed)
        else:
            price_path = PriceGenerator.random_walk(S0, sigma, T, seed=path_seed)

        tracker = TradeTracker(S0, Q0) if track_details else None
        cost = run_single_path(
            strategy=strategy,
            T=T,
            Q0=Q0,
            S0=S0,
            price_path=price_path,
            eta=eta,
            gamma=gamma,
            tracker=tracker
        )

        costs[k] = cost
        if track_details and tracker is not None:
            multi_tracker.add_path(tracker)

    summary = summarize_costs(costs)
    summary['strategy'] = str(strategy)
    summary['n_paths'] = n_paths
    summary['config'] = {
        'T': T,
        'Q0': Q0,
        'S0': S0,
        'sigma': sigma,
        'eta': eta,
        'gamma': gamma,
    }

    if track_details:
        summary['tracker'] = multi_tracker

    return summary



def compare_strategies(
    strategies: list[Strategy],
    T: int,
    Q0: int,
    S0: float,
    sigma: float,
    eta: float,
    gamma: float = 0.0,
    n_paths: int = 1000,
    seed: int = None
) -> dict[str, dict]:
    """Compare multiple strategies side-by-side.
    
    Args:
        strategies: List of Strategy instances
        T, Q0, S0, sigma, eta, gamma: Backtest parameters
        n_paths: Number of paths
        seed: Random seed
        
    Returns:
        Dictionary with results for each strategy
    """
    results = {}
    for strategy in strategies:
        results[str(strategy)] = backtest(
            strategy=strategy,
            T=T,
            Q0=Q0,
            S0=S0,
            sigma=sigma,
            eta=eta,
            gamma=gamma,
            n_paths=n_paths,
            seed=seed,
            track_details=False
        )
    return results
