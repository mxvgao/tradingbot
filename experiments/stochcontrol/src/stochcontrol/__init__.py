"""stochcontrol package. Expose core API."""

from .config import RunConfig
from .backtest import backtest, compare_strategies, run_single_path
from .strategy import Strategy, TWAPStrategy, ImmediateStrategy, POVStrategy, OptimalDPStrategy, LinearVWAPStrategy, RandomStrategy
from .simulator import simulate_costs, PriceGenerator
from .tracker import TradeTracker, MultiPathTracker
from .data_loader import DataLoader, SyntheticDataGenerator
from .policies import solve_optimal_execution_dp, twap_policy, immediate_policy, pov_policy
from .metrics import summarize_costs, add_scores, mean_ci
from .baskets import (
    BasketCandidate,
    PairCandidate,
    clean_price_frame,
    compute_half_life,
    compute_spread,
    evaluate_basket,
    evaluate_pair,
    find_best_pairs,
    find_correlated_baskets,
    fit_hedge_ratio,
    rolling_cointegration_pass_rate,
    score_basket,
    score_pair,
)

__all__ = [
    'RunConfig',
    'backtest', 'compare_strategies', 'run_single_path',
    'Strategy', 'TWAPStrategy', 'ImmediateStrategy', 'POVStrategy', 'OptimalDPStrategy', 'LinearVWAPStrategy', 'RandomStrategy',
    'simulate_costs', 'PriceGenerator',
    'TradeTracker', 'MultiPathTracker',
    'DataLoader', 'SyntheticDataGenerator',
    'solve_optimal_execution_dp', 'twap_policy', 'immediate_policy', 'pov_policy',
    'summarize_costs', 'add_scores', 'mean_ci',
    'BasketCandidate', 'PairCandidate',
    'clean_price_frame', 'compute_half_life', 'compute_spread',
    'evaluate_basket', 'evaluate_pair', 'find_best_pairs', 'find_correlated_baskets',
    'fit_hedge_ratio', 'rolling_cointegration_pass_rate', 'score_basket', 'score_pair',
]
