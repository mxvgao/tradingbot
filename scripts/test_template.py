"""Template for quick testing of new strategies."""

from stochcontrol import Strategy, TWAPStrategy, OptimalDPStrategy, POVStrategy, backtest, compare_strategies, RunConfig
import numpy as np


# ============================================================================
# EXAMPLE 1: Test a single strategy
# ============================================================================

def test_single_strategy():
    """Quick test of TWAP strategy."""
    cfg = RunConfig()
    
    strategy = TWAPStrategy()
    
    results = backtest(
        strategy=strategy,
        T=cfg.T,
        Q0=cfg.Q0,
        S0=cfg.S0,
        sigma=cfg.sigma,
        eta=cfg.eta,
        gamma=cfg.gamma,
        n_paths=cfg.n_paths,
        seed=cfg.seed,
        track_details=False
    )
    
    print(f"\n{'='*60}")
    print(f"Strategy: {results['strategy']}")
    print(f"{'='*60}")
    print(f"Mean cost:    ${results['mean']:.4f}")
    print(f"Std dev:      ${results['std']:.4f}")
    print(f"P95:          ${results['p95']:.4f}")
    print(f"Skew:         {results['skew']:.4f}")
    print(f"Ex. Kurtosis: {results['excess_kurtosis']:.4f}")
    

# ============================================================================
# EXAMPLE 2: Compare multiple strategies
# ============================================================================

def compare_policies_fast():
    """Compare strategies on small backtest."""
    cfg = RunConfig()
    
    strategies = [
        TWAPStrategy(),
        OptimalDPStrategy(sigma=cfg.sigma, eta=cfg.eta, risk_lambda=cfg.risk_lambda),
        POVStrategy(alpha=0.2),
    ]
    
    results = compare_strategies(
        strategies=strategies,
        T=cfg.T,
        Q0=cfg.Q0,
        S0=cfg.S0,
        sigma=cfg.sigma,
        eta=cfg.eta,
        gamma=cfg.gamma,
        n_paths=5000,  # Smaller for quick iteration
        seed=cfg.seed
    )
    
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON")
    print(f"{'='*60}")
    
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Mean:  ${res['mean']:.4f}")
        print(f"  Std:   ${res['std']:.4f}")
        print(f"  P95:   ${res['p95']:.4f}")


# ============================================================================
# EXAMPLE 3: Create and test your own strategy
# ============================================================================

class MyCustomStrategy(Strategy):
    """Custom execution strategy - modify this!"""
    
    def get_execution_schedule(self, T: int, Q0: int) -> np.ndarray:
        """Your idea here.
        
        Example: Execute more aggressively when we have many steps remaining.
        """
        u = np.zeros((T, Q0 + 1), dtype=int)
        
        for t in range(T):
            for q in range(Q0 + 1):
                steps_left = T - t
                
                # Execute fraction that decreases as time runs out
                fraction = 1.0 / (steps_left + 1)
                trade = max(1, int(np.ceil(fraction * q)))
                u[t, q] = min(q, trade)
        
        return u


def test_custom_strategy():
    """Test your custom strategy."""
    cfg = RunConfig()
    
    strategy = MyCustomStrategy()
    
    results = backtest(
        strategy=strategy,
        T=cfg.T,
        Q0=cfg.Q0,
        S0=cfg.S0,
        sigma=cfg.sigma,
        eta=cfg.eta,
        gamma=cfg.gamma,
        n_paths=cfg.n_paths,
        seed=cfg.seed
    )
    
    print(f"\n{'='*60}")
    print(f"Custom Strategy Result")
    print(f"{'='*60}")
    print(f"Mean cost: ${results['mean']:.4f}")
    print(f"Std dev:   ${results['std']:.4f}")


# ============================================================================
# Run examples
# ============================================================================

if __name__ == "__main__":
    # Test 1: Single strategy
    print("\n[Test 1] Single Strategy")
    test_single_strategy()
    
    # Test 2: Compare multiple
    print("\n[Test 2] Policy Comparison")
    compare_policies_fast()
    
    # Test 3: Custom strategy
    print("\n[Test 3] Custom Strategy")
    test_custom_strategy()
    
    print("\n✓ All tests completed")
