"""Quick test file for strategy testing."""

from strategy import TWAPStrategy, ImmediateStrategy, POVStrategy
from backtest import backtest, compare_strategies
from config import RunConfig

if __name__ == "__main__":
    cfg = RunConfig()
    
    print("\n" + "="*60)
    print("QUICK STRATEGY TEST")
    print("="*60)
    
    # Test 1: Single strategy
    print("\n[Test 1] Testing TWAP Strategy")
    print("-" * 60)
    results = backtest(
        strategy=TWAPStrategy(),
        T=cfg.T,
        Q0=cfg.Q0,
        S0=cfg.S0,
        sigma=cfg.sigma,
        eta=cfg.eta,
        gamma=cfg.gamma,
        n_paths=1000,
        seed=cfg.seed
    )
    print(f"Mean cost:    ${results['mean']:.2f}")
    print(f"Std dev:      ${results['std']:.2f}")
    print(f"P95:          ${results['p95']:.2f}")
    print(f"P99:          ${results['p99']:.2f}")
    
    # Test 2: Compare strategies
    print("\n[Test 2] Comparing Strategies")
    print("-" * 60)
    strategies = [
        TWAPStrategy(),
        ImmediateStrategy(),
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
        n_paths=1000,
        seed=cfg.seed
    )
    
    for name, res in results.items():
        print(f"{name:20s}: ${res['mean']:8.2f} (±${res['std']:8.2f})")
    
    print("\n" + "="*60)
    print("✓ Tests completed")
    print("="*60 + "\n")
