"""Compare TWAP vs DP strategies."""

from stochcontrol import TWAPStrategy, OptimalDPStrategy, compare_strategies, RunConfig

if __name__ == "__main__":
    cfg = RunConfig()

    print("\n" + "="*60)
    print("TWAP vs DP STRATEGY COMPARISON")
    print("="*60)

    # Define strategies to compare
    strategies = [
        TWAPStrategy(),
        OptimalDPStrategy(sigma=cfg.sigma, eta=cfg.eta, risk_lambda=cfg.risk_lambda),
    ]

    # Run comparison with smaller parameters for speed
    results = compare_strategies(
        strategies=strategies,
        T=20,  # Smaller T for faster DP computation
        Q0=50,  # Smaller inventory
        S0=cfg.S0,
        sigma=cfg.sigma,
        eta=cfg.eta,
        gamma=cfg.gamma,
        n_paths=1000,  # Fewer paths for speed
        seed=cfg.seed
    )

    print(f"Parameters: T={20}, Q0={50}, sigma={cfg.sigma:.3f}, eta={cfg.eta:.3f}")
    print(f"Risk lambda: {cfg.risk_lambda}")
    print()

    print(f"{'Strategy':<20} {'Mean Cost':>10} {'Std Dev':>10} {'P95':>10} {'P99':>10}")
    print("-" * 70)

    for name, res in results.items():
        print(f"{name:<20} ${res['mean']:>8.2f} ${res['std']:>8.2f} ${res['p95']:>8.2f} ${res['p99']:>8.2f}")

    # Show which is better
    twap_cost = results['TWAPStrategy']['mean']
    dp_cost = results['OptimalDPStrategy']['mean']

    print()
    print("COMPARISON:")
    if dp_cost < twap_cost:
        savings = twap_cost - dp_cost
        print(f"✓ DP is better by ${savings:.2f} on average")
        print("  (DP optimally balances impact costs vs holding risk)")
    else:
        savings = dp_cost - twap_cost
        print(f"✓ TWAP is better by ${savings:.2f} on average")
        print("  (TWAP performs well in this market regime)")

    print("\n" + "="*60)
    print("✓ Comparison completed")
    print("="*60)
