"""Compare TWAP vs DP with different risk levels."""

from stochcontrol import TWAPStrategy, OptimalDPStrategy, compare_strategies, RunConfig

if __name__ == "__main__":
    cfg = RunConfig()

    print("\n" + "="*70)
    print("TWAP vs DP WITH DIFFERENT RISK LEVELS")
    print("="*70)

    # Test different risk lambdas for DP
    risk_lambdas = [0.0, 0.01, 0.1, 0.5, 1.0]

    print(f"Parameters: T=20, Q0=50, sigma={cfg.sigma:.3f}, eta={cfg.eta:.3f}")
    print()

    print(f"{'Strategy':<25} {'Risk λ':>8} {'Mean Cost':>10} {'Std Dev':>10}")
    print("-" * 65)

    # TWAP baseline
    twap_result = compare_strategies(
        strategies=[TWAPStrategy()],
        T=20, Q0=50, S0=cfg.S0, sigma=cfg.sigma, eta=cfg.eta, gamma=cfg.gamma,
        n_paths=1000, seed=cfg.seed
    )
    twap_cost = twap_result['TWAPStrategy']['mean']
    twap_std = twap_result['TWAPStrategy']['std']

    print(f"{'TWAPStrategy':<25} {'N/A':>8} ${twap_cost:>8.2f} ${twap_std:>8.2f}")

    # DP with different risk levels
    for risk_lambda in risk_lambdas:
        dp_result = compare_strategies(
            strategies=[OptimalDPStrategy(sigma=cfg.sigma, eta=cfg.eta, risk_lambda=risk_lambda)],
            T=20, Q0=50, S0=cfg.S0, sigma=cfg.sigma, eta=cfg.eta, gamma=cfg.gamma,
            n_paths=1000, seed=cfg.seed
        )

        dp_cost = dp_result['OptimalDPStrategy']['mean']
        dp_std = dp_result['OptimalDPStrategy']['std']

        print(f"{'OptimalDPStrategy':<25} {risk_lambda:>8.1f} ${dp_cost:>8.2f} ${dp_std:>8.2f}")

        # Show comparison
        diff = dp_cost - twap_cost
        if abs(diff) > 1.0:  # Only show meaningful differences
            status = "better" if diff < 0 else "worse"
            print(f"{'':<25} {'':>8} {'':>10} {'':>10} (DP {status} by ${abs(diff):.1f})")

    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("• λ=0.0: DP ignores risk, trades aggressively (like Immediate)")
    print("• λ=0.1: DP balances impact vs risk")
    print("• λ=1.0+: DP becomes very risk-averse, trades slowly")
    print("• TWAP: Fixed schedule, no risk optimization")
    print("="*70)
