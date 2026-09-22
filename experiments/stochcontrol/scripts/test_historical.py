"""Test strategies using real historical data."""

from stochcontrol import DataLoader, TWAPStrategy, ImmediateStrategy, POVStrategy, backtest
import numpy as np

def test_with_historical_data():
    """Test strategies using real AAPL price data."""

    print("\n" + "="*70)
    print("TESTING WITH REAL HISTORICAL DATA (AAPL)")
    print("="*70)

    # Load historical data
    loader = DataLoader('aapl_historical.csv')
    print(f"✓ Loaded {len(loader)} trading days")
    print(f"✓ Estimated volatility: {loader.estimate_volatility():.4f}")

    # Get price path (first 50 trading days)
    price_path = loader.get_price_segment(0, 50)
    print(f"✓ Using first {len(price_path)} days: ${price_path[0]:.2f} → ${price_path[-1]:.2f}")

    # Test strategies on this real price path
    strategies = [
        TWAPStrategy(),
        ImmediateStrategy(),
        POVStrategy(alpha=0.2),
    ]

    print(f"\n{'Strategy':<20} {'Mean Cost':>10} {'Std Dev':>10} {'P95':>10}")
    print("-" * 60)

    for strategy in strategies:
        results = backtest(
            strategy=strategy,
            T=len(price_path),
            Q0=100,
            S0=price_path[0],
            sigma=loader.estimate_volatility(),
            eta=0.01,
            gamma=0.0,
            n_paths=1000,
            seed=42
        )

        print(f"{strategy.__class__.__name__:<20} ${results['mean']:>8.2f} ${results['std']:>8.2f} ${results['p95']:>8.2f}")

def compare_real_vs_synthetic():
    """Compare performance on real vs synthetic data."""

    print("\n" + "="*70)
    print("REAL VS SYNTHETIC DATA COMPARISON")
    print("="*70)

    # Load real data
    loader = DataLoader('aapl_historical.csv')
    price_path = loader.get_price_segment(0, 50)
    real_volatility = loader.estimate_volatility()

    print(f"Real data: {len(price_path)} days, volatility = {real_volatility:.4f}")

    # Test on real data
    strategy = TWAPStrategy()
    real_results = backtest(
        strategy=strategy,
        T=len(price_path),
        Q0=100,
        S0=price_path[0],
        sigma=real_volatility,
        eta=0.01,
        gamma=0.0,
        n_paths=1000,
        seed=42
    )

    # Test on synthetic data with same parameters
    synthetic_results = backtest(
        strategy=strategy,
        T=50,
        Q0=100,
        S0=price_path[0],  # Same starting price
        sigma=real_volatility,  # Same volatility
        eta=0.01,
        gamma=0.0,
        n_paths=1000,
        seed=42
    )

    print(f"\nTWAP Strategy Results:")
    print(f"{'Data Type':<15} {'Mean Cost':>10} {'Std Dev':>10} {'P95':>10}")
    print("-" * 50)
    print(f"{'Real':<15} ${real_results['mean']:>8.2f} ${real_results['std']:>8.2f} ${real_results['p95']:>8.2f}")
    print(f"{'Synthetic':<15} ${synthetic_results['mean']:>8.2f} ${synthetic_results['std']:>8.2f} ${synthetic_results['p95']:>8.2f}")

    # Show actual price path vs synthetic
    print(f"\nReal price path (first 10 days):")
    print([f"{p:.2f}" for p in price_path[:10]])

    # Generate synthetic path for comparison
    rng = np.random.default_rng(42)
    synthetic_path = np.zeros(11)
    synthetic_path[0] = price_path[0]
    for t in range(10):
        dW = rng.standard_normal()
        synthetic_path[t + 1] = synthetic_path[t] * np.exp(
            -0.5 * real_volatility**2 + real_volatility * np.sqrt(1.0) * dW
        )
    print(f"Synthetic path (first 10 days):")
    print([f"{p:.2f}" for p in synthetic_path[:10]])

if __name__ == "__main__":
    test_with_historical_data()
    compare_real_vs_synthetic()

    print("\n" + "="*70)
    print("✓ Historical data testing completed")
    print("="*70)
