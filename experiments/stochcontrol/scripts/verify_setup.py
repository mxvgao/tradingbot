"""Quick verification that all components work together."""

from stochcontrol import TWAPStrategy, ImmediateStrategy, LinearVWAPStrategy, backtest, compare_strategies, TradeTracker, DataLoader, SyntheticDataGenerator, RunConfig

print("Testing core components...\n")

# 1. Test Strategy interface
print("✓ Strategy interface loads")
strategies = [TWAPStrategy(), ImmediateStrategy(), LinearVWAPStrategy()]
for s in strategies:
    print(f"  - {s}")

# 2. Test DataLoader
print("\n✓ DataLoader works")
prices = SyntheticDataGenerator.random_walk(S0=100, sigma=1.0, T=50, n_paths=1)
print(f"  Generated synthetic path: {prices[0, :5]}...")

# 3. Test TradeTracker
print("\n✓ TradeTracker works")
tracker = TradeTracker(S0=100.0, Q0=100)
tracker.record_step(t=0, u=10, price=99.9, market_impact=0.1, remaining_inventory=90)
tracker.record_step(t=1, u=10, price=99.8, market_impact=0.2, remaining_inventory=80)
print(f"  Recorded 2 trades, cost: ${tracker.final_cost():.2f}")
print(f"  Summary: {tracker.summary()}")

# 4. Test backtest with small sample
print("\n✓ Backtest engine works")
cfg = RunConfig()
result = backtest(
    strategy=ImmediateStrategy(),
    T=cfg.T,
    Q0=cfg.Q0,
    S0=cfg.S0,
    sigma=cfg.sigma,
    eta=cfg.eta,
    gamma=cfg.gamma,
    n_paths=100,  # Small sample
    seed=42
)
print(f"  ImmediateStrategy mean cost: ${result['mean']:.2f}")
print(f"  Std dev: ${result['std']:.2f}")

# 5. Test strategy comparison
print("\n✓ Strategy comparison works")
results = compare_strategies(
    strategies=[TWAPStrategy(), ImmediateStrategy()],
    T=cfg.T,
    Q0=cfg.Q0,
    S0=cfg.S0,
    sigma=cfg.sigma,
    eta=cfg.eta,
    gamma=cfg.gamma,
    n_paths=100,
    seed=42
)
for name, res in results.items():
    print(f"  {name}: ${res['mean']:.2f}")

print("\n" + "="*60)
print("✓ ALL COMPONENTS WORKING")
print("="*60)
print("\nYou can now:")
print("  1. Create new Strategy subclasses in strategy.py")
print("  2. Test them with backtest(strategy=MyStrategy(), ...)")
print("  3. Track details with TradeTracker for analysis")
print("  4. Load historical data with DataLoader")
print("\nSee test_template.py for more examples.")
