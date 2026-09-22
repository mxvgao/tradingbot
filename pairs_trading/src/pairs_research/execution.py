"""Independent, deterministic two-leg fill replay; never alters strategy P&L.

Prices/quantities must use the same units. Supply event-time marks for both legs
at every fill time and subsequent risk observation. This models supplied fills,
not their probability; live rejection/cancel/retry policy belongs to an adapter.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairOrder:
    signal_time: pd.Timestamp
    execution_time: pd.Timestamp
    shares_a: float
    shares_b: float
    benchmark_price_a: float
    benchmark_price_b: float


@dataclass(frozen=True)
class Fill:
    fill_id: str
    time: pd.Timestamp
    leg: str
    shares: float
    price: float
    fee: float = 0.0


def replay_fills(order: PairOrder, marks: pd.DataFrame, fills: list[Fill]) -> pd.DataFrame:
    """Compare actual partial fills against a simultaneous full-fill benchmark.

    marks: time, price_a, price_b. Empty fills represent rejection/non-fill.
    Quantities are signed incremental fills. Residuals remain unfilled; neither
    they nor filled exposure are silently flattened at the replay boundary.
    execution_shortfall = actual net P&L - benchmark gross P&L (negative is worse).
    legging_pnl is the cumulative effect of incomplete holdings between marks;
    fill_price_pnl captures fill-price differences from each contemporaneous mark.
    """
    signal_time, execution_time = (
        pd.Timestamp(order.signal_time),
        pd.Timestamp(order.execution_time),
    )
    if pd.isna(signal_time) or pd.isna(execution_time) or execution_time <= signal_time:
        raise ValueError("Execution must be strictly later than the completed signal")
    targets = {"a": order.shares_a, "b": order.shares_b}
    benchmarks = {"a": order.benchmark_price_a, "b": order.benchmark_price_b}
    if any(not math.isfinite(q) or q == 0 for q in targets.values()):
        raise ValueError("Both signed leg quantities must be finite and nonzero")
    if any(not math.isfinite(p) or p <= 0 for p in benchmarks.values()):
        raise ValueError("Benchmark prices must be finite and positive")
    frame = marks.copy()
    if frame.empty or not {"time", "price_a", "price_b"}.issubset(frame.columns):
        raise ValueError("Provide time and both leg prices for every risk observation")
    frame["time"] = pd.to_datetime(frame["time"])
    if (
        frame.time.isna().any()
        or frame.time.duplicated().any()
        or not frame.time.is_monotonic_increasing
        or frame.time.iloc[0] != execution_time
    ):
        raise ValueError("Marks must start at execution_time and be strictly increasing")
    prices = frame[["price_a", "price_b"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Marks must be finite and positive")
    if any(not np.isclose(frame.iloc[0][f"price_{leg}"], benchmarks[leg]) for leg in targets):
        raise ValueError("First marks must equal the full-fill benchmark prices")
    by_time = {}
    seen = set()
    mark_times = set(frame.time)
    for fill in fills:
        when = pd.Timestamp(fill.time)
        if not fill.fill_id or fill.fill_id in seen:
            raise ValueError("Fill IDs must be nonempty and unique")
        seen.add(fill.fill_id)
        if when not in mark_times or when < execution_time:
            raise ValueError("Each fill needs contemporaneous marks at or after execution_time")
        if (
            fill.leg not in targets
            or not math.isfinite(fill.shares)
            or fill.shares == 0
            or not math.isfinite(fill.price)
            or fill.price <= 0
            or not math.isfinite(fill.fee)
            or fill.fee < 0
        ):
            raise ValueError("Invalid fill leg, quantity, price or fee")
        if fill.shares * targets[fill.leg] <= 0:
            raise ValueError("Fill direction must match its order")
        by_time.setdefault(when, []).append(fill)
    held = {"a": 0.0, "b": 0.0}
    cash = fees = legging_pnl = fill_price_pnl = 0.0
    prior_prices = benchmarks
    rows = []
    for row in frame.itertuples(index=False):
        current = {"a": row.price_a, "b": row.price_b}
        legging_pnl += sum((held[k] - targets[k]) * (current[k] - prior_prices[k]) for k in held)
        for fill in by_time.get(row.time, []):
            leg = fill.leg
            if abs(held[leg] + fill.shares) > abs(targets[leg]) + 1e-9:
                raise ValueError("Fills exceed the ordered quantity")
            held[leg] += fill.shares
            cash -= fill.shares * fill.price
            fees += fill.fee
            fill_price_pnl += fill.shares * (current[leg] - fill.price)
        fractions = {k: held[k] / targets[k] for k in held}
        paired_fraction = min(fractions.values())
        unpaired = sum(abs(held[k] - paired_fraction * targets[k]) * current[k] for k in held)
        gross_pnl = cash + sum(held[k] * current[k] for k in held)
        benchmark_pnl = sum(targets[k] * (current[k] - benchmarks[k]) for k in held)
        rows.append(
            {
                "time": row.time,
                "filled_shares_a": held["a"],
                "filled_shares_b": held["b"],
                "remaining_shares_a": targets["a"] - held["a"],
                "remaining_shares_b": targets["b"] - held["b"],
                "fill_fraction_a": fractions["a"],
                "fill_fraction_b": fractions["b"],
                "unpaired_gross_exposure": unpaired,
                "net_exposure": sum(held[k] * current[k] for k in held),
                "gross_exposure": sum(abs(held[k] * current[k]) for k in held),
                "benchmark_gross_pnl": benchmark_pnl,
                "actual_gross_pnl": gross_pnl,
                "fees": fees,
                "actual_net_pnl": gross_pnl - fees,
                "execution_shortfall": gross_pnl - fees - benchmark_pnl,
                "legging_pnl": legging_pnl,
                "fill_price_pnl": fill_price_pnl,
                "complete": all(np.isclose(fractions[k], 1.0) for k in held),
            }
        )
        prior_prices = current
    return pd.DataFrame(rows)
