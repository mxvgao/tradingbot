from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pairs_research.execution import Fill, PairOrder, replay_fills


@pytest.fixture
def scenario():
    times = pd.date_range("2026-01-06 16:00", periods=3, freq="min", tz="UTC")
    marks = pd.DataFrame({"time": times, "price_a": [100, 110, 120], "price_b": [100, 105, 110]})
    order = PairOrder(times[0] - pd.Timedelta(days=1), times[0], 10, -10, 100, 100)
    return order, marks, times


def test_full_simultaneous_fills_equal_benchmark(scenario):
    order, marks, times = scenario
    fills = [Fill("a", times[0], "a", 10, 100), Fill("b", times[0], "b", -10, 100)]
    ledger = replay_fills(order, marks, fills)
    assert ledger.execution_shortfall.eq(0).all()
    assert ledger.unpaired_gross_exposure.eq(0).all()
    assert ledger.complete.all()


def test_partial_delayed_leg_and_cost_decomposition(scenario):
    order, marks, times = scenario
    fills = [
        Fill("a1", times[0], "a", 5, 101, 1),
        Fill("a2", times[1], "a", 5, 110, 1),
        Fill("b1", times[2], "b", -10, 109, 1),
    ]
    ledger = replay_fills(order, marks, fills)
    assert ledger.iloc[0].remaining_shares_a == 5
    assert ledger.iloc[0].remaining_shares_b == -10
    assert ledger.iloc[0].unpaired_gross_exposure == 500
    assert ledger.iloc[1].unpaired_gross_exposure == 1100
    assert ledger.iloc[-1].unpaired_gross_exposure == 0
    assert ledger.iloc[-1].actual_gross_pnl == 135
    assert ledger.iloc[-1].execution_shortfall == 32
    np.testing.assert_allclose(
        ledger.execution_shortfall, ledger.legging_pnl + ledger.fill_price_pnl - ledger.fees
    )


def test_unfilled_order_and_residual_risk_preserved(scenario):
    order, marks, times = scenario
    rejected = replay_fills(order, marks, [])
    assert rejected.actual_net_pnl.eq(0).all()
    assert not rejected.complete.any()
    partial = replay_fills(order, marks, [Fill("a", times[0], "a", 10, 100)])
    assert partial.iloc[-1].gross_exposure == 1200
    assert partial.iloc[-1].remaining_shares_b == -10


@pytest.mark.parametrize(
    "kind", ["early", "overfill", "wrong_side", "duplicate", "unmarked", "negative_fee"]
)
def test_invalid_fill_rejected(scenario, kind):
    order, marks, times = scenario
    fill = Fill("id", times[0], "a", 10, 100)
    fills = {
        "early": [replace(fill, time=times[0] - pd.Timedelta(seconds=1))],
        "overfill": [replace(fill, shares=11)],
        "wrong_side": [replace(fill, shares=-1)],
        "duplicate": [fill, fill],
        "unmarked": [replace(fill, time=times[0] + pd.Timedelta(seconds=1))],
        "negative_fee": [replace(fill, fee=-1)],
    }[kind]
    with pytest.raises(ValueError):
        replay_fills(order, marks, fills)


def test_order_cannot_fill_at_signal_time(scenario):
    order, marks, _ = scenario
    with pytest.raises(ValueError, match="strictly later"):
        replay_fills(replace(order, signal_time=order.execution_time), marks, [])
