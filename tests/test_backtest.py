from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pairs_research.backtest_pair import BacktestConfig, compute_walk_forward_signals, run_backtest
from pairs_research.walk_forward_portfolio import WalkForwardConfig, backtest_pair_on_prices


def signals(z, prices=None, dates=None):
    n = len(z)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=n)
            if dates is None
            else pd.to_datetime(dates),
            "ticker_a_price": [100.0] * n if prices is None else prices,
            "ticker_b_price": [100.0] * n,
            "zscore": z,
            "spread": z,
            "hedge_ratio": [1.0] * n,
        }
    )


@pytest.fixture
def config():
    return BacktestConfig(ticker_a="A", ticker_b="B", round_trip_cost_bps=0)


def test_next_close_fixed_quantities_and_delayed_exit(config):
    frame = signals([3, 0, 0, 0], [100, 120, 110, 105])
    trades, daily, summary = run_backtest(frame, config)
    trade = trades.iloc[0]
    assert trade.entry_signal_date == frame.date[0]
    assert trade.entry_date == frame.date[1]
    assert trade.exit_signal_date == frame.date[1]
    assert trade.exit_date == frame.date[2]
    assert trade.shares_a == -50  # Sized at 100, not at next close of 120.
    assert trade.entry_ticker_a_price == 120
    assert daily.net_daily_pnl.tolist() == [0, 0, 500, 0]
    assert summary.total_pnl_dollars.iloc[0] == 500


@pytest.mark.parametrize("direction", [1, -1])
def test_mark_to_market_and_boundary_reconcile(config, direction):
    cfg = replace(config, round_trip_cost_bps=4)
    frame = signals([-3 * direction] * 4, [100, 100, 110, 120])
    trades, daily, summary = run_backtest(frame, cfg)
    trade = trades.iloc[0]
    assert trade.exit_reason == "test_boundary"
    assert pd.isna(trade.exit_signal_date)
    assert trade.sessions_held == 2
    assert daily.iloc[-1].position == 0
    assert daily.iloc[-1].gross_exposure == 0
    assert daily.iloc[-1].unrealized_pnl == 0
    assert daily.iloc[2].gross_exposure == 10500
    assert daily.iloc[2].unrealized_pnl == pytest.approx(direction * 500 - 2)
    assert trade.exit_cost_dollars == pytest.approx(2.2)
    assert trade.pnl_dollars == pytest.approx(direction * 1000 - 4.2)
    np.testing.assert_allclose(daily.realized_pnl + daily.unrealized_pnl, daily.cumulative_pnl)
    assert daily.net_daily_pnl.sum() == pytest.approx(trades.pnl_dollars.sum())
    assert summary.total_pnl_dollars.iloc[0] == pytest.approx(trades.pnl_dollars.sum())
    assert daily.daily_return.iloc[2] == pytest.approx(direction * 500 / 99998)


def test_holding_limit_counts_sessions_across_holiday(config):
    frame = signals(
        [3] * 5, dates=["2026-01-15", "2026-01-16", "2026-01-20", "2026-01-21", "2026-01-22"]
    )
    trades, _, _ = run_backtest(frame, replace(config, max_holding_sessions=1))
    assert trades.iloc[0].entry_date == pd.Timestamp("2026-01-16")
    assert trades.iloc[0].exit_date == pd.Timestamp("2026-01-20")
    assert trades.iloc[0].sessions_held == 1
    assert trades.iloc[0].exit_reason == "max_holding_sessions"


def test_stop_is_delayed(config):
    frame = signals([3, 3, 5, 3, 3])
    trades, _, _ = run_backtest(frame, replace(config, stop_z=4))
    assert trades.iloc[0].exit_signal_date == frame.date[2]
    assert trades.iloc[0].exit_date == frame.date[3]
    assert trades.iloc[0].exit_reason == "stop_z"


def test_reentry_block_until_neutral(config):
    frame = signals([3, 3, 3, 3, 0, 3, 3, 3, 3])
    trades, _, _ = run_backtest(
        frame, replace(config, max_holding_sessions=1, block_reentry_after_max_hold=True)
    )
    assert trades.entry_date.tolist() == [frame.date[1], frame.date[6]]


@pytest.mark.parametrize("z", [[], [3], [3, 3], [0, 0, 3], [0, 3, 3]])
def test_no_final_close_entry_and_no_trade_schema(config, z):
    trades, daily, summary = run_backtest(signals(z), config)
    assert trades.empty
    assert summary.completed_trades.iloc[0] == 0
    assert summary.total_return.iloc[0] == 0
    assert summary.sharpe.iloc[0] == 0
    if not daily.empty:
        assert daily.position.eq(0).all()


@pytest.mark.parametrize("flag", [False, np.nan, None])
def test_entry_filters_fail_closed(config, flag):
    frame = signals([3] * 5)
    frame["regime_allowed"] = flag
    assert run_backtest(frame, replace(config, require_regime_allowed=True))[0].empty
    assert run_backtest(frame, replace(config, require_rolling_pass=True))[0].empty


def test_filter_and_hedge_snapshot_from_signal_day(config):
    frame = signals([3, 3, 3, 3])
    frame["regime_allowed"] = [True, False, False, False]
    frame["hedge_ratio"] = [2, 10, 10, 10]
    trades, _, _ = run_backtest(frame, replace(config, require_regime_allowed=True))
    assert trades.iloc[0].entry_hedge_ratio == 2
    assert trades.iloc[0].shares_b == pytest.approx(200 / 3)


@pytest.mark.parametrize("bad", [np.nan, 0, -1, np.inf])
def test_invalid_prices_rejected(config, bad):
    frame = signals([3] * 4)
    frame.loc[2, "ticker_a_price"] = bad
    with pytest.raises(ValueError, match="positive prices"):
        run_backtest(frame, config)


def test_duplicate_or_unsorted_dates_rejected(config):
    frame = signals([3] * 4)
    with pytest.raises(ValueError, match="increasing"):
        run_backtest(frame.iloc[::-1], config)
    frame.loc[1, "date"] = frame.date[0]
    with pytest.raises(ValueError, match="unique"):
        run_backtest(frame, config)


def test_nan_z_still_marks_and_liquidates(config):
    trades, daily, _ = run_backtest(
        signals([3, np.nan, np.nan, np.nan], [100, 100, 110, 120]), config
    )
    assert trades.iloc[0].exit_reason == "test_boundary"
    assert daily.iloc[-1].cumulative_pnl == -1000


def test_test_window_starts_flat_and_ends_flat():
    frame = signals([3] * 8)
    trades, daily, _ = backtest_pair_on_prices(
        pd.DataFrame(),
        "A",
        "B",
        WalkForwardConfig(),
        start_date=frame.date[3],
        end_date=frame.date[6],
        precomputed_signals=frame,
    )
    assert trades.iloc[0].entry_signal_date == frame.date[3]
    assert trades.iloc[0].entry_date == frame.date[4]
    assert trades.iloc[0].exit_date == frame.date[6]
    assert daily.iloc[0].position == daily.iloc[-1].position == 0


def price_history():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=70)
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    a = b * np.exp(rng.normal(0, 0.015, len(dates)))
    return pd.concat(
        [
            pd.DataFrame({"date": dates, "ticker": t, "adj_close": p})
            for t, p in [("A", a), ("B", b)]
        ],
        ignore_index=True,
    )


def test_future_prices_cannot_change_past_signals(config):
    prices = price_history()
    cfg = replace(config, formation_days=20)
    before = compute_walk_forward_signals(prices, cfg)
    cutoff = before.date.iloc[20]
    prices.loc[prices.date > cutoff, "adj_close"] *= 2
    after = compute_walk_forward_signals(prices, cfg)
    pd.testing.assert_frame_equal(before[before.date <= cutoff], after[after.date <= cutoff])


def test_missing_leg_does_not_compress_session_count(config):
    prices = price_history()
    prices.loc[10, "adj_close"] = np.nan
    with pytest.raises(ValueError, match="Missing or invalid"):
        compute_walk_forward_signals(prices, replace(config, formation_days=5))


@pytest.mark.parametrize(
    "kw",
    [
        {"max_holding_sessions": 0},
        {"max_holding_sessions": 1.5},
        {"entry_z": 0},
        {"round_trip_cost_bps": -1},
        {"stop_z": 1},
    ],
)
def test_config_validation(kw):
    with pytest.raises(ValueError):
        BacktestConfig(**kw)


def test_missing_trailing_leg_is_an_error(config):
    prices = price_history()
    prices.loc[69, "adj_close"] = np.nan
    with pytest.raises(ValueError, match="Missing or invalid"):
        compute_walk_forward_signals(prices, replace(config, formation_days=5))


def test_empty_csv_outputs_remain_readable(tmp_path, config):
    from pairs_research.backtest_pair import write_backtest_outputs

    prices = price_history()
    prices = prices[prices.date <= prices.date.min()]
    source = tmp_path / "prices.csv"
    prices.to_csv(source, index=False)
    paths = write_backtest_outputs(source, tmp_path, config)
    trades, daily, summary = [pd.read_csv(path) for path in paths]
    assert trades.empty and daily.empty
    assert summary.completed_trades.iloc[0] == 0


def test_overlapping_portfolio_windows_rejected():
    with pytest.raises(ValueError, match="double-count"):
        WalkForwardConfig(test_days=20, step_days=10)


def test_both_missing_legs_cannot_remove_a_session(config):
    prices = price_history()
    prices.loc[prices.date == prices.date.iloc[30], "adj_close"] = np.nan
    with pytest.raises(ValueError, match="Missing or invalid"):
        compute_walk_forward_signals(prices, replace(config, formation_days=5))


def test_duplicate_raw_prices_rejected(config):
    prices = price_history()
    prices = pd.concat([prices, prices.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        compute_walk_forward_signals(prices, config)
