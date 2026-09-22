from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pairs_research.backtest_pair import BacktestConfig, compute_walk_forward_signals, run_backtest
from pairs_research.config import ENGINE_VERSION
from pairs_research.engine import SimulatedCloseExecution, process_session
from pairs_research.ledger import MemoryLedger, SQLiteLedger
from pairs_research.models import RunSpec, Session, StrategyState
from pairs_research.replay import replay, sessions_from_prices, sessions_from_signals
from pairs_research.strategy import generate_target
from test_backtest import price_history, signals


@pytest.fixture
def scenario():
    frame = signals(
        [-3, -3, 0, 0, 3, 3, 5, 5, 0, -3, -3, -3, -3, 0, 3, 3],
        [100, 101, 110, 106, 101, 100, 105, 110, 108, 104, 100, 99, 102, 110, 108, 107],
    )
    cfg = BacktestConfig(ticker_a="A", ticker_b="B", max_holding_sessions=3, stop_z=4)
    return frame, cfg


def spec_for(frame, cfg, liquidate=False):
    terminal = str(frame.date.iloc[-1].date()) if liquidate else None
    return RunSpec("test-run", cfg, "signals", terminal)


def run_file(path, spec, frame):
    with SQLiteLedger(path, spec) as ledger:
        replay(ledger, sessions_from_signals(frame))
        return ledger.snapshot()


@pytest.mark.parametrize("z,target", [(-3, 1), (3, -1), (0, 0)])
def test_pure_target_does_not_mutate_or_create_orders(z, target):
    cfg = BacktestConfig()
    history = list(sessions_from_signals(signals([z])))
    state = StrategyState.initial(cfg)
    state.session_index = 0
    original = state.to_json()
    decision = generate_target(history=history, state=state, config=cfg)
    assert decision.target_position == target
    assert decision.reason == ("entry" if target else "hold")
    assert not hasattr(decision, "order_id")
    assert state.to_json() == original


def test_existing_next_close_economics_preserved():
    fixture = json.loads((Path(__file__).parent / "fixtures/next_close_baseline.json").read_text())
    frame = pd.DataFrame(fixture["signals"])
    frame.date = pd.to_datetime(frame.date)
    trades, daily, _ = run_backtest(frame, BacktestConfig(**fixture["config"]))
    for actual, expected_rows, dates in [
        (
            trades,
            fixture["trades"],
            ("entry_signal_date", "entry_date", "exit_signal_date", "exit_date"),
        ),
        (daily, fixture["daily"], ("date", "executed_signal_date")),
    ]:
        expected = pd.DataFrame(expected_rows)
        for name in dates:
            expected[name] = pd.to_datetime(expected[name])
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("liquidate", [False, True])
def test_memory_batch_matches_sqlite_economics(tmp_path, scenario, liquidate):
    frame, cfg = scenario
    trades, daily, _ = run_backtest(frame, cfg, liquidate_at_end=liquidate)
    snapshot = run_file(tmp_path / "ledger.db", spec_for(frame, cfg, liquidate), frame)
    recorded_daily = pd.DataFrame([json.loads(row["daily_json"]) for row in snapshot["positions"]])
    for column in ("date", "executed_signal_date"):
        recorded_daily[column] = pd.to_datetime(recorded_daily[column])
    pd.testing.assert_frame_equal(daily, recorded_daily[daily.columns], check_dtype=False)
    recorded_trades = pd.DataFrame([json.loads(row["trade_json"]) for row in snapshot["trades"]])
    for column in ("entry_signal_date", "entry_date", "exit_signal_date", "exit_date"):
        recorded_trades[column] = pd.to_datetime(recorded_trades[column])
    pd.testing.assert_frame_equal(trades, recorded_trades[trades.columns], check_dtype=False)
    orders = {row["order_id"]: row for row in snapshot["orders"]}
    for fill in snapshot["fills"]:
        order = orders[fill["order_id"]]
        if order["order_purpose"] != "boundary":
            assert fill["fill_session"] > order["signal_session"]


@pytest.mark.parametrize("liquidate", [False, True])
def test_restart_after_every_session_matches_uninterrupted(tmp_path, scenario, liquidate):
    frame, cfg = scenario
    spec = spec_for(frame, cfg, liquidate)
    expected = run_file(tmp_path / "reference.db", spec, frame)
    for split in range(1, len(frame) + 1):
        path = tmp_path / f"split-{split}.db"
        run_file(path, spec, frame.iloc[:split])
        # Restart from the full source: committed sessions must be no-ops.
        actual = run_file(path, spec, frame)
        assert actual == expected
        assert run_file(path, spec, frame) == expected


def test_database_constraints_enforce_event_uniqueness(tmp_path, scenario):
    frame, cfg = scenario
    with SQLiteLedger(tmp_path / "unique.db", spec_for(frame, cfg)) as ledger:
        replay(ledger, sessions_from_signals(frame))
        before = ledger.snapshot()
        for table in ("signals", "orders", "fills", "positions", "checkpoints"):
            with pytest.raises(sqlite3.IntegrityError):
                ledger.connection.execute(f"INSERT INTO {table} SELECT * FROM {table} LIMIT 1")
        assert ledger.snapshot() == before
        assert replay(ledger, sessions_from_signals(frame))["processed"] == 0
        assert ledger.snapshot() == before


def test_sql_rejects_same_signal_session_fill(tmp_path):
    frame = signals([-3, -3])
    with SQLiteLedger(tmp_path / "early.db", spec_for(frame, BacktestConfig())) as ledger:
        process_session(ledger, next(sessions_from_signals(frame)))
        order = ledger.snapshot()["orders"][0]
        with pytest.raises(sqlite3.IntegrityError, match="fill must follow"):
            ledger.connection.execute(
                "INSERT INTO fills VALUES (?,?,?,?,?,?,?)",
                ("early", order["order_id"], order["signal_session"], "a", 1, 100, 0),
            )
        assert not ledger.snapshot()["fills"]


@pytest.mark.parametrize("change", ["config", "engine", "strategy", "execution", "policy", "input"])
def test_existing_run_rejects_incompatible_configuration(tmp_path, scenario, change):
    frame, cfg = scenario
    spec = spec_for(frame, cfg)
    path = tmp_path / "versioned.db"
    original = run_file(path, spec, frame.iloc[:2])
    changes = {
        "config": {"config": replace(cfg, entry_z=2.5)},
        "engine": {"engine_version": "v-next"},
        "strategy": {"strategy_version": "v-next"},
        "execution": {"execution_version": "v-next"},
        "policy": {"liquidation_session": str(frame.date.iloc[-1].date())},
        "input": {"input_kind": "prices"},
    }
    with pytest.raises(ValueError, match="mismatch"):
        SQLiteLedger(path, replace(spec, **changes[change]))
    with SQLiteLedger(path, spec) as ledger:
        assert ledger.snapshot() == original
    with SQLiteLedger(path, replace(spec, strategy_id="another-run", **changes[change])) as ledger:
        assert ledger.load_state().last_processed_session is None


def test_fault_before_checkpoint_rolls_back_fills_orders_and_position(tmp_path, scenario):
    frame, cfg = scenario
    spec = spec_for(frame, cfg)
    with SQLiteLedger(tmp_path / "failure.db", spec) as ledger:
        process_session(ledger, next(sessions_from_signals(frame)))
        before = ledger.snapshot()
        ledger.connection.execute(
            "CREATE TEMP TRIGGER fail_checkpoint BEFORE UPDATE ON checkpoints BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
            process_session(ledger, list(sessions_from_signals(frame))[1])
        assert ledger.snapshot() == before
        ledger.connection.execute("DROP TRIGGER fail_checkpoint")
        replay(ledger, sessions_from_signals(frame))
        assert ledger.snapshot() == run_file(tmp_path / "expected.db", spec, frame)


def test_missing_price_does_not_commit_failed_session_and_recovers(tmp_path):
    prices = price_history()
    cfg = BacktestConfig(ticker_a="A", ticker_b="B", formation_days=10, entry_z=1)
    spec = RunSpec("raw", cfg)
    bad_day = prices.date.iloc[30]
    broken = prices[~((prices.date == bad_day) & (prices.ticker == "A"))]
    with SQLiteLedger(tmp_path / "missing.db", spec) as ledger:
        with pytest.raises(ValueError, match="exactly one price"):
            replay(ledger, sessions_from_prices(broken, cfg))
        snapshot = ledger.snapshot()
        assert len(snapshot["signals"]) == 30
        assert snapshot["checkpoints"][0]["last_processed_session"] == str(
            prices.date.iloc[29].date()
        )
        assert not any(row["session"] == str(bad_day.date()) for row in snapshot["positions"])
        replay(ledger, sessions_from_prices(prices, cfg))
        actual = ledger.snapshot()
    with SQLiteLedger(tmp_path / "complete.db", spec) as ledger:
        replay(ledger, sessions_from_prices(prices, cfg))
        assert ledger.snapshot() == actual


def test_future_prices_never_reach_decision_engine(tmp_path, monkeypatch):
    from pairs_research import engine

    prices = price_history()
    cutoff = prices.date.iloc[35]
    cfg = BacktestConfig(ticker_a="A", ticker_b="B", formation_days=10, entry_z=1)
    seen = []
    original = engine.generate_target

    def spy(history, state, config):
        assert all(s.date <= history[-1].date for s in history)
        assert len(history) <= config.formation_days + 1
        seen.append(history[-1].date)
        return original(history, state, config)

    monkeypatch.setattr(engine, "generate_target", spy)
    changed = prices.copy()
    changed.loc[changed.date > cutoff, "adj_close"] *= 100
    snapshots = []
    for i, source in enumerate([prices, changed]):
        with SQLiteLedger(tmp_path / f"future-{i}.db", RunSpec("causal", cfg)) as ledger:
            replay(ledger, sessions_from_prices(source, cfg), stop_after=36)
            snapshots.append(ledger.snapshot())
    assert seen and max(seen) == str(cutoff.date())
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("side", [-1, 1])
def test_final_asof_mark_and_optional_liquidation_reconcile(tmp_path, side):
    frame = signals([-3 * side] * 4, [100, 100, 110, 120])
    cfg = BacktestConfig(ticker_a="A", ticker_b="B")
    for liquidate in (False, True):
        spec = spec_for(frame, cfg, liquidate)
        with SQLiteLedger(tmp_path / f"terminal-{liquidate}.db", spec) as ledger:
            replay(ledger, sessions_from_signals(frame))
            snapshot, state = ledger.snapshot(), ledger.load_state()
            position = snapshot["positions"][-1]
            cash = cfg.initial_capital - sum(
                f["shares"] * f["price"] + f["fee"] for f in snapshot["fills"]
            )
            assert position["cash"] == pytest.approx(cash)
            assert position["equity"] == pytest.approx(
                cash + position["shares_a"] * 120 + position["shares_b"] * 100
            )
            assert position["realized_pnl"] + position["unrealized_pnl"] == pytest.approx(
                position["equity"] - cfg.initial_capital
            )
            assert state.position == (0 if liquidate else side)
            assert bool(snapshot["trades"]) == liquidate
            if liquidate:
                assert snapshot["orders"][-1]["order_purpose"] == "boundary"
                assert position["cash"] == pytest.approx(position["equity"])
                assert state.pending is None
            else:
                assert position["unrealized_pnl"] == pytest.approx(side * 1000 - 2)


def test_raw_replay_and_batch_share_observations_and_economics(tmp_path):
    prices = price_history()
    cfg = BacktestConfig(
        ticker_a="A", ticker_b="B", formation_days=10, entry_z=1, max_holding_sessions=3
    )
    features = compute_walk_forward_signals(prices, cfg)
    _, batch, _ = run_backtest(features, cfg, liquidate_at_end=False)
    with SQLiteLedger(tmp_path / "raw.db", RunSpec("raw", cfg)) as ledger:
        replay(ledger, sessions_from_prices(prices, cfg))
        rows = ledger.snapshot()["positions"][cfg.formation_days :]
        recorded = pd.DataFrame([json.loads(row["daily_json"]) for row in rows])
        np.testing.assert_allclose(recorded.equity, batch.equity)
        np.testing.assert_allclose(recorded.zscore, batch.zscore)
        assert recorded.action.tolist() == batch.action.tolist()


def test_replay_cli_restart_and_terminal_policy(tmp_path, scenario):
    frame, _ = scenario
    source = tmp_path / "signals.csv"
    frame.to_csv(source, index=False)
    cmd = [
        sys.executable,
        "-m",
        "pairs_research.replay",
        "--database",
        str(tmp_path / "cli.db"),
        "--run-id",
        "cli",
        "--signals",
        str(source),
        "--ticker-a",
        "A",
        "--ticker-b",
        "B",
        "--liquidate-at-end",
    ]
    first = subprocess.run(cmd + ["--stop-after", "2"], check=True, capture_output=True, text=True)
    assert json.loads(first.stdout)["processed"] == 2
    second = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert json.loads(second.stdout)["processed"] == len(frame) - 2
    third = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert json.loads(third.stdout)["processed"] == 0
    assert json.loads(third.stdout)["position"] == 0
    mismatch = subprocess.run(cmd + ["--entry-z", "2.5"], capture_output=True, text=True)
    assert mismatch.returncode != 0
    assert "mismatch" in mismatch.stderr


def test_default_asof_run_can_extend_and_execute_pending_entry(tmp_path):
    frame = signals([-3, -3, -3], [100, 105, 110])
    spec = spec_for(frame, BacktestConfig())
    path = tmp_path / "extend.db"
    with SQLiteLedger(path, spec) as ledger:
        replay(ledger, sessions_from_signals(frame.iloc[:1]))
        assert ledger.load_state().pending.order_purpose == "entry"
        assert not ledger.snapshot()["fills"]
    with SQLiteLedger(path, spec) as ledger:
        replay(ledger, sessions_from_signals(frame.iloc[1:]))
        assert ledger.load_state().position == 1
        assert len(ledger.snapshot()["fills"]) == 2
        assert not ledger.snapshot()["trades"]


def test_liquidation_boundary_cancels_last_close_entry_and_disallows_extension(tmp_path):
    frame = signals([-3] * 3)
    spec = spec_for(frame.iloc[:2], BacktestConfig(), True)
    with SQLiteLedger(tmp_path / "bounded.db", spec) as ledger:
        replay(ledger, sessions_from_signals(frame.iloc[:2]))
        before = ledger.snapshot()
        assert before["orders"][0]["status"] == "cancelled"
        assert not before["fills"]
        assert ledger.load_state().position == 0
        with pytest.raises(ValueError, match="after a terminal"):
            process_session(ledger, list(sessions_from_signals(frame))[2])
        assert ledger.snapshot() == before


def test_concurrent_duplicate_processing_commits_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    frame = signals([-3])
    spec = spec_for(frame, BacktestConfig())
    path = tmp_path / "race.db"
    with SQLiteLedger(path, spec):
        pass

    def worker(_):
        with SQLiteLedger(path, spec) as ledger:
            return replay(ledger, sessions_from_signals(frame))["processed"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(worker, [0, 1])) == [0, 1]
    with SQLiteLedger(path, spec) as ledger:
        assert len(ledger.snapshot()["signals"]) == 1
        assert len(ledger.snapshot()["orders"]) == 1


def test_execution_source_may_change_fill_prices_without_changing_decision_code(tmp_path):
    class SlippedExecution(SimulatedCloseExecution):
        version = "test-slippage-v1"

        def fill(self, order, session, config):
            return [
                replace(f, price=f.price + (1 if f.shares > 0 else -1))
                for f in super().fill(order, session, config)
            ]

    frame = signals([-3, -3, 0, 0, 0])
    spec = replace(spec_for(frame, BacktestConfig()), execution_version=SlippedExecution.version)
    with SQLiteLedger(tmp_path / "slippage.db", spec) as ledger:
        for session in sessions_from_signals(frame):
            process_session(ledger, session, execution=SlippedExecution())
        state = ledger.load_state()
        assert state.position == 0
        assert state.realized_pnl == pytest.approx(-204)
        assert state.cash - spec.config.initial_capital == pytest.approx(-204)


def test_invalid_fill_provider_rolls_back_session(tmp_path):
    class EarlyExecution(SimulatedCloseExecution):
        def fill(self, order, session, config):
            return [
                replace(f, fill_session=order.signal_session)
                for f in super().fill(order, session, config)
            ]

    frame = signals([-3, -3, -3])
    with SQLiteLedger(tmp_path / "invalid-fill.db", spec_for(frame, BacktestConfig())) as ledger:
        items = list(sessions_from_signals(frame))
        process_session(ledger, items[0])
        before = ledger.snapshot()
        with pytest.raises(ValueError, match="Invalid execution fill"):
            process_session(ledger, items[1], execution=EarlyExecution())
        assert ledger.snapshot() == before


def test_pure_engine_rejects_history_with_future_row_before_current_close():
    items = list(sessions_from_signals(signals([-3, -3])))
    state = StrategyState.initial(BacktestConfig())
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_target([items[1], items[0]], state, BacktestConfig())
