from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import subprocess
import sys

import pytest

from pairs_research.alpaca_readonly import AlpacaReadOnly, AlpacaSnapshot, ENDPOINTS
from pairs_research.config import BacktestConfig
from pairs_research.dry_run import DRY_SCHEMA, DryRunPolicy, run_dry_run, aware
from pairs_research.engine import process_session
from pairs_research.ledger import MemoryLedger, SQLiteLedger
from pairs_research.models import Observation, RunSpec, Session
from pairs_research.replay import replay


@pytest.fixture
def snapshot():
    days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
    return AlpacaSnapshot(
        "2026-01-08T21:30:00Z",
        [{"date": d, "open": "09:30", "close": "16:00"} for d in days],
        {
            s: [{"t": d + "T05:00:00Z", "c": 100.0, "v": 100000} for d in days[:4]]
            for s in ("A", "B")
        },
        [],
        [],
        {
            "id": "paper-test",
            "status": "ACTIVE",
            "trading_blocked": False,
            "account_blocked": False,
            "buying_power": "100000",
            "shorting_enabled": True,
        },
        source="offline_fixture",
    )


@pytest.fixture
def config():
    return BacktestConfig(ticker_a="A", ticker_b="B", formation_days=3)


@pytest.fixture
def exact_signal(monkeypatch):
    from pairs_research import strategy

    # Control only the statistical input to test exact planning/reconciliation;
    # the real target, order planner, fill accounting and transactions all run.
    def observation(history, cfg):
        row = history[-1]
        ready = len(history) > cfg.formation_days
        return Observation(
            row.date,
            row.price_a,
            row.price_b,
            -3.0 if ready else None,
            0.02 if ready else None,
            1.0 if ready else None,
            ready=ready,
        )

    monkeypatch.setattr(strategy, "observe_close", observation)


def policy(**kwargs):
    return DryRunPolicy(mode="offline_fixture", **kwargs)


def apply(ledger, snapshot, **kwargs):
    return run_dry_run(ledger, snapshot, now=aware(snapshot.observed_at), policy=policy(**kwargs))


def test_proposals_equal_shared_engine_quantities_and_are_idempotent(
    tmp_path, snapshot, config, exact_signal
):
    spec = RunSpec("daily", config)
    memory = MemoryLedger(spec)
    for row in snapshot.bars["A"]:
        process_session(memory, Session(row["t"][:10], 100, 100))
    required = memory.load_state().pending
    path = tmp_path / "dry.db"
    with SQLiteLedger(path, spec) as ledger:
        report = apply(ledger, snapshot)
        assert report["status"] == "proposed"
        assert [p["signed_qty"] for p in report["proposed_orders"]] == [
            required.shares_a,
            required.shares_b,
        ]
        assert report["desired_positions"] == {"A": 50, "B": -50}
        assert report["gross_exposure"] == 10000
        assert report["signal"] == 1 and report["reason"] == "entry"
        assert all(p["execution_session"] > p["signal_session"] for p in report["proposed_orders"])
        assert not report["submission_enabled"] and not report["qualifies_for_paper_gate"]
        before = ledger.snapshot()
        second = apply(ledger, snapshot)
        assert second["proposed_orders"] == report["proposed_orders"]
        assert ledger.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 2
        assert ledger.snapshot() == before
    with SQLiteLedger(path, spec) as ledger:
        assert apply(ledger, snapshot)["proposed_orders"] == report["proposed_orders"]
        assert ledger.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 2


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "stale",
        "nan",
        "duplicate",
        "unexpected",
        "unresolved",
        "blocked",
        "buying_power",
        "gross",
        "shorting",
    ],
)
def test_risk_or_data_failure_has_no_outbox_or_checkpoint(
    tmp_path, snapshot, config, exact_signal, bad
):
    snap = deepcopy(snapshot)
    kwargs = {}
    if bad == "missing":
        snap.bars["B"].pop()
    elif bad == "stale":
        snap.bars["A"] = snap.bars["A"][:-1]
        snap.bars["B"] = snap.bars["B"][:-1]
    elif bad == "nan":
        snap.bars["A"][-1]["c"] = float("nan")
    elif bad == "duplicate":
        snap.bars["A"].append(snap.bars["A"][-1])
    elif bad == "unexpected":
        snap.positions.append({"symbol": "SPY", "qty": "5", "side": "long"})
    elif bad == "unresolved":
        snap.open_orders.append(
            {"symbol": "SPY", "id": "pending-other-symbol", "status": "pending_cancel"}
        )
    elif bad == "blocked":
        snap.account["trading_blocked"] = True
    elif bad == "buying_power":
        snap.account["buying_power"] = "1"
    elif bad == "gross":
        kwargs["max_gross_exposure"] = 5000
    elif bad == "shorting":
        snap.account["shorting_enabled"] = False
    with SQLiteLedger(tmp_path / "blocked.db", RunSpec("daily", config)) as ledger:
        report = apply(ledger, snap, **kwargs)
        assert report["status"] == "blocked"
        assert report["failures"]
        assert report["proposed_orders"] == []
        assert ledger.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
        assert ledger.load_state().last_processed_session is None
        assert ledger.snapshot()["signals"] == []


def test_fractional_cls_requirement_is_reported_without_rounding(
    tmp_path, snapshot, config, exact_signal
):
    cfg = replace(config, gross_notional_per_trade=10001)
    with SQLiteLedger(tmp_path / "fractional.db", RunSpec("daily", cfg)) as ledger:
        report = apply(ledger, snapshot, max_gross_exposure=11000)
        assert report["status"] == "blocked"
        assert "Fractional" in report["failures"][0]
        assert report["required_orders"][0]["qty"] == "50.005"
        assert report["proposed_orders"] == []
        assert ledger.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0


@pytest.mark.parametrize("table", ["outbox", "dry_run_reports"])
def test_database_fault_rolls_back_proposals_and_engine(
    tmp_path, snapshot, config, exact_signal, table
):
    with SQLiteLedger(tmp_path / "fault.db", RunSpec("daily", config)) as ledger:
        ledger.connection.executescript(DRY_SCHEMA)
        condition = " WHEN NEW.leg='b'" if table == "outbox" else ""
        ledger.connection.execute(
            f"CREATE TEMP TRIGGER fail_write BEFORE INSERT ON {table}{condition} BEGIN SELECT RAISE(ABORT, 'injected database failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            apply(ledger, snapshot)
        assert ledger.snapshot()["signals"] == []
        assert ledger.load_state().last_processed_session is None
        assert ledger.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
        ledger.connection.execute("DROP TRIGGER fail_write")
        assert apply(ledger, snapshot)["status"] == "proposed"


def test_paper_drift_on_following_day_blocks_without_advancing(
    tmp_path, snapshot, config, exact_signal
):
    with SQLiteLedger(tmp_path / "drift.db", RunSpec("daily", config)) as ledger:
        assert apply(ledger, snapshot)["status"] == "proposed"
        next_day = replace(deepcopy(snapshot), observed_at="2026-01-09T21:30:00Z")
        for bars in next_day.bars.values():
            bars.append({"t": "2026-01-09T05:00:00Z", "c": 100.0})
        before = ledger.snapshot()
        report = apply(ledger, next_day)
        assert report["status"] == "blocked"
        assert "Unexpected broker quantities" in report["failures"][0]
        assert ledger.snapshot() == before
        # A verified matching paper snapshot permits the next session to commit.
        next_day.positions.extend(
            [
                {"symbol": "A", "side": "long", "qty": "50"},
                {"symbol": "B", "side": "short", "qty": "-50"},
            ]
        )
        assert apply(ledger, next_day)["status"] == "hold"
        assert ledger.load_state().last_processed_session == "2026-01-09"


def test_revised_committed_bar_is_not_silently_reused(tmp_path, snapshot, config, exact_signal):
    with SQLiteLedger(tmp_path / "revision.db", RunSpec("daily", config)) as ledger:
        apply(ledger, snapshot)
        snapshot.bars["A"][-1]["c"] = 105
        assert "revised" in apply(ledger, snapshot)["failures"][0]
        assert ledger.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 2


def test_incomplete_current_bar_ignored_and_holiday_weekend_not_stale(
    tmp_path, snapshot, config, exact_signal
):
    # Saturday uses Friday's completed close, next execution is Monday.
    snap = replace(deepcopy(snapshot), observed_at="2026-01-10T15:00:00Z")
    for bars in snap.bars.values():
        bars.append({"t": "2026-01-09T05:00:00Z", "c": 100.0})
        bars.append({"t": "2026-01-12T05:00:00Z", "c": 999999.0})
    with SQLiteLedger(tmp_path / "weekend.db", RunSpec("daily", config)) as ledger:
        report = apply(ledger, snap)
        assert report["data_session"] == "2026-01-09"
        assert report["status"] == "proposed"
        assert report["proposed_orders"][0]["execution_session"] == "2026-01-12"
        assert report["desired_positions"]["A"] == 50


def test_stale_broker_snapshot_fails_closed(tmp_path, snapshot, config):
    with SQLiteLedger(tmp_path / "stale.db", RunSpec("daily", config)) as ledger:
        report = run_dry_run(
            ledger,
            snapshot,
            now=aware(snapshot.observed_at) + timedelta(minutes=5),
            policy=policy(),
        )
        assert "Stale broker" in report["failures"][0]
        assert not report["proposed_orders"]


def test_early_close_calendar_respected(tmp_path, snapshot, config, exact_signal):
    snapshot.calendar[3]["close"] = "13:00"
    early = replace(snapshot, observed_at="2026-01-08T18:30:00Z")
    with SQLiteLedger(tmp_path / "early.db", RunSpec("daily", config)) as ledger:
        report = apply(ledger, early)
        assert report["status"] == "proposed"
        assert report["completed_close_at"].endswith("13:00:00-05:00")


def test_http_client_only_gets_fixed_paper_and_data_hosts(monkeypatch, snapshot, config):
    calls = []

    class Response:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

    def get(_self, url, **kwargs):
        calls.append((url, kwargs))
        assert url in ENDPOINTS.values()
        assert kwargs["allow_redirects"] is False
        if url == ENDPOINTS["clock"]:
            return Response({"timestamp": snapshot.observed_at})
        if url == ENDPOINTS["calendar"]:
            return Response(snapshot.calendar)
        if url == ENDPOINTS["positions"]:
            return Response(snapshot.positions)
        if url == ENDPOINTS["orders"]:
            assert kwargs["params"]["status"] == "open"
            assert "symbols" not in kwargs["params"]
            return Response([])
        if url == ENDPOINTS["account"]:
            return Response(snapshot.account)
        if kwargs["params"].get("page_token") == "page2":
            return Response({"bars": {"B": snapshot.bars["B"]}, "next_page_token": None})
        return Response({"bars": {"A": snapshot.bars["A"]}, "next_page_token": "page2"})

    monkeypatch.setattr("requests.Session.get", get)

    def forbidden(*a, **kw):
        raise AssertionError("External mutation attempted")

    for method in ("post", "put", "patch", "delete"):
        monkeypatch.setattr(f"requests.Session.{method}", forbidden)
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
    client = AlpacaReadOnly("fixture-key", "fixture-secret")
    try:
        actual = client.snapshot(config)
        assert actual.bars == snapshot.bars
        assert all(not url.startswith("https://api.alpaca.markets") for url, _ in calls)
        with pytest.raises(ValueError, match="not allowed"):
            client._get("https://api.alpaca.markets/v2/orders")
    finally:
        client.close()


def test_redirect_is_rejected_without_following(monkeypatch):
    class Redirect:
        status_code = 302

    calls = []

    def get(*a, **kw):
        calls.append(kw)
        return Redirect()

    monkeypatch.setattr("requests.Session.get", get)
    client = AlpacaReadOnly("fixture-key", "fixture-secret")
    try:
        with pytest.raises(ValueError, match="HTTP 302"):
            client._get("positions")
        assert len(calls) == 1 and calls[0]["allow_redirects"] is False
    finally:
        client.close()


def test_real_raw_bars_cli_offline_never_needs_credentials(tmp_path, snapshot):
    path, cfg = tmp_path / "snapshot.json", tmp_path / "config.json"
    path.write_text(json.dumps(snapshot.to_dict()))
    cfg.write_text(json.dumps({"ticker_a": "A", "ticker_b": "B", "formation_days": 3}))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pairs_research.dry_run",
            "--database",
            str(tmp_path / "cli.db"),
            "--run-id",
            "cli",
            "--config",
            str(cfg),
            "--snapshot",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "hold"
    assert not report["submission_enabled"]
    assert not report["qualifies_for_paper_gate"]


def test_auction_window_and_cutoff(tmp_path, snapshot, config, exact_signal):
    with SQLiteLedger(tmp_path / "cutoff.db", RunSpec("daily", config)) as ledger:
        report = apply(ledger, snapshot)
        proposal = report["proposed_orders"][0]
        assert proposal["submit_not_before"] == "2026-01-08T19:00:00-05:00"
        assert proposal["submit_before"] == "2026-01-09T15:50:00-05:00"
        before = ledger.snapshot()
        late = replace(snapshot, observed_at="2026-01-09T20:50:00Z")
        report = apply(ledger, late)
        assert report["status"] == "blocked"
        assert "cutoff" in report["failures"][0]
        assert ledger.snapshot() == before
        assert ledger.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 2


@pytest.mark.parametrize("changed", ["account", "policy", "pair_quantity"])
def test_reconciliation_identity_is_pinned(tmp_path, snapshot, config, exact_signal, changed):
    with SQLiteLedger(tmp_path / "identity.db", RunSpec("daily", config)) as ledger:
        apply(ledger, snapshot)
        before = ledger.snapshot()
        kwargs = {}
        if changed == "account":
            snapshot.account["id"] = "another-paper-account"
        elif changed == "policy":
            kwargs["max_gross_exposure"] = 20000
        else:
            snapshot.positions.append({"symbol": "A", "qty": "1", "side": "long"})
        report = apply(ledger, snapshot, **kwargs)
        assert report["status"] == "blocked"
        assert not report["proposed_orders"]
        assert ledger.snapshot() == before
        assert ledger.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 2


def test_missed_sessions_do_not_invent_round_trip_broker_fills(
    tmp_path, snapshot, config, exact_signal
):
    with SQLiteLedger(tmp_path / "gap.db", RunSpec("daily", config)) as ledger:
        apply(ledger, snapshot)
        before = ledger.snapshot()
        later = replace(snapshot, observed_at="2026-01-12T21:30:00Z")
        snapshot.calendar.append({"date": "2026-01-13", "open": "09:30", "close": "16:00"})
        for bars in snapshot.bars.values():
            bars.extend([{"t": d + "T05:00:00Z", "c": 100} for d in ("2026-01-09", "2026-01-12")])
        report = apply(ledger, later)
        assert "Missed sessions" in report["failures"][0]
        assert ledger.snapshot() == before


def test_short_entry_and_exit_proposals_use_engine_quantities(
    tmp_path, snapshot, config, monkeypatch
):
    from pairs_research import strategy

    def observation(history, cfg):
        row = history[-1]
        ready = len(history) > cfg.formation_days
        z = 3.0 if row.date <= "2026-01-08" else 0.0
        return Observation(
            row.date, row.price_a, row.price_b, z if ready else None, 0.02, 1.0, ready=ready
        )

    monkeypatch.setattr(strategy, "observe_close", observation)
    with SQLiteLedger(tmp_path / "short.db", RunSpec("daily", config)) as ledger:
        report = apply(ledger, snapshot)
        assert report["signal"] == -1
        assert [p["signed_qty"] for p in report["proposed_orders"]] == [-50, 50]
        next_day = replace(snapshot, observed_at="2026-01-09T21:30:00Z")
        for bars in snapshot.bars.values():
            bars.append({"t": "2026-01-09T05:00:00Z", "c": 100})
        snapshot.positions.extend(
            [
                {"symbol": "A", "qty": "-50", "side": "short"},
                {"symbol": "B", "qty": "50", "side": "long"},
            ]
        )
        report = apply(ledger, next_day)
        assert report["signal"] == 0 and report["reason"] == "mean_reversion"
        assert report["desired_positions"] == {"A": 0, "B": 0}
        assert [p["signed_qty"] for p in report["proposed_orders"]] == [50, -50]
        assert report["broker_gross_exposure"] == 10000
        assert report["gross_exposure"] == 0
        assert len(ledger.snapshot()["fills"]) == 2
        assert ledger.connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 4


def test_replay_ledger_cannot_be_reused_as_broker_state(tmp_path, snapshot, config):
    with SQLiteLedger(tmp_path / "replay.db", RunSpec("daily", config)) as ledger:
        process_session(ledger, Session("2026-01-08", 100, 100))
        before = ledger.snapshot()
        report = apply(ledger, snapshot)
        assert "no pinned broker price policy" in report["failures"][0]
        assert ledger.snapshot() == before
