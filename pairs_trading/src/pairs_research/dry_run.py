"""Daily reconciliation and transactional order intents; never submits orders."""

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import math
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from .alpaca_readonly import AlpacaReadOnly, AlpacaSnapshot
from .config import BacktestConfig
from .engine import process_session
from .ledger import SQLiteLedger
from .models import RunSpec, Session, encode, event_id

NY = ZoneInfo("America/New_York")
DRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS dry_run_instances (
 strategy_id TEXT PRIMARY KEY REFERENCES runs(strategy_id), account_id TEXT NOT NULL,
 policy_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
 proposal_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL REFERENCES runs(strategy_id),
 pair_id TEXT NOT NULL, signal_session TEXT NOT NULL, order_id TEXT NOT NULL REFERENCES orders(order_id),
 leg TEXT NOT NULL CHECK(leg IN ('a','b')), execution_session TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status='dry_run_only'), payload_json TEXT NOT NULL,
 UNIQUE(strategy_id,pair_id,signal_session,leg)
);
CREATE TABLE IF NOT EXISTS dry_run_reports (
 report_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL REFERENCES runs(strategy_id),
 data_session TEXT, observed_at TEXT NOT NULL, status TEXT NOT NULL, report_json TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class DryRunPolicy:
    feed: str = "iex"
    close_delay_minutes: int = 20
    max_snapshot_age_seconds: int = 120
    max_gross_exposure: float = 10_000
    mode: str = "paper_read_only"
    version: str = "alpaca-dry-run-v1"

    def __post_init__(self):
        if (
            self.feed not in {"iex", "sip"}
            or self.mode not in {"paper_read_only", "offline_fixture"}
            or self.close_delay_minutes < 0
            or self.max_snapshot_age_seconds < 1
            or not math.isfinite(self.max_gross_exposure)
            or self.max_gross_exposure <= 0
        ):
            raise ValueError("Invalid dry-run risk/data policy")


def aware(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Data timestamps must be timezone aware")
    return stamp


def calendar_sessions(snapshot: AlpacaSnapshot, now: datetime, policy: DryRunPolicy):
    observed = aware(snapshot.observed_at)
    if (
        now.tzinfo is None
        or abs((now - observed).total_seconds()) > policy.max_snapshot_age_seconds
    ):
        raise ValueError("Stale broker snapshot")
    if (
        snapshot.feed != policy.feed
        or snapshot.adjustment != "raw"
        or snapshot.source != policy.mode
    ):
        raise ValueError("Snapshot feed, price basis or source differs from the pinned policy")
    sessions = {}
    for row in snapshot.calendar:
        day = row["date"]
        close = datetime.fromisoformat(day + "T" + row["close"]).replace(tzinfo=NY)
        if day in sessions:
            raise ValueError("Duplicate market-calendar session")
        sessions[day] = close
    completed = sorted(
        day
        for day, close in sessions.items()
        if close + timedelta(minutes=policy.close_delay_minutes) <= observed
    )
    if not completed:
        raise ValueError("No completed trading session in the calendar")
    latest = completed[-1]
    following = sorted(day for day in sessions if day > latest)
    if not following or sessions[following[0]] <= observed:
        raise ValueError("Market calendar is stale or the next close is unavailable")
    return completed, following[0], sessions


def validated_bars(snapshot, days, config):
    bars = {}
    for symbol in (config.ticker_a.upper(), config.ticker_b.upper()):
        values = {}
        for bar in snapshot.bars.get(symbol, []):
            stamp = aware(bar["t"])
            day = stamp.astimezone(NY).date().isoformat()
            if day not in days:
                continue  # In-progress/future bars cannot enter the engine.
            if day in values:
                raise ValueError(f"Duplicate {symbol} bar on {day}")
            price = float(bar["c"])
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"Invalid {symbol} price on {day}")
            values[day] = price
        if missing := set(days).difference(values):
            raise ValueError(f"Stale or missing {symbol} prices: {min(missing)}")
        bars[symbol] = values
    a, b = config.ticker_a.upper(), config.ticker_b.upper()
    return [Session(day, bars[a][day], bars[b][day]) for day in days]


def broker_positions(snapshot, symbols):
    result = {symbol: 0.0 for symbol in symbols}
    seen = set()
    for row in snapshot.positions:
        symbol, qty = row["symbol"], float(row["qty"])
        side = row["side"]
        if (
            symbol in seen
            or not math.isfinite(qty)
            or side not in {"long", "short"}
            or (side == "long" and qty < 0)
        ):
            raise ValueError("Malformed or duplicated broker position")
        seen.add(symbol)
        qty = abs(qty) * (-1 if side == "short" else 1)
        result[symbol] = qty
    return result


def save_report(ledger, report):
    payload = encode(report)
    identifier = event_id(ledger.spec.strategy_id, payload)
    ledger.connection.execute(
        "INSERT INTO dry_run_reports VALUES (?,?,?,?,?,?) ON CONFLICT(report_id) DO NOTHING",
        (
            identifier,
            ledger.spec.strategy_id,
            report.get("data_session"),
            report["observed_at"],
            report["status"],
            payload,
        ),
    )


class ReconciliationBlocked(ValueError):
    pass


def run_dry_run(
    ledger: SQLiteLedger, snapshot: AlpacaSnapshot, *, now: datetime, policy: DryRunPolicy
) -> dict:
    """Use a pre-fetched snapshot only: this function performs no network I/O.

    An outer transaction includes all new engine sessions, reconciliation,
    immutable outbox proposals and the success report. Failures roll it back.
    """
    if ledger.connection.in_transaction:
        raise ValueError("Dry-run owns its outer transaction")
    ledger.connection.executescript(DRY_SCHEMA)
    cfg, spec = ledger.spec.config, ledger.spec
    if spec.input_kind != "prices" or spec.liquidation_session is not None:
        raise ValueError("Dry-run requires an unbounded raw-price strategy instance")
    if cfg.require_regime_allowed or cfg.require_rolling_pass:
        raise ValueError("Daily adapter currently requires external regime/rolling gates disabled")
    symbols = [cfg.ticker_a.upper(), cfg.ticker_b.upper()]
    report = {
        "strategy_id": spec.strategy_id,
        "pair_id": spec.pair_id,
        "mode": policy.mode,
        "observed_at": snapshot.observed_at,
        "data_session": None,
        "data_timestamp": None,
        "status": "blocked",
        "signal": None,
        "reason": None,
        "broker_positions": {},
        "desired_positions": {},
        "expected_positions": {},
        "required_orders": [],
        "proposed_orders": [],
        "gross_exposure": None,
        "broker_gross_exposure": None,
        "failures": [],
        "submission_enabled": False,
        "qualifies_for_paper_gate": False,
    }
    try:
        completed, execution_day, calendar = calendar_sessions(snapshot, now, policy)
        report["data_session"] = completed[-1]
        report["completed_close_at"] = calendar[completed[-1]].isoformat()
        report["data_timestamp"] = {
            symbol: max(
                (
                    bar["t"]
                    for bar in snapshot.bars.get(symbol, [])
                    if aware(bar["t"]).astimezone(NY).date().isoformat() <= completed[-1]
                ),
                default=None,
            )
            for symbol in symbols
        }
        current = broker_positions(snapshot, symbols)
        report["broker_positions"] = current
        if any(symbol not in symbols and qty for symbol, qty in current.items()):
            raise ReconciliationBlocked("Unexpected broker position outside the configured pair")
        if snapshot.open_orders:
            raise ReconciliationBlocked("Unresolved broker orders block all new proposals")
        account = snapshot.account
        if (
            not account.get("id")
            or account.get("status") != "ACTIVE"
            or account.get("trading_blocked") is not False
            or account.get("account_blocked") is not False
        ):
            raise ReconciliationBlocked(
                "Paper account is inactive, blocked or missing required status fields"
            )
        with ledger.transaction():
            pinned = ledger.connection.execute(
                "SELECT account_id,policy_json FROM dry_run_instances WHERE strategy_id=?",
                (spec.strategy_id,),
            ).fetchone()
            policy_json = encode(asdict(policy))
            if pinned is not None and (pinned[0] != account["id"] or pinned[1] != policy_json):
                raise ReconciliationBlocked(
                    "Paper account or dry-run policy changed; use a new run ID"
                )
            state = ledger.load_state()
            if pinned is None:
                if state.last_processed_session is not None:
                    raise ReconciliationBlocked(
                        "Existing replay state has no pinned broker price policy; use a new run ID"
                    )
                ledger.connection.execute(
                    "INSERT INTO dry_run_instances VALUES (?,?,?)",
                    (spec.strategy_id, account["id"], policy_json),
                )
            if state.last_processed_session is None:
                required = completed[-(cfg.formation_days + 1) :]
                if len(required) < cfg.formation_days + 1:
                    raise ReconciliationBlocked("Insufficient completed formation history")
            else:
                if state.last_processed_session not in completed:
                    raise ReconciliationBlocked(
                        "Checkpoint is outside the fetched completed calendar"
                    )
                required = [day for day in completed if day >= state.last_processed_session]
                if len(required) > 2:
                    raise ReconciliationBlocked(
                        "Missed sessions require explicit reconciliation; cannot infer intervening broker fills"
                    )
            sessions = validated_bars(snapshot, required, cfg)
            for session in sessions:
                previous = ledger.connection.execute(
                    "SELECT price_a,price_b FROM positions WHERE strategy_id=? AND pair_id=? AND session=?",
                    (spec.strategy_id, spec.pair_id, session.date),
                ).fetchone()
                if previous is not None and (
                    previous[0] != session.price_a or previous[1] != session.price_b
                ):
                    raise ReconciliationBlocked(
                        "Committed prices were revised; use a new run or investigate the data change"
                    )
                process_session(ledger, session)
            state = ledger.load_state()
            signal_row = ledger.connection.execute(
                "SELECT target_position,reason FROM signals WHERE strategy_id=? AND pair_id=? AND signal_session=?",
                (spec.strategy_id, spec.pair_id, completed[-1]),
            ).fetchone()
            report["signal"], report["reason"] = signal_row[0], signal_row[1]
            expected = {
                symbols[0]: state.open_trade["shares_a"] if state.open_trade else 0.0,
                symbols[1]: state.open_trade["shares_b"] if state.open_trade else 0.0,
            }
            report["expected_positions"] = expected
            pending = state.pending
            desired = expected.copy()
            if pending is not None:
                desired[symbols[0]] += pending.shares_a
                desired[symbols[1]] += pending.shares_b
            report["desired_positions"] = desired
            prices = {symbols[0]: sessions[-1].price_a, symbols[1]: sessions[-1].price_b}
            report["gross_exposure"] = sum(abs(desired[s] * prices[s]) for s in symbols)
            report["broker_gross_exposure"] = sum(
                abs(current.get(s, 0) * prices[s]) for s in symbols
            )
            if any(
                not math.isclose(current[s], expected[s], rel_tol=0, abs_tol=1e-8) for s in symbols
            ):
                raise ReconciliationBlocked(
                    "Unexpected broker quantities: paper holdings differ from the engine's current holdings"
                )
            proposals = []
            if pending is not None:
                for leg, symbol, qty in (
                    ("a", symbols[0], pending.shares_a),
                    ("b", symbols[1], pending.shares_b),
                ):
                    if not qty:
                        continue
                    proposals.append(
                        {
                            "proposal_id": event_id(pending.order_id, leg, "dry-run"),
                            "order_id": pending.order_id,
                            "leg": leg,
                            "symbol": symbol,
                            "side": "buy" if qty > 0 else "sell",
                            "qty": str(Decimal(str(abs(qty)))),
                            "signed_qty": qty,
                            "signal_session": pending.signal_session,
                            "execution_session": execution_day,
                            "execution_at": calendar[execution_day].isoformat(),
                            "submit_not_before": datetime.fromisoformat(
                                pending.signal_session + "T19:00:00"
                            )
                            .replace(tzinfo=NY)
                            .isoformat(),
                            "submit_before": (
                                calendar[execution_day] - timedelta(minutes=10)
                            ).isoformat(),
                            "order_type": "market_on_close",
                            "time_in_force": "cls",
                            "status": "dry_run_only",
                        }
                    )
            report["required_orders"] = proposals
            if proposals and now >= calendar[execution_day] - timedelta(minutes=10):
                raise ReconciliationBlocked(
                    "Next-session closing-auction submission cutoff has passed"
                )
            if report["gross_exposure"] > policy.max_gross_exposure + 1e-7:
                raise ReconciliationBlocked("Desired gross exposure exceeds the configured limit")
            if any(desired[s] < 0 for s in symbols) and account.get("shorting_enabled") is not True:
                raise ReconciliationBlocked("Paper account shorting is unavailable")
            buying_power = float(account.get("buying_power", "nan"))
            if not math.isfinite(buying_power) or buying_power < 0:
                raise ReconciliationBlocked("Invalid paper-account buying power")
            increasing = sum(max(0, abs(desired[s]) - abs(current[s])) * prices[s] for s in symbols)
            if increasing * 1.03 > buying_power:
                raise ReconciliationBlocked(
                    "Insufficient paper buying power with a 3% sizing buffer"
                )
            # Alpaca CLS does not support fractional equity quantities. Preserve
            # the engine's exact requirements instead of silently rounding them.
            if any(Decimal(p["qty"]) != Decimal(p["qty"]).to_integral_value() for p in proposals):
                raise ReconciliationBlocked(
                    "Fractional engine quantities cannot be submitted as Alpaca CLS orders; an aligned whole-share sizing policy is required"
                )
            for proposal in proposals:
                payload = encode(proposal)
                old = ledger.connection.execute(
                    "SELECT payload_json FROM outbox WHERE proposal_id=?",
                    (proposal["proposal_id"],),
                ).fetchone()
                if old is not None and old[0] != payload:
                    raise ReconciliationBlocked(
                        "Existing immutable outbox intent differs; manual investigation required"
                    )
                if old is None:
                    ledger.connection.execute(
                        "INSERT INTO outbox VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            proposal["proposal_id"],
                            spec.strategy_id,
                            spec.pair_id,
                            proposal["signal_session"],
                            proposal["order_id"],
                            proposal["leg"],
                            execution_day,
                            "dry_run_only",
                            payload,
                        ),
                    )
            report["proposed_orders"] = proposals
            report["status"] = "proposed" if proposals else "hold"
            report["qualifies_for_paper_gate"] = policy.mode == "paper_read_only"
            save_report(ledger, report)
    except (ValueError, KeyError, TypeError) as exc:
        report["status"] = "blocked"
        report["proposed_orders"] = []
        report["failures"] = [str(exc)]
        with ledger.transaction():
            save_report(ledger, report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Read-only Alpaca daily reconciliation; records proposals but cannot submit orders."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feed", choices=["iex", "sip"], default="iex")
    parser.add_argument("--max-gross-exposure", type=float, default=10000)
    parser.add_argument(
        "--snapshot", type=Path, help="Offline fixture snapshot; makes no network requests."
    )
    parser.add_argument(
        "--report", type=Path, help="Optional copy of the reconciliation report JSON."
    )
    args = parser.parse_args()
    try:
        cfg = BacktestConfig(**json.loads(args.config.read_text()))
        if args.snapshot:
            snapshot = AlpacaSnapshot(**json.loads(args.snapshot.read_text()))
            snapshot = replace(snapshot, source="offline_fixture")
            now = aware(snapshot.observed_at)
        else:
            client = AlpacaReadOnly.from_environment()
            try:
                snapshot = client.snapshot(cfg, args.feed)  # All networking before any transaction.
            finally:
                client.close()
            now = datetime.now(timezone.utc)
        policy = DryRunPolicy(
            feed=args.feed, max_gross_exposure=args.max_gross_exposure, mode=snapshot.source
        )
        with SQLiteLedger(args.database, RunSpec(args.run_id, cfg)) as ledger:
            report = run_dry_run(ledger, snapshot, now=now, policy=policy)
        payload = json.dumps(report, indent=2, sort_keys=True)
        if args.report:
            args.report.write_text(payload + "\n")
        print(payload)
        if report["status"] == "blocked":
            raise SystemExit(2)
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"Dry-run failed; no orders submitted: {exc}\n")


if __name__ == "__main__":
    main()
