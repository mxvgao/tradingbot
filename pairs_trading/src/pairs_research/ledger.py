"""Interchangeable memory and SQLite ledgers for the shared session workflow."""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import sqlite3

from .models import RunSpec, SessionResult, StrategyState, encode

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    strategy_id TEXT PRIMARY KEY, pair_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL, spec_json TEXT NOT NULL,
    engine_version TEXT NOT NULL, strategy_version TEXT NOT NULL,
    execution_version TEXT NOT NULL, liquidation_session TEXT,
    UNIQUE(strategy_id, pair_id)
);
CREATE TABLE IF NOT EXISTS signals (
    strategy_id TEXT NOT NULL, pair_id TEXT NOT NULL, signal_session TEXT NOT NULL,
    target_position INTEGER NOT NULL CHECK(target_position IN (-1,0,1)),
    reason TEXT NOT NULL, observation_json TEXT NOT NULL,
    UNIQUE(strategy_id, pair_id, signal_session),
    FOREIGN KEY(strategy_id,pair_id) REFERENCES runs(strategy_id,pair_id)
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL, pair_id TEXT NOT NULL,
    signal_session TEXT NOT NULL, order_purpose TEXT NOT NULL CHECK(order_purpose IN ('entry','exit','boundary')),
    eligible_session_index INTEGER NOT NULL,
    shares_a REAL NOT NULL, shares_b REAL NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','filled','cancelled')),
    order_json TEXT NOT NULL,
    UNIQUE(strategy_id, pair_id, signal_session, order_purpose),
    FOREIGN KEY(strategy_id,pair_id,signal_session) REFERENCES signals(strategy_id,pair_id,signal_session)
);
CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(order_id),
    fill_session TEXT NOT NULL, leg TEXT NOT NULL CHECK(leg IN ('a','b')),
    shares REAL NOT NULL CHECK(shares != 0), price REAL NOT NULL CHECK(price > 0),
    fee REAL NOT NULL CHECK(fee >= 0)
);
CREATE TRIGGER IF NOT EXISTS enforce_fill_timing BEFORE INSERT ON fills BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM orders o JOIN runs r ON r.strategy_id=o.strategy_id
        WHERE o.order_id=NEW.order_id AND (
            (o.order_purpose != 'boundary' AND NEW.fill_session > o.signal_session)
            OR (o.order_purpose = 'boundary' AND NEW.fill_session = r.liquidation_session
                AND NEW.fill_session = o.signal_session)
        )
    ) THEN RAISE(ABORT, 'fill must follow signal, except predeclared boundary liquidation') END;
END;
CREATE TABLE IF NOT EXISTS positions (
    strategy_id TEXT NOT NULL, pair_id TEXT NOT NULL, session TEXT NOT NULL,
    cash REAL NOT NULL, shares_a REAL NOT NULL, shares_b REAL NOT NULL,
    price_a REAL NOT NULL, price_b REAL NOT NULL, equity REAL NOT NULL,
    realized_pnl REAL NOT NULL, unrealized_pnl REAL NOT NULL, daily_json TEXT NOT NULL,
    UNIQUE(strategy_id,pair_id,session),
    FOREIGN KEY(strategy_id,pair_id,session) REFERENCES signals(strategy_id,pair_id,signal_session)
);
CREATE TABLE IF NOT EXISTS trades (
    strategy_id TEXT NOT NULL, pair_id TEXT NOT NULL, entry_session TEXT NOT NULL,
    exit_session TEXT NOT NULL, trade_json TEXT NOT NULL,
    UNIQUE(strategy_id,pair_id,entry_session),
    FOREIGN KEY(strategy_id,pair_id,exit_session) REFERENCES positions(strategy_id,pair_id,session)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    strategy_id TEXT NOT NULL, pair_id TEXT NOT NULL, last_processed_session TEXT NOT NULL,
    state_json TEXT NOT NULL, UNIQUE(strategy_id,pair_id),
    FOREIGN KEY(strategy_id,pair_id,last_processed_session) REFERENCES positions(strategy_id,pair_id,session)
);
"""


class MemoryLedger:
    def __init__(self, spec: RunSpec):
        self.spec = spec
        self.state = StrategyState.initial(spec.config)
        self.sessions = set()
        self.daily, self.trades = [], []

    @contextmanager
    def transaction(self):
        previous = self.state
        daily_count, trade_count = len(self.daily), len(self.trades)
        try:
            yield
        except BaseException:
            self.state = previous
            del self.daily[daily_count:]
            del self.trades[trade_count:]
            self.sessions = {row["date"] for row in self.daily}
            raise

    def has_session(self, session: str) -> bool:
        return session in self.sessions

    def load_state(self) -> StrategyState:
        return self.state

    def record(self, result: SessionResult) -> None:
        self.sessions.add(result.state.last_processed_session)
        self.daily.append(result.daily)
        if result.trade is not None:
            self.trades.append(result.trade)
        self.state = result.state


class SQLiteLedger:
    def __init__(self, path: str | Path, spec: RunSpec):
        self.spec = spec
        self.connection = sqlite3.connect(str(path), isolation_level=None, timeout=30)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError(f"Unsupported ledger schema version: {version}")
            self.connection.executescript(SCHEMA)
            self.connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            with self.transaction():
                previous = self.connection.execute(
                    "SELECT fingerprint FROM runs WHERE strategy_id=?", (spec.strategy_id,)
                ).fetchone()
                if previous is not None and previous[0] != spec.fingerprint:
                    raise ValueError(
                        "Run configuration/version/policy mismatch; choose a new run ID"
                    )
                if previous is None:
                    self.connection.execute(
                        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                        (
                            spec.strategy_id,
                            spec.pair_id,
                            spec.fingerprint,
                            encode(asdict(spec)),
                            spec.engine_version,
                            spec.strategy_version,
                            spec.execution_version,
                            spec.liquidation_session,
                        ),
                    )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def has_session(self, session: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM positions WHERE strategy_id=? AND pair_id=? AND session=?",
                (self.spec.strategy_id, self.spec.pair_id, session),
            ).fetchone()
            is not None
        )

    def load_state(self) -> StrategyState:
        row = self.connection.execute(
            "SELECT state_json FROM checkpoints WHERE strategy_id=? AND pair_id=?",
            (self.spec.strategy_id, self.spec.pair_id),
        ).fetchone()
        return StrategyState.from_json(row[0]) if row else StrategyState.initial(self.spec.config)

    def record(self, result: SessionResult) -> None:
        db, spec = self.connection, self.spec
        day = result.state.last_processed_session
        decision = result.decision
        db.execute(
            "INSERT INTO signals VALUES (?,?,?,?,?,?)",
            (
                spec.strategy_id,
                spec.pair_id,
                day,
                decision.target_position,
                decision.reason,
                encode(asdict(decision.observation)),
            ),
        )
        for order, status in result.orders:
            payload = encode(asdict(order))
            existing = db.execute(
                "SELECT order_json,status FROM orders WHERE order_id=?", (order.order_id,)
            ).fetchone()
            if existing is None:
                db.execute(
                    "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        order.order_id,
                        spec.strategy_id,
                        spec.pair_id,
                        order.signal_session,
                        order.order_purpose,
                        order.eligible_session_index,
                        order.shares_a,
                        order.shares_b,
                        status,
                        payload,
                    ),
                )
            else:
                if (
                    existing["order_json"] != payload
                    or existing["status"] != "pending"
                    or status == "pending"
                ):
                    raise ValueError("Order identity or lifecycle mismatch")
                db.execute("UPDATE orders SET status=? WHERE order_id=?", (status, order.order_id))
        for fill in result.fills:
            db.execute(
                "INSERT INTO fills VALUES (?,?,?,?,?,?,?)",
                (
                    fill.fill_id,
                    fill.order_id,
                    fill.fill_session,
                    fill.leg,
                    fill.shares,
                    fill.price,
                    fill.fee,
                ),
            )
        d = result.daily
        db.execute(
            "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                spec.strategy_id,
                spec.pair_id,
                day,
                d["cash"],
                d["shares_a"],
                d["shares_b"],
                d["price_a"],
                d["price_b"],
                d["equity"],
                d["realized_pnl"],
                d["unrealized_pnl"],
                encode(d),
            ),
        )
        if result.trade is not None:
            db.execute(
                "INSERT INTO trades VALUES (?,?,?,?,?)",
                (
                    spec.strategy_id,
                    spec.pair_id,
                    result.trade["entry_date"],
                    result.trade["exit_date"],
                    encode(result.trade),
                ),
            )
        # Advance only after all event, fill, position and completed-trade writes.
        db.execute(
            "INSERT INTO checkpoints VALUES (?,?,?,?) ON CONFLICT(strategy_id,pair_id) DO UPDATE SET last_processed_session=excluded.last_processed_session, state_json=excluded.state_json",
            (spec.strategy_id, spec.pair_id, day, result.state.to_json()),
        )

    def snapshot(self) -> dict:
        """Canonical economic/ledger contents, independent of SQLite file bytes."""
        output = {}
        for table, sort in (
            ("runs", "strategy_id"),
            ("signals", "signal_session"),
            ("orders", "signal_session,order_purpose"),
            ("fills", "fill_session,fill_id"),
            ("positions", "session"),
            ("trades", "entry_session"),
            ("checkpoints", "pair_id"),
        ):
            if table == "fills":
                rows = self.connection.execute(
                    "SELECT f.* FROM fills f JOIN orders o ON o.order_id=f.order_id WHERE o.strategy_id=? ORDER BY f.fill_session,f.fill_id",
                    (self.spec.strategy_id,),
                )
            else:
                rows = self.connection.execute(
                    f"SELECT * FROM {table} WHERE strategy_id=? ORDER BY {sort}",
                    (self.spec.strategy_id,),
                )
            output[table] = [dict(row) for row in rows]
        return output
