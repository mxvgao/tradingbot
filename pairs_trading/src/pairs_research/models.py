"""Serializable values used by the pure strategy, execution and ledger layers."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any

import pandas as pd

from .config import BacktestConfig, ENGINE_VERSION, STRATEGY_VERSION


def session_date(value: Any) -> str:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError("Sessions must be valid timezone-free daily dates")
    return stamp.date().isoformat()


def finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def flag(value: Any) -> bool:
    return bool(pd.notna(value) and value == True)


def encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def event_id(*parts: str) -> str:
    return hashlib.sha256(encode(parts).encode()).hexdigest()


@dataclass(frozen=True)
class Observation:
    date: str
    price_a: float
    price_b: float
    zscore: float | None = None
    spread: float | None = None
    hedge_ratio: float | None = None
    intercept: float | None = None
    formation_spread_mean: float | None = None
    formation_spread_std: float | None = None
    ready: bool = True
    rolling_pass: bool = False
    regime_allowed: bool = False

    def as_signal_row(self) -> dict:
        return {
            "date": pd.Timestamp(self.date),
            "ticker_a_price": self.price_a,
            "ticker_b_price": self.price_b,
            **{
                k: float("nan") if getattr(self, k) is None else getattr(self, k)
                for k in (
                    "zscore",
                    "spread",
                    "hedge_ratio",
                    "intercept",
                    "formation_spread_mean",
                    "formation_spread_std",
                )
            },
        }


@dataclass(frozen=True)
class Session:
    date: str
    price_a: float
    price_b: float
    observation: Observation | None = None

    def __post_init__(self):
        object.__setattr__(self, "date", session_date(self.date))
        if any(not math.isfinite(p) or p <= 0 for p in (self.price_a, self.price_b)):
            raise ValueError("Both legs need finite positive prices on every supplied session")
        if self.observation is not None and (
            self.observation.date != self.date
            or self.observation.price_a != self.price_a
            or self.observation.price_b != self.price_b
        ):
            raise ValueError("Observation must belong to this session and its prices")

    @classmethod
    def from_signal_row(cls, row: dict) -> "Session":
        required = {"date", "ticker_a_price", "ticker_b_price", "zscore", "spread", "hedge_ratio"}
        if missing := required.difference(row):
            raise ValueError(f"Missing signal columns: {sorted(missing)}")
        day = session_date(row["date"])
        a, b = float(row["ticker_a_price"]), float(row["ticker_b_price"])
        observation = Observation(
            day,
            a,
            b,
            **{
                k: finite_or_none(row.get(k))
                for k in (
                    "zscore",
                    "spread",
                    "hedge_ratio",
                    "intercept",
                    "formation_spread_mean",
                    "formation_spread_std",
                )
            },
            rolling_pass=flag(row.get("rolling_pass", False)),
            regime_allowed=flag(row.get("regime_allowed", False)),
        )
        return cls(day, a, b, observation)


@dataclass(frozen=True)
class Decision:
    target_position: int
    reason: str
    observation: Observation
    blocked_direction: int | None = None


@dataclass(frozen=True)
class Order:
    order_id: str
    signal_session: str
    order_purpose: str
    eligible_session_index: int
    shares_a: float
    shares_b: float
    target_position: int
    reason: str
    observation: Observation


@dataclass(frozen=True)
class FillEvent:
    fill_id: str
    order_id: str
    fill_session: str
    leg: str
    shares: float
    price: float
    fee: float


@dataclass(frozen=True)
class RunSpec:
    strategy_id: str
    config: BacktestConfig
    input_kind: str = "prices"
    liquidation_session: str | None = None
    engine_version: str = ENGINE_VERSION
    strategy_version: str = STRATEGY_VERSION
    execution_version: str = "simulated-close-v1"

    def __post_init__(self):
        if not self.strategy_id.strip():
            raise ValueError("A nonempty strategy/run ID is required")
        if self.input_kind not in {"prices", "signals"}:
            raise ValueError("input_kind must be prices or signals")
        if self.config.ticker_a.upper() == self.config.ticker_b.upper():
            raise ValueError("A pair needs two distinct tickers")
        if self.liquidation_session is not None:
            object.__setattr__(self, "liquidation_session", session_date(self.liquidation_session))

    @property
    def pair_id(self) -> str:
        return f"{self.config.ticker_a.upper()}/{self.config.ticker_b.upper()}"

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(encode(asdict(self)).encode()).hexdigest()


@dataclass
class StrategyState:
    cash: float
    equity: float
    realized_pnl: float = 0.0
    position: int = 0
    open_trade: dict | None = None
    pending: Order | None = None
    blocked_direction: int | None = None
    last_processed_session: str | None = None
    session_index: int = -1
    last_exit_session: str | None = None
    history: list[Session] = field(default_factory=list)

    @classmethod
    def initial(cls, config: BacktestConfig) -> "StrategyState":
        return cls(cash=config.initial_capital, equity=config.initial_capital)

    @classmethod
    def from_json(cls, value: str) -> "StrategyState":
        data = json.loads(value)
        for row in data["history"]:
            if row["observation"] is not None:
                row["observation"] = Observation(**row["observation"])
        data["history"] = [Session(**row) for row in data["history"]]
        if data["pending"] is not None:
            data["pending"]["observation"] = Observation(**data["pending"]["observation"])
            data["pending"] = Order(**data["pending"])
        return cls(**data)

    def to_json(self) -> str:
        return encode(asdict(self))


@dataclass
class SessionResult:
    state: StrategyState
    decision: Decision
    daily: dict
    orders: list[tuple[Order, str]] = field(default_factory=list)
    fills: list[FillEvent] = field(default_factory=list)
    trade: dict | None = None
