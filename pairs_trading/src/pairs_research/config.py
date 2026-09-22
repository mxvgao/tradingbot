"""Versioned configuration shared by batch and incremental research."""

from dataclasses import dataclass
import numpy as np

ENGINE_VERSION = "session-engine-v1"
STRATEGY_VERSION = "next-close-zscore-v1"


@dataclass(frozen=True)
class BacktestConfig:
    ticker_a: str = "SCHQ"
    ticker_b: str = "SPTL"
    formation_days: int = 504
    entry_z: float = 2.0
    exit_z: float = 0.5
    round_trip_cost_bps: float = 4.0
    initial_capital: float = 100_000
    gross_notional_per_trade: float = 10_000
    max_holding_sessions: int | None = None
    stop_z: float | None = None
    require_rolling_pass: bool = False
    require_regime_allowed: bool = False
    block_reentry_after_max_hold: bool = False

    def __post_init__(self):
        if self.formation_days < 3:
            raise ValueError("formation_days must be at least 3")
        for name in ("initial_capital", "gross_notional_per_trade", "entry_z"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.exit_z) or not 0 <= self.exit_z < self.entry_z:
            raise ValueError("exit_z must be nonnegative and below entry_z")
        if not np.isfinite(self.round_trip_cost_bps) or self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps must be finite and nonnegative")
        if self.max_holding_sessions is not None and (
            isinstance(self.max_holding_sessions, bool)
            or not isinstance(self.max_holding_sessions, int)
            or self.max_holding_sessions < 1
        ):
            raise ValueError("max_holding_sessions must be a positive integer")
        if self.stop_z is not None and (
            not np.isfinite(self.stop_z) or self.stop_z <= self.entry_z
        ):
            raise ValueError("stop_z must be finite and above entry_z")
