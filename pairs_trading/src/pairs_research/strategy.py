"""Pure signal calculation and target decisions: no I/O, clock or execution."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import BacktestConfig
from .models import Decision, Observation, Session, StrategyState, finite_or_none


def observe_close(history: Sequence[Session], config: BacktestConfig) -> Observation:
    """Calculate only the current observation, fitting exclusively to prior closes."""
    if not history:
        raise ValueError("History must include the current session")
    if any(left.date >= right.date for left, right in zip(history, history[1:])):
        raise ValueError(
            "Signal history must contain strictly increasing sessions through the current close"
        )
    current = history[-1]
    if current.observation is not None:
        return current.observation
    if len(history) <= config.formation_days:
        return Observation(current.date, current.price_a, current.price_b, ready=False)
    formation = history[-config.formation_days - 1 : -1]
    logs = np.log(
        pd.DataFrame(
            {
                config.ticker_a.upper(): [s.price_a for s in formation],
                config.ticker_b.upper(): [s.price_b for s in formation],
            }
        )
    )
    a, b = config.ticker_a.upper(), config.ticker_b.upper()
    model = sm.OLS(logs[a], sm.add_constant(logs[b], has_constant="add")).fit()
    intercept, hedge = float(model.params["const"]), float(model.params[b])
    spreads = logs[a] - intercept - hedge * logs[b]
    spread = float(np.log(current.price_a) - intercept - hedge * np.log(current.price_b))
    mean, std = float(spreads.mean()), float(spreads.std())
    z = (spread - mean) / std if std > 1e-12 else None
    return Observation(
        current.date,
        current.price_a,
        current.price_b,
        finite_or_none(z),
        spread,
        hedge,
        intercept,
        mean,
        std,
    )


def generate_target(
    history: Sequence[Session],
    state: StrategyState,
    config: BacktestConfig,
) -> Decision:
    """Return a target and reason; never create an order or inspect future data.

    The caller supplies state after prior eligible fills and a history ending at
    the current completed close. Entry/exit quantities belong to the planner.
    """
    observation = observe_close(history, config)
    z = observation.zscore
    blocked = state.blocked_direction
    if z is not None and abs(z) <= config.exit_z:
        blocked = None
    target, reason = state.position, "hold"
    if state.open_trade is not None:
        held = state.session_index - state.open_trade["entry_session"]
        if config.max_holding_sessions is not None and held + 1 >= config.max_holding_sessions:
            target, reason = 0, "max_holding_sessions"
        elif z is not None:
            if (state.position == 1 and z >= -config.exit_z) or (
                state.position == -1 and z <= config.exit_z
            ):
                target, reason = 0, "mean_reversion"
            elif config.stop_z is not None and abs(z) >= config.stop_z:
                target, reason = 0, "stop_z"
    elif state.last_exit_session != observation.date:
        can_enter = (
            observation.ready
            and z is not None
            and observation.hedge_ratio is not None
            and (not config.require_rolling_pass or observation.rolling_pass)
            and (not config.require_regime_allowed or observation.regime_allowed)
        )
        if can_enter:
            direction = -1 if z >= config.entry_z else (1 if z <= -config.entry_z else 0)
            if direction and direction != blocked:
                target, reason = direction, "entry"
    return Decision(target, reason, observation, blocked)
