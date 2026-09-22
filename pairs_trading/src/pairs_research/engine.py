"""One incremental session workflow for memory backtests and durable replay."""

from copy import deepcopy
import math
from typing import Protocol

from .config import BacktestConfig, ENGINE_VERSION, STRATEGY_VERSION
from .models import (
    Decision,
    FillEvent,
    Order,
    RunSpec,
    Session,
    SessionResult,
    StrategyState,
    event_id,
)
from .strategy import generate_target


class FillSource(Protocol):
    version: str

    def fill(self, order: Order, session: Session, config: BacktestConfig) -> list[FillEvent]: ...


class SimulatedCloseExecution:
    """Full simultaneous benchmark fills; separate execution.py models legging risk."""

    version = "simulated-close-v1"

    def fill(self, order: Order, session: Session, config: BacktestConfig) -> list[FillEvent]:
        rate = config.round_trip_cost_bps / 2 / 10_000
        return [
            FillEvent(
                event_id(order.order_id, leg),
                order.order_id,
                session.date,
                leg,
                shares,
                price,
                abs(shares * price) * rate,
            )
            for leg, shares, price in (
                ("a", order.shares_a, session.price_a),
                ("b", order.shares_b, session.price_b),
            )
            if shares
        ]


def plan_order(decision: Decision, state: StrategyState, spec: RunSpec) -> Order | None:
    """Convert a target change into fixed shares eligible at the following close."""
    if decision.target_position == state.position:
        return None
    obs = decision.observation
    if state.position == 0:
        purpose = "entry"
        base = spec.config.gross_notional_per_trade / (1 + abs(obs.hedge_ratio))
        shares_a = decision.target_position * base / obs.price_a
        shares_b = -decision.target_position * obs.hedge_ratio * base / obs.price_b
    else:
        purpose = "exit"
        shares_a, shares_b = -state.open_trade["shares_a"], -state.open_trade["shares_b"]
    return Order(
        event_id(spec.strategy_id, spec.pair_id, obs.date, purpose),
        obs.date,
        purpose,
        state.session_index + 1,
        shares_a,
        shares_b,
        decision.target_position,
        decision.reason,
        obs,
    )


def _checked_fills(
    source: FillSource, order: Order, session: Session, state: StrategyState, spec: RunSpec
) -> list[FillEvent]:
    if order.order_purpose == "boundary":
        if session.date != spec.liquidation_session:
            raise ValueError("Boundary fills require the run's predeclared liquidation session")
    elif (
        session.date <= order.signal_session or state.session_index != order.eligible_session_index
    ):
        raise ValueError("Normal fills require the session after their signal session")
    fills = source.fill(order, session, spec.config)
    seen = set()
    quantities = {"a": 0.0, "b": 0.0}
    for fill in fills:
        if (
            not fill.fill_id
            or fill.fill_id in seen
            or fill.order_id != order.order_id
            or fill.fill_session != session.date
            or fill.leg not in quantities
            or not all(math.isfinite(v) for v in (fill.shares, fill.price, fill.fee))
            or fill.price <= 0
            or fill.fee < 0
            or fill.shares == 0
        ):
            raise ValueError("Invalid execution fill")
        target = order.shares_a if fill.leg == "a" else order.shares_b
        if fill.shares * target <= 0:
            raise ValueError("Fill direction differs from the order")
        seen.add(fill.fill_id)
        quantities[fill.leg] += fill.shares
    if not (
        math.isclose(quantities["a"], order.shares_a, abs_tol=1e-9)
        and math.isclose(quantities["b"], order.shares_b, abs_tol=1e-9)
    ):
        raise ValueError(
            "Benchmark requires complete pair fills; use execution.replay_fills for partial-fill risk"
        )
    return fills


def _apply_fills(
    state: StrategyState,
    order: Order,
    fills: list[FillEvent],
    session: Session,
    spec: RunSpec,
    starting_equity: float,
) -> dict | None:
    fee = sum(f.fee for f in fills)
    state.cash -= sum(f.shares * f.price + f.fee for f in fills)
    prices = {}
    for leg in ("a", "b"):
        group = [f for f in fills if f.leg == leg]
        quantity = sum(f.shares for f in group)
        prices[leg] = (
            sum(f.shares * f.price for f in group) / quantity
            if quantity
            else (session.price_a if leg == "a" else session.price_b)
        )
    if order.order_purpose == "entry":
        obs = order.observation
        state.position = order.target_position
        state.open_trade = {
            "entry_signal_date": order.signal_session,
            "entry_date": session.date,
            "entry_session": state.session_index,
            "direction": "long_spread" if state.position == 1 else "short_spread",
            "entry_z": obs.zscore,
            "entry_spread": obs.spread,
            "entry_hedge_ratio": obs.hedge_ratio,
            "entry_start_equity": starting_equity,
            "entry_equity": state.cash
            + order.shares_a * session.price_a
            + order.shares_b * session.price_b,
            "entry_ticker_a_price": prices["a"],
            "entry_ticker_b_price": prices["b"],
            "shares_a": order.shares_a,
            "shares_b": order.shares_b,
            "entry_gross_notional": sum(abs(f.shares * f.price) for f in fills),
            "entry_cost_dollars": fee,
        }
        return None
    opened = state.open_trade
    gross = opened["shares_a"] * (prices["a"] - opened["entry_ticker_a_price"]) + opened[
        "shares_b"
    ] * (prices["b"] - opened["entry_ticker_b_price"])
    costs = opened["entry_cost_dollars"] + fee
    net = gross - costs
    state.realized_pnl += net
    held = state.session_index - opened["entry_session"]
    scale = 10_000 / spec.config.gross_notional_per_trade
    trade = {
        **opened,
        "exit_signal_date": None if order.order_purpose == "boundary" else order.signal_session,
        "exit_date": session.date,
        "exit_reason": order.reason,
        "exit_z": order.observation.zscore,
        "exit_spread": order.observation.spread,
        "sessions_held": held,
        "days_held": held,
        "gross_pnl_dollars": gross,
        "gross_pnl_bps": gross * scale,
        "cost_bps": costs * scale,
        "net_pnl_bps": net * scale,
        "pnl_dollars": net,
        "return_on_gross_notional": net / spec.config.gross_notional_per_trade,
        "exit_ticker_a_price": prices["a"],
        "exit_ticker_b_price": prices["b"],
        "exit_cost_dollars": fee,
    }
    if spec.config.block_reentry_after_max_hold and order.reason == "max_holding_sessions":
        state.blocked_direction = state.position
    state.position, state.open_trade = 0, None
    state.last_exit_session = session.date
    return trade


def process_session(
    ledger, session: Session, *, execution: FillSource | None = None
) -> SessionResult | None:
    """Commit one close atomically; return None for an already committed session.

    Both batch and SQLite runners call this function. A ledger transaction owns
    the read/check/transition/write/checkpoint unit. No price source or broker
    access belongs in the strategy. Fill sources must be side-effect-free here;
    sending external orders requires a later transactional-outbox adapter.
    """
    spec = ledger.spec
    execution = execution or SimulatedCloseExecution()
    if spec.engine_version != ENGINE_VERSION or spec.strategy_version != STRATEGY_VERSION:
        raise ValueError("Run engine/strategy version is incompatible with this process")
    if execution.version != spec.execution_version:
        raise ValueError("Execution provider version differs from the run")
    with ledger.transaction():
        if ledger.has_session(session.date):
            return None
        state = deepcopy(ledger.load_state())
        if (
            state.last_processed_session is not None
            and session.date <= state.last_processed_session
        ):
            raise ValueError("Sessions must be processed strictly in chronological order")
        if spec.liquidation_session is not None and session.date > spec.liquidation_session:
            raise ValueError(
                "Cannot append sessions after a terminal liquidation; use a new run ID"
            )
        if (session.observation is not None) != (spec.input_kind == "signals"):
            raise ValueError("Session input kind differs from the run configuration")
        state.session_index += 1
        history = (state.history + [session])[-(spec.config.formation_days + 1) :]
        starting_equity = state.equity
        boundary = session.date == spec.liquidation_session
        orders, fills, trade = [], [], None
        action, executed_signal_date = "hold", None
        pending = state.pending
        state.pending = None
        # Endpoint policy belongs to this workflow, never to generate_target.
        if boundary and pending is not None:
            orders.append((pending, "cancelled"))
            pending = None
        if boundary and state.open_trade is not None:
            from .strategy import observe_close

            obs = observe_close(history, spec.config)
            pending = Order(
                event_id(spec.strategy_id, spec.pair_id, session.date, "boundary"),
                session.date,
                "boundary",
                state.session_index,
                -state.open_trade["shares_a"],
                -state.open_trade["shares_b"],
                0,
                "test_boundary",
                obs,
            )
        if pending is not None:
            fills = _checked_fills(execution, pending, session, state, spec)
            prior_direction = state.open_trade["direction"] if state.open_trade else None
            trade = _apply_fills(state, pending, fills, session, spec, starting_equity)
            action = (
                f"{state.open_trade['direction']}_entry"
                if pending.order_purpose == "entry"
                else f"{prior_direction}_exit"
            )
            executed_signal_date = None if boundary else pending.signal_session
            orders.append((pending, "filled"))
        decision = generate_target(history, state, spec.config)
        state.blocked_direction = decision.blocked_direction
        if not boundary:
            state.pending = plan_order(decision, state, spec)
            if state.pending is not None:
                orders.append((state.pending, "pending"))
        unrealized = gross_exposure = value = 0.0
        if state.open_trade is not None:
            opened = state.open_trade
            value = opened["shares_a"] * session.price_a + opened["shares_b"] * session.price_b
            unrealized = (
                opened["shares_a"] * (session.price_a - opened["entry_ticker_a_price"])
                + opened["shares_b"] * (session.price_b - opened["entry_ticker_b_price"])
                - opened["entry_cost_dollars"]
            )
            gross_exposure = abs(opened["shares_a"] * session.price_a) + abs(
                opened["shares_b"] * session.price_b
            )
        state.equity = state.cash + value
        net_pnl = state.equity - starting_equity
        trading_cost = sum(f.fee for f in fills)
        obs = decision.observation
        daily = {
            "date": session.date,
            "zscore": obs.zscore,
            "spread": obs.spread,
            "hedge_ratio": obs.hedge_ratio,
            "position": state.position,
            "action": action,
            "executed_signal_date": executed_signal_date,
            "pending_order": state.pending.order_purpose if state.pending else "none",
            "rolling_pass": obs.rolling_pass,
            "regime_allowed": obs.regime_allowed,
            "daily_pnl": net_pnl + trading_cost,
            "trading_cost": trading_cost,
            "net_daily_pnl": net_pnl,
            "realized_pnl": state.realized_pnl,
            "unrealized_pnl": unrealized,
            "cumulative_pnl": state.equity - spec.config.initial_capital,
            "equity": state.equity,
            "daily_return": net_pnl / starting_equity if starting_equity else 0.0,
            "gross_exposure": gross_exposure,
            "cash": state.cash,
            "shares_a": state.open_trade["shares_a"] if state.open_trade else 0.0,
            "shares_b": state.open_trade["shares_b"] if state.open_trade else 0.0,
            "price_a": session.price_a,
            "price_b": session.price_b,
        }
        state.history = history[-spec.config.formation_days :]
        state.last_processed_session = session.date
        if not math.isclose(state.realized_pnl + unrealized, daily["cumulative_pnl"], abs_tol=1e-7):
            raise ArithmeticError("Cash, position and trade P&L do not reconcile")
        result = SessionResult(state, decision, daily, orders, fills, trade)
        ledger.record(result)
        return result
