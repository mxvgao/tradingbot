"""Resumable daily replay CLI. No broker or network access is used."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from collections.abc import Iterable, Iterator

import pandas as pd

from .config import BacktestConfig
from .engine import process_session
from .ledger import SQLiteLedger
from .models import RunSpec, Session, session_date


def sessions_from_prices(prices: pd.DataFrame, config: BacktestConfig) -> Iterator[Session]:
    """Source adapter; never pass the complete file to the strategy engine."""
    required = {"date", "ticker", "adj_close"}
    if missing := required.difference(prices.columns):
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    frame = prices.copy()
    frame["date"] = frame["date"].map(session_date)
    frame["ticker"] = frame["ticker"].str.upper()
    for day, rows in frame.groupby("date", sort=True):
        values = []
        for ticker in (config.ticker_a.upper(), config.ticker_b.upper()):
            leg = rows[rows.ticker == ticker]
            if len(leg) != 1:
                raise ValueError(f"Session {day} requires exactly one price for {ticker}")
            values.append(float(leg.adj_close.iloc[0]))
        yield Session(day, *values)


def sessions_from_signals(signals: pd.DataFrame) -> Iterator[Session]:
    """Accept causal precomputed observations, including externally fitted gates."""
    # Preserve source order; the workflow rejects out-of-order new sessions.
    for row in signals.to_dict("records"):
        yield Session.from_signal_row(row)


def replay(ledger, sessions: Iterable[Session], *, stop_after: int | None = None) -> dict:
    """Stopping is a pause, never an implicit liquidation."""
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    processed = skipped = 0
    for session in sessions:
        result = process_session(ledger, session)
        if result is None:
            skipped += 1
        else:
            processed += 1
            if stop_after is not None and processed >= stop_after:
                break
    state = ledger.load_state()
    return {
        "strategy_id": ledger.spec.strategy_id,
        "pair_id": ledger.spec.pair_id,
        "processed": processed,
        "skipped": skipped,
        "last_processed_session": state.last_processed_session,
        "cash": state.cash,
        "equity": state.equity,
        "position": state.position,
        "realized_pnl": state.realized_pnl,
        "unrealized_pnl": (state.equity - ledger.spec.config.initial_capital - state.realized_pnl)
        if state.open_trade
        else 0.0,
        "pending_order": state.pending.order_purpose if state.pending else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one session at a time through the shared strategy and SQLite ledger."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Persistent strategy instance; changing config requires a new ID.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prices", type=Path, help="Long CSV: date,ticker,adj_close; includes formation history."
    )
    source.add_argument(
        "--signals", type=Path, help="Causal precomputed signal CSV, identical to the batch input."
    )
    parser.add_argument("--config", type=Path, help="JSON object of BacktestConfig fields.")
    parser.add_argument("--ticker-a")
    parser.add_argument("--ticker-b")
    parser.add_argument("--formation-days", type=int)
    parser.add_argument("--entry-z", type=float)
    parser.add_argument("--exit-z", type=float)
    parser.add_argument("--max-holding-sessions", type=int)
    parser.add_argument("--stop-z", type=float)
    parser.add_argument("--round-trip-cost-bps", type=float)
    parser.add_argument("--initial-capital", type=float)
    parser.add_argument("--gross-notional-per-trade", type=float)
    parser.add_argument(
        "--start", help="First input session (include formation history for raw prices)."
    )
    parser.add_argument("--end", help="Last supplied session to process, inclusive.")
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Pause after N newly committed sessions; resume with the same command.",
    )
    parser.add_argument(
        "--liquidate-at-end",
        action="store_true",
        help="Declare the selected final session as the run's immutable liquidation boundary.",
    )
    args = parser.parse_args()
    try:
        settings = json.loads(args.config.read_text()) if args.config else {}
        for name in asdict(BacktestConfig()):
            value = getattr(args, name, None)
            if value is not None:
                settings[name] = value
        config = BacktestConfig(**settings)
        if args.prices and (config.require_rolling_pass or config.require_regime_allowed):
            raise ValueError(
                "Raw replay does not fit external gates; supply causal --signals with gate flags"
            )
        data = pd.read_csv(args.prices or args.signals)
        data["date"] = data["date"].map(session_date)
        if args.start:
            data = data[data.date >= session_date(args.start)]
        if args.end:
            end = session_date(args.end)
            if args.liquidate_at_end and end not in set(data.date):
                raise ValueError("The declared liquidation endpoint has no supplied session")
            data = data[data.date <= end]
        if data.empty:
            raise ValueError("No input sessions in the selected range")
        terminal = max(data.date) if args.liquidate_at_end else None
        spec = RunSpec(args.run_id, config, "prices" if args.prices else "signals", terminal)
        source = sessions_from_prices(data, config) if args.prices else sessions_from_signals(data)
        with SQLiteLedger(args.database, spec) as ledger:
            print(json.dumps(replay(ledger, source, stop_after=args.stop_after), sort_keys=True))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Replay failed: {exc}\n")


if __name__ == "__main__":
    main()
