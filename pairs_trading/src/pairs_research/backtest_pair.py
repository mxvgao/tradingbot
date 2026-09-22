"""Causal next-session-close pairs benchmark with explicit boundary liquidation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .scan_cointegration import load_price_matrix
from .config import BacktestConfig
from .models import Session, RunSpec
from .strategy import observe_close
from .engine import process_session
from .ledger import MemoryLedger


SIGNAL_COLUMNS = [
    "date",
    "ticker_a_price",
    "ticker_b_price",
    "spread",
    "zscore",
    "hedge_ratio",
    "intercept",
    "formation_spread_mean",
    "formation_spread_std",
]
TRADE_COLUMNS = [
    "entry_signal_date",
    "entry_date",
    "entry_session",
    "direction",
    "entry_z",
    "entry_spread",
    "entry_hedge_ratio",
    "entry_start_equity",
    "entry_equity",
    "entry_ticker_a_price",
    "entry_ticker_b_price",
    "shares_a",
    "shares_b",
    "entry_gross_notional",
    "entry_cost_dollars",
    "exit_signal_date",
    "exit_date",
    "exit_reason",
    "exit_z",
    "exit_spread",
    "sessions_held",
    "days_held",
    "gross_pnl_dollars",
    "gross_pnl_bps",
    "cost_bps",
    "net_pnl_bps",
    "pnl_dollars",
    "return_on_gross_notional",
    "exit_ticker_a_price",
    "exit_ticker_b_price",
    "exit_cost_dollars",
]
DAILY_COLUMNS = [
    "date",
    "zscore",
    "spread",
    "hedge_ratio",
    "position",
    "action",
    "executed_signal_date",
    "pending_order",
    "rolling_pass",
    "regime_allowed",
    "daily_pnl",
    "trading_cost",
    "net_daily_pnl",
    "realized_pnl",
    "unrealized_pnl",
    "cumulative_pnl",
    "equity",
    "daily_return",
    "gross_exposure",
]


def compute_walk_forward_signals(
    price_history: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Estimate hedge ratio and z-score each day using prior data only."""
    dates = pd.to_datetime(price_history["date"])
    if dates.isna().any():
        raise ValueError("Price history contains invalid session dates")
    history = price_history.copy()
    history["date"] = dates
    if history.duplicated(["date", "ticker"]).any():
        raise ValueError("Price history contains duplicate ticker/session observations")
    # pivot_table drops entirely missing rows; retain the supplied session grid.
    price_matrix = load_price_matrix(history).reindex(
        pd.DatetimeIndex(dates.unique()).sort_values()
    )
    ticker_a = config.ticker_a.upper()
    ticker_b = config.ticker_b.upper()
    pair_prices = price_matrix[[ticker_a, ticker_b]]
    # Allow different listing dates, but never compress a missing session inside
    # the pair's common history into an apparently consecutive observation.
    valid = pair_prices.notna().all(axis=1)
    if not valid.any():
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    pair_prices = pair_prices.loc[valid[valid].index[0] :]
    values = pair_prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Missing or invalid pair prices inside the common session history")

    rows = []
    history = []
    for date, prices in pair_prices.iterrows():
        history.append(
            Session(
                date=str(pd.Timestamp(date).date()),
                price_a=float(prices[ticker_a]),
                price_b=float(prices[ticker_b]),
            )
        )
        history = history[-(config.formation_days + 1) :]
        observation = observe_close(history, config)
        if observation.ready:
            rows.append(observation.as_signal_row())
    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)


def run_backtest(
    signals: pd.DataFrame,
    config: BacktestConfig,
    *,
    liquidate_at_end: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Batch adapter for the same incremental workflow used by SQLite replay.

    Historical callers retain explicit terminal liquidation by default. Use
    liquidate_at_end=False to retain final marked positions and pending orders.
    """
    frame = validate_signals(signals)
    terminal = str(frame.date.iloc[-1].date()) if liquidate_at_end and len(frame) else None
    ledger = MemoryLedger(
        RunSpec("batch", config, input_kind="signals", liquidation_session=terminal)
    )
    for row in frame.to_dict("records"):
        process_session(ledger, Session.from_signal_row(row))
    trades_df = pd.DataFrame(ledger.trades, columns=TRADE_COLUMNS)
    daily_df = pd.DataFrame(ledger.daily, columns=DAILY_COLUMNS)
    for column in ("entry_signal_date", "entry_date", "exit_signal_date", "exit_date"):
        trades_df[column] = pd.to_datetime(trades_df[column])
    for column in ("date", "executed_signal_date"):
        daily_df[column] = pd.to_datetime(daily_df[column])
    return trades_df, daily_df, summarize_backtest(trades_df, daily_df, config)


def validate_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    required = {"date", "ticker_a_price", "ticker_b_price", "zscore", "spread", "hedge_ratio"}
    if missing := required.difference(signals.columns):
        raise ValueError(f"Missing signal columns: {sorted(missing)}")
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = frame["date"]
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Signal dates must be unique, valid and strictly increasing")
    prices = frame[["ticker_a_price", "ticker_b_price"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Both legs need finite positive prices on every supplied session")
    return frame


def summarize_backtest(trades, daily, config) -> pd.DataFrame:
    """Consistent summary schema, including empty and no-trade windows."""
    has_trades, has_days = not trades.empty, not daily.empty
    returns = daily["daily_return"] if has_days else pd.Series(dtype=float)
    equity = (
        pd.concat([pd.Series([config.initial_capital]), daily["equity"]], ignore_index=True)
        if has_days
        else pd.Series([config.initial_capital])
    )
    summary = {
        "ticker_a": config.ticker_a.upper(),
        "ticker_b": config.ticker_b.upper(),
        "execution_convention": "signal_close_t_fill_close_t_plus_1",
        "model_version": "0.2.0",
        "holding_period_unit": "trading_sessions",
        "formation_days": config.formation_days,
        "entry_z": config.entry_z,
        "exit_z": config.exit_z,
        "round_trip_cost_bps": config.round_trip_cost_bps,
        "initial_capital": config.initial_capital,
        "gross_notional_per_trade": config.gross_notional_per_trade,
        "max_holding_sessions": config.max_holding_sessions,
        "stop_z": config.stop_z,
        "require_rolling_pass": config.require_rolling_pass,
        "require_regime_allowed": config.require_regime_allowed,
        "completed_trades": len(trades),
        "win_rate": float(trades["net_pnl_bps"].gt(0).mean()) if has_trades else 0.0,
        "total_pnl_dollars": float(equity.iloc[-1] - config.initial_capital),
        "total_return": float(equity.iloc[-1] / config.initial_capital - 1),
        "annualized_return": float(
            (equity.iloc[-1] / config.initial_capital) ** (252 / len(daily)) - 1
        )
        if has_days and equity.iloc[-1] > 0
        else 0.0,
        "annualized_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "exposure_pct": float(daily["position"].ne(0).mean()) if has_days else 0.0,
        "first_trade_date": trades["entry_date"].min() if has_trades else pd.NaT,
        "last_trade_date": trades["entry_date"].max() if has_trades else pd.NaT,
    }
    for name, column, operation in [
        ("avg_days_held", "sessions_held", "mean"),
        ("median_days_held", "sessions_held", "median"),
        ("avg_sessions_held", "sessions_held", "mean"),
        ("avg_gross_pnl_bps", "gross_pnl_bps", "mean"),
        ("avg_net_pnl_bps", "net_pnl_bps", "mean"),
        ("total_net_pnl_bps", "net_pnl_bps", "sum"),
        ("worst_net_pnl_bps", "net_pnl_bps", "min"),
        ("best_net_pnl_bps", "net_pnl_bps", "max"),
    ]:
        summary[name] = float(getattr(trades[column], operation)()) if has_trades else 0.0
    return pd.DataFrame([summary])


def annualized_return(equity: pd.Series) -> float:
    """Annualized return from an equity curve."""
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(equity) / 252
    return float((1 + total_return) ** (1 / years) - 1)


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized volatility from daily returns."""
    return float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 1 else 0.0


def sharpe_ratio(daily_returns: pd.Series) -> float:
    """Simple zero-risk-free Sharpe ratio."""
    vol = annualized_volatility(daily_returns)
    if vol == 0:
        return 0.0
    return float((daily_returns.mean() * 252) / vol)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown from an equity curve."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def write_backtest_outputs(
    price_history_csv: str | Path,
    output_dir: str | Path,
    config: BacktestConfig = BacktestConfig(),
) -> tuple[Path, Path, Path]:
    """Run the backtest and write trades, daily, and summary CSVs."""
    price_history = pd.read_csv(price_history_csv)
    signals = compute_walk_forward_signals(price_history, config)
    trades, daily, summary = run_backtest(signals, config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_name = f"{config.ticker_a.upper()}_{config.ticker_b.upper()}"
    trades_path = output_dir / f"backtest_{pair_name}_trades.csv"
    daily_path = output_dir / f"backtest_{pair_name}_daily.csv"
    summary_path = output_dir / f"backtest_{pair_name}_summary.csv"

    trades.to_csv(trades_path, index=False)
    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    return trades_path, daily_path, summary_path


if __name__ == "__main__":
    base_dir = Path.cwd() / "pairs_trading"
    paths = write_backtest_outputs(
        price_history_csv=base_dir / "data" / "etf_price_history.csv",
        output_dir=base_dir / "outputs",
    )
    for path in paths:
        print(f"Wrote {path}")
