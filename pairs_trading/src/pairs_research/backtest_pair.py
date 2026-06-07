"""Barebones walk-forward pairs trading backtest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scan_cointegration import load_price_matrix


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
    max_holding_days: int | None = None
    stop_z: float | None = None
    require_rolling_pass: bool = False
    require_regime_allowed: bool = False
    block_reentry_after_max_hold: bool = False


def compute_walk_forward_signals(
    price_history: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Estimate hedge ratio and z-score each day using prior data only."""
    price_matrix = load_price_matrix(price_history)
    ticker_a = config.ticker_a.upper()
    ticker_b = config.ticker_b.upper()
    pair_prices = price_matrix[[ticker_a, ticker_b]].dropna()

    rows: list[dict[str, object]] = []
    for idx in range(config.formation_days, len(pair_prices)):
        formation_prices = pair_prices.iloc[idx - config.formation_days : idx]
        current_date = pair_prices.index[idx]
        current_prices = pair_prices.iloc[idx]

        formation_log = np.log(formation_prices[[ticker_a, ticker_b]])
        model = sm.OLS(
            formation_log[ticker_a],
            sm.add_constant(formation_log[ticker_b]),
        ).fit()
        intercept = float(model.params["const"])
        hedge_ratio = float(model.params[ticker_b])
        formation_spread = (
            formation_log[ticker_a]
            - intercept
            - hedge_ratio * formation_log[ticker_b]
        )
        current_spread = (
            np.log(current_prices[ticker_a])
            - intercept
            - hedge_ratio * np.log(current_prices[ticker_b])
        )
        zscore = float(
            (current_spread - formation_spread.mean()) / formation_spread.std()
        )

        rows.append(
            {
                "date": current_date,
                "ticker_a_price": float(current_prices[ticker_a]),
                "ticker_b_price": float(current_prices[ticker_b]),
                "spread": float(current_spread),
                "zscore": zscore,
                "hedge_ratio": hedge_ratio,
                "intercept": intercept,
                "formation_spread_mean": float(formation_spread.mean()),
                "formation_spread_std": float(formation_spread.std()),
            }
        )

    return pd.DataFrame(rows)


def run_backtest(
    signals: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a simple z-score entry/exit strategy on walk-forward signals."""
    position = 0
    open_trade: dict[str, object] | None = None
    trades: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    equity = config.initial_capital
    cumulative_pnl = 0.0
    prior_row = None
    blocked_reentry_direction: int | None = None

    for row in signals.itertuples(index=False):
        date = row.date
        zscore = float(row.zscore)
        spread = float(row.spread)
        action = "hold"
        daily_pnl = 0.0
        trading_cost = 0.0
        rolling_pass = bool(getattr(row, "rolling_pass", True))
        regime_allowed = bool(getattr(row, "regime_allowed", True))
        can_enter = (
            ((not config.require_rolling_pass) or rolling_pass)
            and ((not config.require_regime_allowed) or regime_allowed)
        )
        if blocked_reentry_direction is not None and abs(zscore) <= config.exit_z:
            blocked_reentry_direction = None

        if open_trade is not None and prior_row is not None:
            shares_a = float(open_trade["shares_a"])
            shares_b = float(open_trade["shares_b"])
            daily_pnl = (
                shares_a * (float(row.ticker_a_price) - float(prior_row.ticker_a_price))
                + shares_b * (float(row.ticker_b_price) - float(prior_row.ticker_b_price))
            )
            equity += daily_pnl
            cumulative_pnl += daily_pnl

        if position == 0:
            if (
                can_enter
                and zscore >= config.entry_z
                and blocked_reentry_direction != -1
            ):
                position = -1
                action = "short_spread_entry"
                hedge_ratio = float(row.hedge_ratio)
                base_notional = config.gross_notional_per_trade / (1 + abs(hedge_ratio))
                leg_a_notional = -base_notional
                leg_b_notional = hedge_ratio * base_notional
                entry_start_equity = equity
                trading_cost = config.gross_notional_per_trade * (
                    config.round_trip_cost_bps / 2
                ) / 10_000
                equity -= trading_cost
                cumulative_pnl -= trading_cost
                open_trade = {
                    "entry_date": date,
                    "direction": "short_spread",
                    "entry_z": zscore,
                    "entry_spread": spread,
                    "entry_hedge_ratio": float(row.hedge_ratio),
                    "entry_equity": equity,
                    "entry_start_equity": entry_start_equity,
                    "entry_ticker_a_price": float(row.ticker_a_price),
                    "entry_ticker_b_price": float(row.ticker_b_price),
                    "shares_a": leg_a_notional / float(row.ticker_a_price),
                    "shares_b": leg_b_notional / float(row.ticker_b_price),
                    "entry_cost_dollars": trading_cost,
                }
            elif (
                can_enter
                and zscore <= -config.entry_z
                and blocked_reentry_direction != 1
            ):
                position = 1
                action = "long_spread_entry"
                hedge_ratio = float(row.hedge_ratio)
                base_notional = config.gross_notional_per_trade / (1 + abs(hedge_ratio))
                leg_a_notional = base_notional
                leg_b_notional = -hedge_ratio * base_notional
                entry_start_equity = equity
                trading_cost = config.gross_notional_per_trade * (
                    config.round_trip_cost_bps / 2
                ) / 10_000
                equity -= trading_cost
                cumulative_pnl -= trading_cost
                open_trade = {
                    "entry_date": date,
                    "direction": "long_spread",
                    "entry_z": zscore,
                    "entry_spread": spread,
                    "entry_hedge_ratio": float(row.hedge_ratio),
                    "entry_equity": equity,
                    "entry_start_equity": entry_start_equity,
                    "entry_ticker_a_price": float(row.ticker_a_price),
                    "entry_ticker_b_price": float(row.ticker_b_price),
                    "shares_a": leg_a_notional / float(row.ticker_a_price),
                    "shares_b": leg_b_notional / float(row.ticker_b_price),
                    "entry_cost_dollars": trading_cost,
                }
        elif open_trade is not None:
            days_held = (pd.Timestamp(date) - pd.Timestamp(open_trade["entry_date"])).days
            exit_reason = None
            if position == 1 and zscore >= -config.exit_z:
                exit_reason = "mean_reversion"
            elif position == -1 and zscore <= config.exit_z:
                exit_reason = "mean_reversion"
            elif config.max_holding_days is not None and days_held >= config.max_holding_days:
                exit_reason = "max_holding_days"
            elif config.stop_z is not None and abs(zscore) >= config.stop_z:
                exit_reason = "stop_z"

            if exit_reason is not None:
                exit_cost = config.gross_notional_per_trade * (
                    config.round_trip_cost_bps / 2
                ) / 10_000
                equity -= exit_cost
                cumulative_pnl -= exit_cost
                trading_cost += exit_cost
                trade_pnl_dollars = (
                    equity - float(open_trade["entry_start_equity"])
                )
                net_pnl_bps = (
                    trade_pnl_dollars / config.gross_notional_per_trade
                ) * 10_000
                gross_pnl_bps = net_pnl_bps + config.round_trip_cost_bps

                action = f"{open_trade['direction']}_exit"
                trades.append(
                    {
                        **open_trade,
                        "exit_date": date,
                        "exit_reason": exit_reason,
                        "exit_z": zscore,
                        "exit_spread": spread,
                        "days_held": days_held,
                        "gross_pnl_bps": gross_pnl_bps,
                        "cost_bps": config.round_trip_cost_bps,
                        "net_pnl_bps": net_pnl_bps,
                        "pnl_dollars": trade_pnl_dollars,
                        "return_on_gross_notional": trade_pnl_dollars / config.gross_notional_per_trade,
                        "exit_ticker_a_price": float(row.ticker_a_price),
                        "exit_ticker_b_price": float(row.ticker_b_price),
                        "exit_cost_dollars": exit_cost,
                    }
                )
                if (
                    config.block_reentry_after_max_hold
                    and exit_reason == "max_holding_days"
                    and abs(zscore) > config.exit_z
                ):
                    blocked_reentry_direction = position
                position = 0
                open_trade = None

        daily_rows.append(
            {
                "date": date,
                "zscore": zscore,
                "spread": spread,
                "hedge_ratio": float(row.hedge_ratio),
                "position": position,
                "action": action,
                "rolling_pass": rolling_pass,
                "regime_allowed": regime_allowed,
                "daily_pnl": daily_pnl,
                "trading_cost": trading_cost,
                "net_daily_pnl": daily_pnl - trading_cost,
                "cumulative_pnl": cumulative_pnl,
                "equity": equity,
                "daily_return": (daily_pnl - trading_cost) / config.initial_capital,
                "gross_exposure": config.gross_notional_per_trade if position != 0 else 0.0,
            }
        )
        prior_row = row

    trades_df = pd.DataFrame(trades)
    daily_df = pd.DataFrame(daily_rows)
    summary_df = summarize_backtest(trades_df, daily_df, config)
    return trades_df, daily_df, summary_df


def summarize_backtest(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Create one summary row for the backtest."""
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "ticker_a": config.ticker_a.upper(),
                    "ticker_b": config.ticker_b.upper(),
                    "completed_trades": 0,
                    "total_net_pnl_bps": 0.0,
                }
            ]
        )

    summary = {
        "ticker_a": config.ticker_a.upper(),
        "ticker_b": config.ticker_b.upper(),
        "formation_days": config.formation_days,
        "entry_z": config.entry_z,
        "exit_z": config.exit_z,
        "round_trip_cost_bps": config.round_trip_cost_bps,
        "initial_capital": config.initial_capital,
        "gross_notional_per_trade": config.gross_notional_per_trade,
        "max_holding_days": config.max_holding_days,
        "stop_z": config.stop_z,
        "require_rolling_pass": config.require_rolling_pass,
        "require_regime_allowed": config.require_regime_allowed,
        "completed_trades": int(len(trades)),
        "win_rate": float(trades["net_pnl_bps"].gt(0).mean()),
        "avg_days_held": float(trades["days_held"].mean()),
        "median_days_held": float(trades["days_held"].median()),
        "avg_gross_pnl_bps": float(trades["gross_pnl_bps"].mean()),
        "avg_net_pnl_bps": float(trades["net_pnl_bps"].mean()),
        "total_net_pnl_bps": float(trades["net_pnl_bps"].sum()),
        "worst_net_pnl_bps": float(trades["net_pnl_bps"].min()),
        "best_net_pnl_bps": float(trades["net_pnl_bps"].max()),
        "total_pnl_dollars": float(daily["cumulative_pnl"].iloc[-1]),
        "total_return": float(daily["equity"].iloc[-1] / config.initial_capital - 1),
        "annualized_return": annualized_return(daily["equity"]),
        "annualized_volatility": annualized_volatility(daily["daily_return"]),
        "sharpe": sharpe_ratio(daily["daily_return"]),
        "max_drawdown": max_drawdown(daily["equity"]),
        "exposure_pct": float(daily["position"].ne(0).mean()),
        "first_trade_date": trades["entry_date"].min(),
        "last_trade_date": trades["entry_date"].max(),
    }
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
    return float(daily_returns.std() * np.sqrt(252))


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
    base_dir = Path(__file__).resolve().parents[2]
    paths = write_backtest_outputs(
        price_history_csv=base_dir / "data" / "etf_price_history.csv",
        output_dir=base_dir / "outputs",
    )
    for path in paths:
        print(f"Wrote {path}")
