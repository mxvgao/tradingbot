"""Combine individual pair backtests into a simple multi-pair portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_pair import annualized_return, annualized_volatility, max_drawdown, sharpe_ratio


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 100_000
    max_pairs: int = 5
    min_completed_trades: int = 5
    require_positive_return: bool = True
    allocation_method: str = "vol_target"
    total_gross_budget: float = 50_000
    max_pairs_per_group: int = 2
    min_pair_gross: float = 5_000
    max_pair_gross: float = 20_000


def pair_name(ticker_a: str, ticker_b: str) -> str:
    """Return the pair file stem."""
    return f"{ticker_a.upper()}_{ticker_b.upper()}"


def select_pairs(
    backtest_comparison: pd.DataFrame,
    config: PortfolioConfig = PortfolioConfig(),
) -> pd.DataFrame:
    """Select pairs for the portfolio from ranked backtest results."""
    candidates = backtest_comparison.copy()
    candidates = candidates[candidates["completed_trades"].ge(config.min_completed_trades)]
    if config.require_positive_return:
        candidates = candidates[candidates["total_return"].gt(0)]

    if candidates.empty:
        candidates = backtest_comparison.copy()

    sort_cols = [col for col in ["ranking_score", "sharpe", "total_return"] if col in candidates]
    ranked = candidates.sort_values(sort_cols, ascending=False)

    selected_rows = []
    group_counts: dict[str, int] = {}
    for row in ranked.itertuples(index=False):
        group = str(getattr(row, "group", "unknown"))
        if group_counts.get(group, 0) >= config.max_pairs_per_group:
            continue
        selected_rows.append(row._asdict())
        group_counts[group] = group_counts.get(group, 0) + 1
        if len(selected_rows) >= config.max_pairs:
            break

    if not selected_rows:
        return ranked.head(config.max_pairs)

    return pd.DataFrame(selected_rows)


def load_pair_daily(output_dir: Path, ticker_a: str, ticker_b: str) -> pd.DataFrame:
    """Load one pair's daily backtest output."""
    path = output_dir / f"backtest_{pair_name(ticker_a, ticker_b)}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pair daily file: {path}")

    daily = pd.read_csv(path, parse_dates=["date"])
    daily = daily[["date", "net_daily_pnl", "gross_exposure", "position"]].copy()
    daily["pair"] = pair_name(ticker_a, ticker_b)
    return daily


def estimate_pair_risk(pair_daily: pd.DataFrame, original_gross: float) -> float:
    """Estimate annualized P&L volatility per dollar of pair gross notional."""
    if original_gross <= 0:
        return np.nan
    returns_on_gross = pair_daily["net_daily_pnl"] / original_gross
    risk = float(returns_on_gross.std() * np.sqrt(252))
    return risk if risk > 0 else np.nan


def add_allocations(
    selected_pairs: pd.DataFrame,
    output_dir: Path,
    config: PortfolioConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Assign gross notional to selected pairs and return loaded daily data."""
    selected = selected_pairs.copy()
    daily_by_pair: dict[str, pd.DataFrame] = {}
    risks = []
    original_grosses = []

    for row in selected.itertuples(index=False):
        name = pair_name(row.ticker_a, row.ticker_b)
        daily = load_pair_daily(output_dir, row.ticker_a, row.ticker_b)
        original_gross = float(getattr(row, "gross_notional_per_trade", 10_000))
        risk = estimate_pair_risk(daily, original_gross)
        daily_by_pair[name] = daily
        risks.append(risk)
        original_grosses.append(original_gross)

    selected["pair"] = [pair_name(row.ticker_a, row.ticker_b) for row in selected.itertuples(index=False)]
    selected["original_gross_notional"] = original_grosses
    selected["pair_risk"] = risks

    if selected.empty:
        selected["allocated_gross_notional"] = []
        selected["allocation_weight"] = []
        selected["pnl_scale"] = []
        return selected, daily_by_pair

    if config.allocation_method == "equal_gross":
        raw_weights = np.ones(len(selected), dtype=float)
    elif config.allocation_method == "score_weighted" and "ranking_score" in selected:
        raw_weights = selected["ranking_score"].clip(lower=0).to_numpy(dtype=float)
        if raw_weights.sum() == 0:
            raw_weights = np.ones(len(selected), dtype=float)
    elif config.allocation_method == "vol_target":
        safe_risk = selected["pair_risk"].replace([np.inf, -np.inf], np.nan)
        fallback = safe_risk.median()
        if pd.isna(fallback) or fallback <= 0:
            fallback = 1.0
        safe_risk = safe_risk.fillna(fallback).clip(lower=fallback / 10)
        raw_weights = (1 / safe_risk).to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown allocation_method: {config.allocation_method}")

    weights = raw_weights / raw_weights.sum()
    allocated = weights * config.total_gross_budget
    allocated = np.clip(allocated, config.min_pair_gross, config.max_pair_gross)
    if allocated.sum() > config.total_gross_budget:
        allocated = allocated / allocated.sum() * config.total_gross_budget

    selected["allocated_gross_notional"] = allocated
    selected["allocation_weight"] = selected["allocated_gross_notional"] / selected["allocated_gross_notional"].sum()
    selected["pnl_scale"] = selected["allocated_gross_notional"] / selected["original_gross_notional"]
    selected["allocation_method"] = config.allocation_method
    return selected, daily_by_pair


def build_portfolio_daily(
    selected_pairs: pd.DataFrame,
    output_dir: str | Path,
    config: PortfolioConfig = PortfolioConfig(),
) -> pd.DataFrame:
    """Aggregate selected pair daily P&L into a portfolio equity curve."""
    output_dir = Path(output_dir)
    selected_pairs, daily_by_pair = add_allocations(selected_pairs, output_dir, config)
    frames = []
    for row in selected_pairs.itertuples(index=False):
        daily = daily_by_pair[row.pair].copy()
        daily["net_daily_pnl"] = daily["net_daily_pnl"] * float(row.pnl_scale)
        daily["gross_exposure"] = daily["gross_exposure"] * float(row.pnl_scale)
        daily["allocated_gross_notional"] = float(row.allocated_gross_notional)
        frames.append(daily)
    if not frames:
        return pd.DataFrame()

    pair_daily = pd.concat(frames, ignore_index=True)
    portfolio = (
        pair_daily.groupby("date", as_index=False)
        .agg(
            net_daily_pnl=("net_daily_pnl", "sum"),
            gross_exposure=("gross_exposure", "sum"),
            active_pairs=("position", lambda x: int((x != 0).sum())),
        )
        .sort_values("date")
    )
    portfolio["cumulative_pnl"] = portfolio["net_daily_pnl"].cumsum()
    portfolio["equity"] = config.initial_capital + portfolio["cumulative_pnl"]
    portfolio["daily_return"] = portfolio["net_daily_pnl"] / config.initial_capital
    portfolio["gross_exposure_pct"] = portfolio["gross_exposure"] / config.initial_capital
    return portfolio


def summarize_portfolio(
    portfolio_daily: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    config: PortfolioConfig = PortfolioConfig(),
) -> pd.DataFrame:
    """Create a one-row portfolio summary."""
    if portfolio_daily.empty:
        return pd.DataFrame(
            [{"initial_capital": config.initial_capital, "selected_pairs": 0}]
        )

    summary = {
        "initial_capital": config.initial_capital,
        "selected_pairs": int(len(selected_pairs)),
        "allocation_method": config.allocation_method,
        "total_gross_budget": config.total_gross_budget,
        "max_pairs_per_group": config.max_pairs_per_group,
        "total_pnl_dollars": float(portfolio_daily["cumulative_pnl"].iloc[-1]),
        "total_return": float(portfolio_daily["equity"].iloc[-1] / config.initial_capital - 1),
        "annualized_return": annualized_return(portfolio_daily["equity"]),
        "annualized_volatility": annualized_volatility(portfolio_daily["daily_return"]),
        "sharpe": sharpe_ratio(portfolio_daily["daily_return"]),
        "max_drawdown": max_drawdown(portfolio_daily["equity"]),
        "avg_active_pairs": float(portfolio_daily["active_pairs"].mean()),
        "max_active_pairs": int(portfolio_daily["active_pairs"].max()),
        "avg_gross_exposure_pct": float(portfolio_daily["gross_exposure_pct"].mean()),
        "max_gross_exposure_pct": float(portfolio_daily["gross_exposure_pct"].max()),
        "active_day_pct": float(portfolio_daily["active_pairs"].gt(0).mean()),
    }
    return pd.DataFrame([summary])


def write_portfolio_outputs(
    backtest_comparison_csv: str | Path,
    output_dir: str | Path,
    config: PortfolioConfig = PortfolioConfig(),
) -> tuple[Path, Path, Path]:
    """Write selected allocations, portfolio daily, and portfolio summary."""
    output_dir = Path(output_dir)
    comparison = pd.read_csv(backtest_comparison_csv)
    selected = select_pairs(comparison, config)
    selected, _ = add_allocations(selected, output_dir, config)
    portfolio_daily = build_portfolio_daily(selected, output_dir, config)
    summary = summarize_portfolio(portfolio_daily, selected, config)

    allocations_path = output_dir / "portfolio_allocations.csv"
    daily_path = output_dir / "portfolio_daily.csv"
    summary_path = output_dir / "portfolio_summary.csv"

    selected.to_csv(allocations_path, index=False)
    portfolio_daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    return allocations_path, daily_path, summary_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    paths = write_portfolio_outputs(
        backtest_comparison_csv=base_dir / "outputs" / "backtest_comparison.csv",
        output_dir=base_dir / "outputs",
    )
    for path in paths:
        print(f"Wrote {path}")
