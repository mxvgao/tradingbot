"""Walk-forward portfolio validation for ETF pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_pair import (
    BacktestConfig,
    annualized_return,
    annualized_volatility,
    compute_walk_forward_signals,
    max_drawdown,
    run_backtest,
    sharpe_ratio,
)
from hmm_regime_filter import (
    HMMRegimeConfig,
    add_regime_allowed,
    fit_predict_hmm_regimes,
    infer_allowed_states_from_trades,
)
from scan_cointegration import scan_pairs


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 504
    validation_days: int = 126
    test_days: int = 126
    step_days: int = 126
    max_folds: int | None = None
    latest_folds_first: bool = False
    max_pairs: int = 5
    max_pairs_per_group: int = 2
    max_pairs_per_ticker: int = 1
    max_train_candidates: int = 15
    min_train_trades: int = 4
    min_train_trades_per_year: float = 2.0
    min_train_sharpe: float = 0.25
    min_train_win_rate: float = 0.50
    min_train_profit_factor: float = 1.10
    min_train_avg_net_pnl_bps: float = 0.0
    max_train_drawdown: float = -0.03
    min_validation_trades: int = 3
    min_validation_total_return: float = 0.0
    min_validation_avg_net_pnl_bps: float = 0.0
    max_validation_drawdown: float = -0.03
    max_validation_max_hold_exit_pct: float = 0.50
    use_hmm_filter: bool = True
    require_positive_train_return: bool = True
    initial_capital: float = 100_000
    total_gross_budget: float = 50_000
    formation_days: int = 252
    entry_z: float = 2.0
    exit_z: float = 0.5
    round_trip_cost_bps: float = 4.0
    max_holding_days: int | None = None
    stop_z: float | None = None
    strategy_grid: tuple[tuple[float, float, int | None, float | None], ...] = (
        (1.5, 0.5, 20, 3.0),
        (1.5, 0.5, 40, 3.0),
        (2.0, 0.5, 20, 3.0),
        (2.0, 0.5, 40, 3.0),
        (2.0, 1.0, 20, 3.0),
        (2.0, 1.0, 40, 3.0),
    )


def slice_prices(
    price_history: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return price history between two dates, inclusive."""
    prices = price_history.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    return prices[(prices["date"] >= start_date) & (prices["date"] <= end_date)].copy()


def add_missing_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """Ensure train/test summaries have the ranking fields we need."""
    defaults = {
        "completed_trades": 0,
        "total_return": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "total_pnl_dollars": 0.0,
        "exposure_pct": 0.0,
    }
    result = summary.copy()
    for column, default in defaults.items():
        if column not in result:
            result[column] = default
    return result


def backtest_pair_on_prices(
    price_history: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    config: WalkForwardConfig,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    gross_notional: float | None = None,
    formation_days: int | None = None,
    entry_z: float | None = None,
    exit_z: float | None = None,
    max_holding_days: int | None = None,
    stop_z: float | None = None,
    block_reentry_after_max_hold: bool = False,
    require_regime_allowed: bool = False,
    precomputed_signals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest one pair and optionally keep only a date range of signals."""
    backtest_config = BacktestConfig(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        formation_days=formation_days or config.formation_days,
        entry_z=entry_z if entry_z is not None else config.entry_z,
        exit_z=exit_z if exit_z is not None else config.exit_z,
        round_trip_cost_bps=config.round_trip_cost_bps,
        initial_capital=config.initial_capital,
        gross_notional_per_trade=gross_notional or config.total_gross_budget / config.max_pairs,
        max_holding_days=max_holding_days if max_holding_days is not None else config.max_holding_days,
        stop_z=stop_z if stop_z is not None else config.stop_z,
        block_reentry_after_max_hold=block_reentry_after_max_hold,
        require_regime_allowed=require_regime_allowed,
    )
    signals = (
        precomputed_signals.copy()
        if precomputed_signals is not None
        else compute_walk_forward_signals(price_history, backtest_config)
    )
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame(), add_missing_summary_columns(pd.DataFrame([{}]))

    signals["date"] = pd.to_datetime(signals["date"])
    if start_date is not None:
        signals = signals[signals["date"] >= start_date]
    if end_date is not None:
        signals = signals[signals["date"] <= end_date]
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame(), add_missing_summary_columns(pd.DataFrame([{}]))

    trades, daily, summary = run_backtest(signals, backtest_config)
    return trades, daily, add_missing_summary_columns(summary)


def build_hmm_filtered_signals(
    price_history: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    config: WalkForwardConfig,
    subtrain_end: pd.Timestamp,
    entry_z: float,
    exit_z: float,
    max_holding_days: int | None,
    stop_z: float | None,
) -> tuple[pd.DataFrame, set[int]]:
    """Fit HMM on subtrain signals and mark allowed regimes."""
    base_backtest_config = BacktestConfig(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        formation_days=config.formation_days,
        entry_z=entry_z,
        exit_z=exit_z,
        round_trip_cost_bps=config.round_trip_cost_bps,
        initial_capital=config.initial_capital,
        gross_notional_per_trade=config.total_gross_budget / config.max_pairs,
        max_holding_days=max_holding_days,
        stop_z=stop_z,
        block_reentry_after_max_hold=True,
    )
    signals = compute_walk_forward_signals(price_history, base_backtest_config)
    signals["date"] = pd.to_datetime(signals["date"])
    subtrain_signals = signals[signals["date"] <= subtrain_end]
    subtrain_trades, _, _ = run_backtest(subtrain_signals, base_backtest_config)
    regimes = fit_predict_hmm_regimes(
        signals,
        subtrain_end,
        HMMRegimeConfig(),
    )
    allowed_states = infer_allowed_states_from_trades(
        subtrain_trades,
        regimes,
        HMMRegimeConfig(),
    )
    return add_regime_allowed(signals, regimes, allowed_states), allowed_states


def score_train_pair(summary: pd.Series, coint_pvalue: float, adf_pvalue: float) -> float:
    """Blend train-period trading quality and statistical quality."""
    pvalue_bonus = max(0.0, -np.log10(max(coint_pvalue, 1e-12)) - 1)
    adf_bonus = max(0.0, -np.log10(max(adf_pvalue, 1e-12)) - 1)
    trade_bonus = min(float(summary.get("completed_trades", 0)), 10.0) / 10.0
    profit_factor_bonus = min(float(summary.get("profit_factor", 0.0)), 3.0) / 3.0
    drawdown_penalty = abs(min(float(summary.get("max_drawdown", 0.0)), 0.0)) * 20
    return (
        float(summary.get("sharpe", 0.0))
        + float(summary.get("total_return", 0.0)) * 100
        + trade_bonus
        + profit_factor_bonus
        + 0.25 * pvalue_bonus
        + 0.25 * adf_bonus
        - drawdown_penalty
    )


def score_validation_tradability(summary: pd.Series) -> float:
    """Score candidates by validation-period trading quality."""
    return (
        float(summary.get("validation_sharpe", 0.0))
        + float(summary.get("validation_total_return", 0.0)) * 100
        + float(summary.get("validation_avg_net_pnl_bps", 0.0)) / 100
        + min(float(summary.get("validation_completed_trades", 0.0)), 10.0) / 10
        - float(summary.get("validation_max_hold_exit_pct", 1.0)) * 0.5
        - abs(min(float(summary.get("validation_max_drawdown", 0.0)), 0.0)) * 10
    )


def train_trade_frequency(summary: pd.Series, train_days: int) -> float:
    """Estimate completed trades per train year."""
    years = train_days / 252
    if years <= 0:
        return 0.0
    return float(summary.get("completed_trades", 0)) / years


def add_train_edge_metrics(trades: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Add per-trade edge metrics used for train-window selection."""
    result = summary.copy()
    if trades.empty or "net_pnl_bps" not in trades:
        result["avg_net_pnl_bps"] = 0.0
        result["profit_factor"] = 0.0
        result["max_hold_exit_pct"] = 0.0
        return result

    gains = float(trades.loc[trades["net_pnl_bps"] > 0, "net_pnl_bps"].sum())
    losses = float(trades.loc[trades["net_pnl_bps"] < 0, "net_pnl_bps"].sum())
    if losses < 0:
        profit_factor = gains / abs(losses)
    else:
        profit_factor = np.inf if gains > 0 else 0.0

    result["avg_net_pnl_bps"] = float(trades["net_pnl_bps"].mean())
    result["profit_factor"] = float(profit_factor)
    if "exit_reason" in trades:
        result["max_hold_exit_pct"] = float(trades["exit_reason"].eq("max_holding_days").mean())
    else:
        result["max_hold_exit_pct"] = 0.0
    return result


def passes_train_edge_filter(
    summary: pd.Series,
    config: WalkForwardConfig,
    train_days: int,
) -> bool:
    """Decide whether a train-period pair/config has enough edge to test."""
    if summary.get("completed_trades", 0) < config.min_train_trades:
        return False
    if train_trade_frequency(summary, train_days) < config.min_train_trades_per_year:
        return False
    if config.require_positive_train_return and summary.get("total_return", 0.0) <= 0:
        return False
    if summary.get("sharpe", 0.0) < config.min_train_sharpe:
        return False
    if summary.get("win_rate", 0.0) < config.min_train_win_rate:
        return False
    if summary.get("profit_factor", 0.0) < config.min_train_profit_factor:
        return False
    if summary.get("avg_net_pnl_bps", 0.0) <= config.min_train_avg_net_pnl_bps:
        return False
    if summary.get("max_drawdown", 0.0) < config.max_train_drawdown:
        return False
    return True


def evaluate_train_strategy_grid(
    subtrain_prices: pd.DataFrame,
    validation_context_prices: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    subtrain_end: pd.Timestamp,
    ticker_a: str,
    ticker_b: str,
    config: WalkForwardConfig,
) -> dict[str, object] | None:
    """Find the best train-window rule that also passes validation."""
    candidates: list[dict[str, object]] = []
    for entry_z, exit_z, max_holding_days, stop_z in config.strategy_grid:
        precomputed_signals = None
        allowed_states: set[int] = set()
        require_regime_allowed = False
        if config.use_hmm_filter:
            precomputed_signals, allowed_states = build_hmm_filtered_signals(
                validation_context_prices,
                ticker_a,
                ticker_b,
                config,
                subtrain_end,
                entry_z,
                exit_z,
                max_holding_days,
                stop_z,
            )
            if not allowed_states:
                continue
            require_regime_allowed = True

        train_trades, _, train_summary = backtest_pair_on_prices(
            subtrain_prices,
            ticker_a,
            ticker_b,
            config,
            end_date=subtrain_end,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding_days=max_holding_days,
            stop_z=stop_z,
            block_reentry_after_max_hold=True,
            require_regime_allowed=require_regime_allowed,
            precomputed_signals=precomputed_signals,
        )
        train_summary = add_train_edge_metrics(train_trades, train_summary)
        train_row = train_summary.iloc[0].to_dict()
        train_series = pd.Series(train_row)
        subtrain_days = config.train_days - config.validation_days
        if not passes_train_edge_filter(train_series, config, subtrain_days):
            continue

        validation_trades, _, validation_summary = backtest_pair_on_prices(
            validation_context_prices,
            ticker_a,
            ticker_b,
            config,
            start_date=validation_start,
            end_date=validation_end,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding_days=max_holding_days,
            stop_z=stop_z,
            block_reentry_after_max_hold=True,
            require_regime_allowed=require_regime_allowed,
            precomputed_signals=precomputed_signals,
        )
        validation_summary = add_train_edge_metrics(
            validation_trades,
            validation_summary,
        )
        validation_row = validation_summary.iloc[0].to_dict()
        if validation_row.get("completed_trades", 0) < config.min_validation_trades:
            continue
        if validation_row.get("total_return", 0.0) < config.min_validation_total_return:
            continue
        if validation_row.get("avg_net_pnl_bps", 0.0) <= config.min_validation_avg_net_pnl_bps:
            continue
        if validation_row.get("max_drawdown", 0.0) < config.max_validation_drawdown:
            continue
        if validation_row.get("max_hold_exit_pct", 0.0) > config.max_validation_max_hold_exit_pct:
            continue

        train_row.update(
            {
                "selected_entry_z": entry_z,
                "selected_exit_z": exit_z,
                "selected_max_holding_days": max_holding_days,
                "selected_stop_z": stop_z,
                "selected_hmm_states": "|".join(str(state) for state in sorted(allowed_states)),
                "train_trades_per_year": train_trade_frequency(
                    train_series,
                    subtrain_days,
                ),
            }
        )
        train_row.update({f"validation_{key}": value for key, value in validation_row.items()})
        candidates.append(train_row)

    if not candidates:
        return None

    ranked = pd.DataFrame(candidates).sort_values(
        [
            "validation_sharpe",
            "validation_total_return",
            "sharpe",
            "profit_factor",
            "completed_trades",
        ],
        ascending=[False, False, False, False, False],
    )
    return ranked.iloc[0].to_dict()


def select_fold_pairs(
    train_scan: pd.DataFrame,
    subtrain_prices: pd.DataFrame,
    validation_context_prices: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    subtrain_end: pd.Timestamp,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    """Select pairs using only subtrain and validation windows."""
    passed = train_scan[train_scan["passes_coint_filter"]].copy()
    if passed.empty:
        return pd.DataFrame()
    passed = passed.head(config.max_train_candidates)

    rows: list[dict[str, object]] = []
    for row in passed.itertuples(index=False):
        summary_row = evaluate_train_strategy_grid(
            subtrain_prices,
            validation_context_prices,
            validation_start,
            validation_end,
            subtrain_end,
            row.ticker_a,
            row.ticker_b,
            config,
        )
        if summary_row is None:
            continue

        selected_row = row._asdict()
        selected_row.update({f"train_{key}": value for key, value in summary_row.items()})
        selected_row["selection_score"] = score_train_pair(
            pd.Series(summary_row),
            float(row.coint_pvalue),
            float(row.adf_pvalue),
        )
        selected_row["validation_tradability_score"] = score_validation_tradability(
            pd.Series(summary_row)
        )
        rows.append(selected_row)

    if not rows:
        return pd.DataFrame()

    ranked = pd.DataFrame(rows).sort_values(
        ["validation_tradability_score", "selection_score"],
        ascending=[False, False],
    )
    selected_rows = []
    group_counts: dict[str, int] = {}
    ticker_counts: dict[str, int] = {}
    for row in ranked.itertuples(index=False):
        group = str(getattr(row, "group", "unknown"))
        if group_counts.get(group, 0) >= config.max_pairs_per_group:
            continue
        ticker_a = str(getattr(row, "ticker_a")).upper()
        ticker_b = str(getattr(row, "ticker_b")).upper()
        if ticker_counts.get(ticker_a, 0) >= config.max_pairs_per_ticker:
            continue
        if ticker_counts.get(ticker_b, 0) >= config.max_pairs_per_ticker:
            continue
        selected_rows.append(row._asdict())
        group_counts[group] = group_counts.get(group, 0) + 1
        ticker_counts[ticker_a] = ticker_counts.get(ticker_a, 0) + 1
        ticker_counts[ticker_b] = ticker_counts.get(ticker_b, 0) + 1
        if len(selected_rows) >= config.max_pairs:
            break

    return pd.DataFrame(selected_rows)


def summarize_daily_portfolio(
    daily: pd.DataFrame,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    """Create one row for the combined walk-forward portfolio."""
    if daily.empty:
        return pd.DataFrame(
            [{"initial_capital": config.initial_capital, "selected_pairs": 0}]
        )

    return pd.DataFrame(
        [
            {
                "initial_capital": config.initial_capital,
                "selected_pairs": int(daily["active_pairs_possible"].max()),
                "total_gross_budget": config.total_gross_budget,
                "total_pnl_dollars": float(daily["cumulative_pnl"].iloc[-1]),
                "total_return": float(daily["equity"].iloc[-1] / config.initial_capital - 1),
                "annualized_return": annualized_return(daily["equity"]),
                "annualized_volatility": annualized_volatility(daily["daily_return"]),
                "sharpe": sharpe_ratio(daily["daily_return"]),
                "max_drawdown": max_drawdown(daily["equity"]),
                "avg_active_pairs": float(daily["active_pairs"].mean()),
                "max_active_pairs": int(daily["active_pairs"].max()),
                "active_day_pct": float(daily["active_pairs"].gt(0).mean()),
                "avg_gross_exposure_pct": float(daily["gross_exposure_pct"].mean()),
                "max_gross_exposure_pct": float(daily["gross_exposure_pct"].max()),
            }
        ]
    )


def run_walk_forward(
    price_history: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    config: WalkForwardConfig = WalkForwardConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run rolling train/select/test portfolio validation."""
    prices = price_history.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    dates = pd.Index(sorted(prices["date"].dropna().unique()))

    fold_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    pair_result_rows: list[dict[str, object]] = []
    portfolio_daily_frames: list[pd.DataFrame] = []
    portfolio_equity_offset = 0.0

    last_start = len(dates) - config.train_days - config.test_days
    if last_start < 0:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            summarize_daily_portfolio(pd.DataFrame(), config),
        )

    start_indices = list(range(0, last_start + 1, config.step_days))
    if config.latest_folds_first:
        start_indices = list(reversed(start_indices))

    for fold_id, start_idx in enumerate(start_indices, start=1):
        if config.max_folds is not None and fold_id > config.max_folds:
            break
        train_start = pd.Timestamp(dates[start_idx])
        train_end = pd.Timestamp(dates[start_idx + config.train_days - 1])
        validation_start = pd.Timestamp(dates[start_idx + config.train_days - config.validation_days])
        validation_end = train_end
        test_start = pd.Timestamp(dates[start_idx + config.train_days])
        test_end = pd.Timestamp(dates[start_idx + config.train_days + config.test_days - 1])

        train_prices = slice_prices(prices, train_start, train_end)
        subtrain_prices = slice_prices(
            prices,
            train_start,
            pd.Timestamp(dates[start_idx + config.train_days - config.validation_days - 1]),
        )
        validation_context_prices = slice_prices(prices, train_start, validation_end)
        context_prices = slice_prices(prices, train_start, test_end)
        train_scan = scan_pairs(candidate_pairs, subtrain_prices)
        selected = select_fold_pairs(
            train_scan,
            subtrain_prices,
            validation_context_prices,
            validation_start,
            validation_end,
            pd.Timestamp(dates[start_idx + config.train_days - config.validation_days - 1]),
            config,
        )

        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_start": train_start.date(),
                "subtrain_end": pd.Timestamp(dates[start_idx + config.train_days - config.validation_days - 1]).date(),
                "validation_start": validation_start.date(),
                "validation_end": validation_end.date(),
                "train_end": train_end.date(),
                "test_start": test_start.date(),
                "test_end": test_end.date(),
                "train_pairs_tested": int(train_scan.shape[0]),
                "train_pairs_passed_coint": int(train_scan.get("passes_coint_filter", pd.Series(dtype=bool)).sum()),
                "selected_pairs": int(selected.shape[0]),
            }
        )

        if selected.empty:
            continue

        selected = selected.copy()
        selected["fold_id"] = fold_id
        selected["test_start"] = test_start.date()
        selected["test_end"] = test_end.date()
        selected_frames.append(selected)

        per_pair_gross = config.total_gross_budget / len(selected)
        fold_pair_daily = []
        subtrain_end = pd.Timestamp(
            dates[start_idx + config.train_days - config.validation_days - 1]
        )
        for row in selected.itertuples(index=False):
            test_signals = None
            if config.use_hmm_filter:
                test_signals, _ = build_hmm_filtered_signals(
                    context_prices,
                    row.ticker_a,
                    row.ticker_b,
                    config,
                    subtrain_end,
                    entry_z=float(row.train_selected_entry_z),
                    exit_z=float(row.train_selected_exit_z),
                    max_holding_days=(
                        None
                        if pd.isna(row.train_selected_max_holding_days)
                        else int(row.train_selected_max_holding_days)
                    ),
                    stop_z=(
                        None
                        if pd.isna(row.train_selected_stop_z)
                        else float(row.train_selected_stop_z)
                    ),
                )
            trades, daily, summary = backtest_pair_on_prices(
                context_prices,
                row.ticker_a,
                row.ticker_b,
                config,
                start_date=test_start,
                end_date=test_end,
                gross_notional=per_pair_gross,
                entry_z=float(row.train_selected_entry_z),
                exit_z=float(row.train_selected_exit_z),
                max_holding_days=(
                    None
                    if pd.isna(row.train_selected_max_holding_days)
                    else int(row.train_selected_max_holding_days)
                ),
                stop_z=(
                    None
                    if pd.isna(row.train_selected_stop_z)
                    else float(row.train_selected_stop_z)
                ),
                block_reentry_after_max_hold=True,
                require_regime_allowed=config.use_hmm_filter,
                precomputed_signals=test_signals,
            )
            summary_row = summary.iloc[0].to_dict()
            summary_row.update(
                {
                    "fold_id": fold_id,
                    "ticker_a": row.ticker_a,
                    "ticker_b": row.ticker_b,
                    "group": row.group,
                    "subgroup": row.subgroup,
                    "selection_score": row.selection_score,
                    "allocated_gross_notional": per_pair_gross,
                    "entry_z": row.train_selected_entry_z,
                    "exit_z": row.train_selected_exit_z,
                    "max_holding_days": row.train_selected_max_holding_days,
                    "stop_z": row.train_selected_stop_z,
                    "hmm_states": getattr(row, "train_selected_hmm_states", ""),
                    "test_start": test_start.date(),
                    "test_end": test_end.date(),
                }
            )
            pair_result_rows.append(summary_row)

            if not daily.empty:
                pair_daily = daily.copy()
                pair_daily["date"] = pd.to_datetime(pair_daily["date"])
                pair_daily["fold_id"] = fold_id
                pair_daily["pair"] = f"{row.ticker_a}_{row.ticker_b}"
                fold_pair_daily.append(pair_daily)

        if not fold_pair_daily:
            continue

        pair_daily_all = pd.concat(fold_pair_daily, ignore_index=True)
        fold_daily = (
            pair_daily_all.groupby("date", as_index=False)
            .agg(
                net_daily_pnl=("net_daily_pnl", "sum"),
                gross_exposure=("gross_exposure", "sum"),
                active_pairs=("position", lambda x: int((x != 0).sum())),
            )
            .sort_values("date")
        )
        fold_daily["fold_id"] = fold_id
        fold_daily["active_pairs_possible"] = int(selected.shape[0])
        fold_daily["cumulative_pnl"] = portfolio_equity_offset + fold_daily["net_daily_pnl"].cumsum()
        fold_daily["equity"] = config.initial_capital + fold_daily["cumulative_pnl"]
        fold_daily["daily_return"] = fold_daily["net_daily_pnl"] / config.initial_capital
        fold_daily["gross_exposure_pct"] = fold_daily["gross_exposure"] / config.initial_capital
        portfolio_equity_offset = float(fold_daily["cumulative_pnl"].iloc[-1])
        portfolio_daily_frames.append(fold_daily)

    folds = pd.DataFrame(fold_rows)
    selected_pairs = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    pair_results = pd.DataFrame(pair_result_rows)
    portfolio_daily = (
        pd.concat(portfolio_daily_frames, ignore_index=True)
        if portfolio_daily_frames
        else pd.DataFrame()
    )
    summary = summarize_daily_portfolio(portfolio_daily, config)
    if not folds.empty:
        summary["folds"] = int(folds.shape[0])
        summary["folds_with_selected_pairs"] = int(folds["selected_pairs"].gt(0).sum())
        summary["avg_selected_pairs_per_fold"] = float(folds["selected_pairs"].mean())

    return folds, selected_pairs, pair_results, portfolio_daily, summary


def write_walk_forward_outputs(
    price_history_csv: str | Path,
    candidate_pairs_csv: str | Path,
    output_dir: str | Path,
    config: WalkForwardConfig = WalkForwardConfig(),
) -> tuple[Path, Path, Path, Path, Path]:
    """Run walk-forward validation and write the main CSV outputs."""
    price_history = pd.read_csv(price_history_csv)
    candidate_pairs = pd.read_csv(candidate_pairs_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds, selected_pairs, pair_results, portfolio_daily, summary = run_walk_forward(
        price_history,
        candidate_pairs,
        config,
    )

    folds_path = output_dir / "walk_forward_folds.csv"
    selected_path = output_dir / "walk_forward_selected_pairs.csv"
    pair_results_path = output_dir / "walk_forward_pair_results.csv"
    daily_path = output_dir / "walk_forward_portfolio_daily.csv"
    summary_path = output_dir / "walk_forward_summary.csv"

    folds.to_csv(folds_path, index=False)
    selected_pairs.to_csv(selected_path, index=False)
    pair_results.to_csv(pair_results_path, index=False)
    portfolio_daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)

    return folds_path, selected_path, pair_results_path, daily_path, summary_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    paths = write_walk_forward_outputs(
        price_history_csv=base_dir / "data" / "etf_price_history.csv",
        candidate_pairs_csv=base_dir / "data" / "candidate_pairs.csv",
        output_dir=base_dir / "outputs",
    )
    for path in paths:
        print(f"Wrote {path}")
