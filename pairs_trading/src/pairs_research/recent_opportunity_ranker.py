"""Rank pairs by recent train/validation tradability."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from scan_cointegration import scan_pairs
from walk_forward_portfolio import (
    WalkForwardConfig,
    evaluate_train_strategy_grid,
    slice_prices,
)


def load_price_history(path: str | Path) -> pd.DataFrame:
    """Load price history with compact dtypes for larger 10-year files."""
    return pd.read_csv(
        path,
        usecols=["date", "ticker", "adj_close", "volume"],
        dtype={
            "date": "string",
            "ticker": "category",
            "adj_close": "float32",
            "volume": "float32",
        },
    )


def recent_window_dates(
    price_history: pd.DataFrame,
    config: WalkForwardConfig,
    latest: bool = True,
) -> dict[str, pd.Timestamp]:
    """Return subtrain/validation/test dates for one recent opportunity window."""
    prices = price_history.copy(deep=False)
    prices["date"] = pd.to_datetime(prices["date"])
    dates = pd.Index(sorted(prices["date"].dropna().unique()))
    last_start = len(dates) - config.train_days - config.test_days
    if last_start < 0:
        raise ValueError("Not enough price history for the configured windows")

    start_idx = last_start if latest else 0
    subtrain_end_idx = start_idx + config.train_days - config.validation_days - 1
    return {
        "train_start": pd.Timestamp(dates[start_idx]),
        "subtrain_end": pd.Timestamp(dates[subtrain_end_idx]),
        "validation_start": pd.Timestamp(dates[subtrain_end_idx + 1]),
        "validation_end": pd.Timestamp(dates[start_idx + config.train_days - 1]),
        "test_start": pd.Timestamp(dates[start_idx + config.train_days]),
        "test_end": pd.Timestamp(dates[start_idx + config.train_days + config.test_days - 1]),
    }


def rank_recent_opportunities(
    price_history: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    config: WalkForwardConfig = WalkForwardConfig(use_hmm_filter=False),
    latest: bool = True,
    max_candidates: int = 75,
) -> pd.DataFrame:
    """Rank recent pair opportunities by validation-period trading quality."""
    dates = recent_window_dates(price_history, config, latest=latest)
    subtrain_prices = slice_prices(
        price_history,
        dates["train_start"],
        dates["subtrain_end"],
    )
    validation_context_prices = slice_prices(
        price_history,
        dates["train_start"],
        dates["validation_end"],
    )
    train_scan = scan_pairs(candidate_pairs, subtrain_prices)
    passed = train_scan[train_scan["passes_coint_filter"]].head(max_candidates)

    rows: list[dict[str, object]] = []
    eval_config = replace(config, max_train_candidates=max_candidates)
    for row in passed.itertuples(index=False):
        metrics = evaluate_train_strategy_grid(
            subtrain_prices,
            validation_context_prices,
            dates["validation_start"],
            dates["validation_end"],
            dates["subtrain_end"],
            row.ticker_a,
            row.ticker_b,
            eval_config,
        )
        if metrics is None:
            continue

        result = row._asdict()
        result.update({f"recent_{key}": value for key, value in metrics.items()})
        result.update({key: value.date() for key, value in dates.items()})
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    ranked = pd.DataFrame(rows)
    ranked["recent_tradability_score"] = (
        ranked["recent_validation_sharpe"].fillna(0)
        + ranked["recent_validation_total_return"].fillna(0) * 100
        + ranked["recent_validation_avg_net_pnl_bps"].fillna(0) / 100
        + ranked["recent_validation_completed_trades"].fillna(0).clip(upper=10) / 10
        - ranked["recent_validation_max_hold_exit_pct"].fillna(1) * 0.5
        - ranked["recent_validation_max_drawdown"].fillna(0).abs() * 10
    )
    return ranked.sort_values(
        [
            "recent_tradability_score",
            "recent_validation_total_return",
            "recent_validation_sharpe",
            "recent_validation_completed_trades",
        ],
        ascending=[False, False, False, False],
    )


def write_recent_opportunity_rankings(
    price_history_csv: str | Path,
    candidate_pairs_csv: str | Path,
    output_csv: str | Path,
    config: WalkForwardConfig = WalkForwardConfig(use_hmm_filter=False),
    latest: bool = True,
    max_candidates: int = 75,
) -> Path:
    """Write recent opportunity rankings to CSV."""
    price_history = load_price_history(price_history_csv)
    candidate_pairs = pd.read_csv(candidate_pairs_csv)
    rankings = rank_recent_opportunities(
        price_history,
        candidate_pairs,
        config=config,
        latest=latest,
        max_candidates=max_candidates,
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    written_path = write_recent_opportunity_rankings(
        price_history_csv=base_dir / "data" / "etf_price_history.csv",
        candidate_pairs_csv=base_dir / "data" / "candidate_pairs.csv",
        output_csv=base_dir / "outputs" / "recent_opportunity_rankings.csv",
    )
    print(f"Wrote {written_path}")
