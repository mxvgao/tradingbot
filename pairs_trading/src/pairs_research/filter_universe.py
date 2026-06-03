"""Simple ETF and pair filters for the first cointegration scan."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FilterConfig:
    min_history_days: int = 756
    min_median_dollar_volume_60d: float = 10_000_000
    min_median_price_60d: float = 5
    max_missing_price_pct: float = 0.02
    min_pair_overlap_days: int = 756


PRICE_COLUMNS = {"date", "ticker", "adj_close", "volume"}
UNIVERSE_COLUMNS = {"ticker", "name", "asset_class", "group", "subgroup"}


def prepare_price_history(price_history: pd.DataFrame) -> pd.DataFrame:
    """Normalize price history to the format used by the filters."""
    missing_columns = PRICE_COLUMNS.difference(price_history.columns)
    if missing_columns:
        raise ValueError(f"Missing price history columns: {sorted(missing_columns)}")

    prices = price_history.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].str.upper()
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    prices["dollar_volume"] = prices["adj_close"] * prices["volume"]
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_etf_metrics(price_history: pd.DataFrame) -> pd.DataFrame:
    """Compute the simple ETF-level metrics used for the first filter."""
    prices = prepare_price_history(price_history)
    rows: list[dict[str, object]] = []

    for ticker, ticker_prices in prices.groupby("ticker", sort=True):
        ticker_prices = ticker_prices.sort_values("date")
        valid_prices = ticker_prices["adj_close"].notna()
        trailing = ticker_prices.tail(60)

        rows.append(
            {
                "ticker": ticker,
                "start_date": ticker_prices["date"].min().date(),
                "end_date": ticker_prices["date"].max().date(),
                "history_days": int(valid_prices.sum()),
                "median_dollar_volume_60d": trailing["dollar_volume"].median(),
                "median_price_60d": trailing["adj_close"].median(),
                "missing_price_pct": float(ticker_prices["adj_close"].isna().mean()),
            }
        )

    return pd.DataFrame(rows)


def filter_liquid_etfs(
    universe: pd.DataFrame,
    price_history: pd.DataFrame,
    config: FilterConfig = FilterConfig(),
) -> pd.DataFrame:
    """Join universe metadata to price metrics and apply ETF-level filters."""
    missing_columns = UNIVERSE_COLUMNS.difference(universe.columns)
    if missing_columns:
        raise ValueError(f"Missing universe columns: {sorted(missing_columns)}")

    universe = universe.copy()
    universe["ticker"] = universe["ticker"].str.upper()
    metrics = compute_etf_metrics(price_history)
    filtered = universe.merge(metrics, on="ticker", how="left")

    passes_filter = (
        filtered["history_days"].ge(config.min_history_days)
        & filtered["median_dollar_volume_60d"].ge(config.min_median_dollar_volume_60d)
        & filtered["median_price_60d"].ge(config.min_median_price_60d)
        & filtered["missing_price_pct"].le(config.max_missing_price_pct)
        & ~filtered.get("leveraged_or_inverse", False).astype(bool)
    )

    filtered["passes_etf_filter"] = passes_filter.fillna(False)
    return filtered.sort_values(["passes_etf_filter", "asset_class", "group", "subgroup", "ticker"], ascending=[False, True, True, True, True])


def compute_pair_overlap(price_history: pd.DataFrame, ticker_a: str, ticker_b: str) -> int:
    """Count dates where both tickers have valid adjusted close prices."""
    prices = prepare_price_history(price_history)
    pair_prices = prices[prices["ticker"].isin([ticker_a.upper(), ticker_b.upper()])]
    wide_prices = pair_prices.pivot(index="date", columns="ticker", values="adj_close")

    if ticker_a.upper() not in wide_prices or ticker_b.upper() not in wide_prices:
        return 0

    return int(wide_prices[[ticker_a.upper(), ticker_b.upper()]].dropna().shape[0])


def generate_candidate_pairs(
    filtered_universe: pd.DataFrame,
    price_history: pd.DataFrame,
    config: FilterConfig = FilterConfig(),
) -> pd.DataFrame:
    """Create same-subgroup candidate pairs with enough overlapping history."""
    liquid = filtered_universe[filtered_universe["passes_etf_filter"]].copy()
    prices = prepare_price_history(price_history)
    rows: list[dict[str, object]] = []

    for (_, _, subgroup), group_rows in liquid.groupby(["asset_class", "group", "subgroup"]):
        tickers = sorted(group_rows["ticker"].unique())
        for ticker_a, ticker_b in combinations(tickers, 2):
            overlap_days = compute_pair_overlap(prices, ticker_a, ticker_b)
            rows.append(
                {
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "asset_class": group_rows["asset_class"].iloc[0],
                    "group": group_rows["group"].iloc[0],
                    "subgroup": subgroup,
                    "overlap_days": overlap_days,
                    "passes_pair_filter": overlap_days >= config.min_pair_overlap_days,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker_a",
                "ticker_b",
                "asset_class",
                "group",
                "subgroup",
                "overlap_days",
                "passes_pair_filter",
            ]
        )

    pairs = pd.DataFrame(rows)
    return pairs.sort_values(["passes_pair_filter", "asset_class", "group", "subgroup", "ticker_a", "ticker_b"], ascending=[False, True, True, True, True, True])


def write_filtered_outputs(
    universe_csv: str | Path,
    price_history_csv: str | Path,
    output_dir: str | Path,
    config: FilterConfig = FilterConfig(),
) -> tuple[Path, Path]:
    """Write liquid ETF and same-subgroup candidate pair CSVs."""
    universe = pd.read_csv(universe_csv)
    price_history = pd.read_csv(price_history_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_universe = filter_liquid_etfs(universe, price_history, config)
    candidate_pairs = generate_candidate_pairs(filtered_universe, price_history, config)

    liquid_path = output_dir / "etf_universe_liquid.csv"
    pairs_path = output_dir / "candidate_pairs.csv"
    filtered_universe.to_csv(liquid_path, index=False)
    candidate_pairs.to_csv(pairs_path, index=False)
    return liquid_path, pairs_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    price_history_path = base_dir / "data" / "etf_price_history.csv"

    if not price_history_path.exists():
        raise SystemExit(
            "Missing data/etf_price_history.csv. Expected columns: "
            "date,ticker,adj_close,volume"
        )

    liquid_csv, pairs_csv = write_filtered_outputs(
        universe_csv=base_dir / "data" / "etf_universe_seed.csv",
        price_history_csv=price_history_path,
        output_dir=base_dir / "data",
    )
    print(f"Wrote {liquid_csv}")
    print(f"Wrote {pairs_csv}")
