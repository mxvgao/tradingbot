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
    min_related_return_corr: float = 0.80
    min_same_group_return_corr: float = 0.90
    min_cross_theme_return_corr: float = 0.65
    max_pairs_per_cross_theme_relation: int = 250


PRICE_COLUMNS = {"date", "ticker", "adj_close", "volume"}
UNIVERSE_COLUMNS = {"ticker", "name", "asset_class", "group", "subgroup"}

RELATED_SUBGROUPS = {
    ("us_equity", "large_blend"): {
        "total_market",
        "large_equal_weight",
        "large_value",
        "large_growth",
    },
    ("us_equity", "total_market"): {"large_blend"},
    ("us_equity", "large_growth"): {"large_blend", "growth"},
    ("us_equity", "large_value"): {"large_blend", "value"},
    ("us_equity", "small_blend"): {"small_value", "small_growth"},
    ("factor", "growth"): {"large_growth", "small_growth", "quality", "momentum"},
    ("factor", "value"): {"large_value", "small_value", "dividend"},
    ("factor", "momentum"): {"quality", "growth"},
    ("factor", "quality"): {"momentum", "growth", "dividend"},
    ("factor", "minimum_volatility"): {"dividend", "quality"},
    ("us_bonds", "treasury"): {
        "short_treasury",
        "intermediate_treasury",
        "long_treasury",
        "extended_treasury",
    },
    ("us_bonds", "long_treasury"): {"extended_treasury", "treasury"},
    ("us_bonds", "short_treasury"): {"ultra_short_treasury", "treasury"},
    ("credit", "investment_grade"): {"short_investment_grade", "broad_bond"},
    ("bonds", "broad_bond"): {"investment_grade", "treasury"},
    ("inflation_linked", "tips"): {"short_tips", "treasury"},
    ("international_equity", "developed_ex_us"): {"global", "ex_us"},
    ("international_equity", "emerging_markets"): {"global", "china", "india"},
    ("global_equity", "global"): {"developed_ex_us", "emerging_markets", "ex_us"},
    ("metals", "gold"): {"silver", "metals"},
    ("metals", "silver"): {"gold", "metals"},
    ("energy", "energy_commodity"): {"energy"},
    ("sector", "technology"): {"semiconductors"},
    ("industry", "semiconductors"): {"technology"},
    ("sector", "financials"): {"banks"},
    ("industry", "banks"): {"financials"},
    ("sector", "healthcare"): {"biotechnology"},
    ("industry", "biotechnology"): {"healthcare"},
}

CROSS_THEME_RELATIONS = [
    ("sector_industry", ("sector", "technology"), ("industry", "semiconductors")),
    ("sector_industry", ("sector", "technology"), ("industry", "software")),
    ("sector_industry", ("sector", "technology"), ("industry", "internet")),
    ("sector_industry", ("sector", "technology"), ("industry", "cloud")),
    ("sector_industry", ("sector", "technology"), ("industry", "cybersecurity")),
    ("sector_industry", ("sector", "technology"), ("industry", "artificial_intelligence")),
    ("sector_industry", ("sector", "financials"), ("industry", "banks")),
    ("sector_industry", ("sector", "financials"), ("industry", "insurance")),
    ("sector_industry", ("sector", "healthcare"), ("industry", "biotechnology")),
    ("sector_industry", ("sector", "healthcare"), ("industry", "pharmaceuticals")),
    ("sector_industry", ("sector", "healthcare"), ("industry", "medical_devices")),
    ("sector_industry", ("sector", "energy"), ("industry", "oil_gas_equity")),
    ("sector_industry", ("sector", "industrials"), ("industry", "aerospace_defense")),
    ("sector_industry", ("sector", "materials"), ("industry", "mining")),
    ("sector_industry", ("sector", "consumer_discretionary"), ("industry", "retail")),
    ("commodity_producer", ("metals", "gold"), ("industry", "mining")),
    ("commodity_producer", ("metals", "silver"), ("industry", "mining")),
    ("commodity_producer", ("energy", "energy_commodity"), ("industry", "oil_gas_equity")),
    ("factor_broad", ("factor", "momentum"), ("us_equity", "large_blend")),
    ("factor_broad", ("factor", "quality"), ("us_equity", "large_blend")),
    ("factor_broad", ("factor", "value"), ("us_equity", "large_blend")),
    ("factor_broad", ("factor", "growth"), ("us_equity", "large_blend")),
    ("factor_broad", ("factor", "minimum_volatility"), ("us_equity", "large_blend")),
    ("factor_broad", ("factor", "dividend"), ("us_equity", "large_blend")),
    ("country_region", ("single_country", "china"), ("international_equity", "emerging_markets")),
    ("country_region", ("single_country", "india"), ("international_equity", "emerging_markets")),
    ("country_region", ("single_country", "brazil"), ("international_equity", "emerging_markets")),
    ("country_region", ("single_country", "taiwan"), ("international_equity", "emerging_markets")),
    ("country_region", ("single_country", "south_korea"), ("international_equity", "emerging_markets")),
    ("country_region", ("single_country", "japan"), ("international_equity", "developed_ex_us")),
    ("country_region", ("single_country", "germany"), ("international_equity", "developed_ex_us")),
    ("country_region", ("single_country", "united_kingdom"), ("international_equity", "developed_ex_us")),
    ("credit_equity", ("credit", "high_yield"), ("us_equity", "large_blend")),
    ("credit_equity", ("credit", "high_yield"), ("sector", "financials")),
]


def prepare_price_history(price_history: pd.DataFrame) -> pd.DataFrame:
    """Normalize price history to the format used by the filters."""
    missing_columns = PRICE_COLUMNS.difference(price_history.columns)
    if missing_columns:
        raise ValueError(f"Missing price history columns: {sorted(missing_columns)}")

    prices = price_history.copy(deep=False)
    prices["date"] = pd.to_datetime(prices["date"])
    if not prices["ticker"].is_monotonic_increasing:
        prices["ticker"] = prices["ticker"].astype("category")
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


def build_price_matrix(price_history: pd.DataFrame) -> pd.DataFrame:
    """Build a date-indexed adjusted-close matrix."""
    prices = prepare_price_history(price_history)
    return prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def compute_return_corr(price_matrix: pd.DataFrame, ticker_a: str, ticker_b: str) -> float:
    """Compute daily return correlation for two tickers."""
    if ticker_a not in price_matrix or ticker_b not in price_matrix:
        return float("nan")
    pair_prices = price_matrix[[ticker_a, ticker_b]].dropna()
    if pair_prices.shape[0] < 2:
        return float("nan")
    returns = pair_prices.pct_change().dropna()
    return float(returns[ticker_a].corr(returns[ticker_b]))


def add_pair_row(
    rows: list[dict[str, object]],
    seen_pairs: set[tuple[str, str]],
    price_matrix: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    asset_class: str,
    group: str,
    subgroup: str,
    candidate_tier: str,
    config: FilterConfig,
    min_corr: float | None = None,
) -> None:
    """Add a candidate pair row if it passes overlap and optional correlation checks."""
    ordered_pair = tuple(sorted((ticker_a.upper(), ticker_b.upper())))
    if ordered_pair in seen_pairs:
        return

    pair_prices = price_matrix[list(ordered_pair)].dropna()
    overlap_days = int(pair_prices.shape[0])
    return_corr = compute_return_corr(price_matrix, *ordered_pair)
    passes_corr = min_corr is None or (pd.notna(return_corr) and return_corr >= min_corr)
    passes_pair_filter = overlap_days >= config.min_pair_overlap_days and passes_corr

    if not passes_pair_filter:
        return

    seen_pairs.add(ordered_pair)
    rows.append(
        {
            "ticker_a": ordered_pair[0],
            "ticker_b": ordered_pair[1],
            "asset_class": asset_class,
            "group": group,
            "subgroup": subgroup,
            "candidate_tier": candidate_tier,
            "overlap_days": overlap_days,
            "return_corr": return_corr,
            "passes_pair_filter": True,
        }
    )


def get_group_tickers(
    liquid: pd.DataFrame,
    group: str,
    subgroup: str,
) -> list[str]:
    """Return liquid tickers in a group/subgroup bucket."""
    rows = liquid[(liquid["group"].eq(group)) & (liquid["subgroup"].eq(subgroup))]
    return sorted(rows["ticker"].unique())


def add_cross_theme_pairs(
    rows: list[dict[str, object]],
    seen_pairs: set[tuple[str, str]],
    liquid: pd.DataFrame,
    price_matrix: pd.DataFrame,
    config: FilterConfig,
) -> None:
    """Add economically related cross-theme pairs with return-correlation gates."""
    for relation_name, left_bucket, right_bucket in CROSS_THEME_RELATIONS:
        left_tickers = get_group_tickers(liquid, *left_bucket)
        right_tickers = get_group_tickers(liquid, *right_bucket)
        candidates: list[tuple[float, str, str]] = []
        for ticker_a in left_tickers:
            for ticker_b in right_tickers:
                if ticker_a == ticker_b:
                    continue
                ordered_pair = tuple(sorted((ticker_a, ticker_b)))
                if ordered_pair in seen_pairs:
                    continue
                corr = compute_return_corr(price_matrix, *ordered_pair)
                if pd.notna(corr) and corr >= config.min_cross_theme_return_corr:
                    candidates.append((corr, ordered_pair[0], ordered_pair[1]))

        for _, ticker_a, ticker_b in sorted(candidates, reverse=True)[
            : config.max_pairs_per_cross_theme_relation
        ]:
            add_pair_row(
                rows=rows,
                seen_pairs=seen_pairs,
                price_matrix=price_matrix,
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                asset_class="cross_asset",
                group=f"{left_bucket[0]}__{right_bucket[0]}",
                subgroup=f"{left_bucket[1]}__{right_bucket[1]}",
                candidate_tier=relation_name,
                config=config,
                min_corr=config.min_cross_theme_return_corr,
            )


def generate_candidate_pairs(
    filtered_universe: pd.DataFrame,
    price_history: pd.DataFrame,
    config: FilterConfig = FilterConfig(),
) -> pd.DataFrame:
    """Create same-subgroup candidate pairs with enough overlapping history."""
    liquid = filtered_universe[filtered_universe["passes_etf_filter"]].copy()
    price_matrix = build_price_matrix(price_history)
    rows: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for (_, _, subgroup), group_rows in liquid.groupby(["asset_class", "group", "subgroup"]):
        tickers = sorted(group_rows["ticker"].unique())
        for ticker_a, ticker_b in combinations(tickers, 2):
            add_pair_row(
                rows=rows,
                seen_pairs=seen_pairs,
                price_matrix=price_matrix,
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                asset_class=group_rows["asset_class"].iloc[0],
                group=group_rows["group"].iloc[0],
                subgroup=subgroup,
                candidate_tier="same_subgroup",
                config=config,
            )

    for (asset_class, group), group_rows in liquid.groupby(["asset_class", "group"]):
        subgroup_map = {
            subgroup: sorted(rows_["ticker"].unique())
            for subgroup, rows_ in group_rows.groupby("subgroup")
        }
        for subgroup, tickers in subgroup_map.items():
            related = RELATED_SUBGROUPS.get((group, subgroup), set())
            for related_subgroup in related:
                if related_subgroup not in subgroup_map:
                    continue
                for ticker_a in tickers:
                    for ticker_b in subgroup_map[related_subgroup]:
                        add_pair_row(
                            rows=rows,
                            seen_pairs=seen_pairs,
                            price_matrix=price_matrix,
                            ticker_a=ticker_a,
                            ticker_b=ticker_b,
                            asset_class=asset_class,
                            group=group,
                            subgroup=f"{subgroup}__{related_subgroup}",
                            candidate_tier="related_subgroup",
                            config=config,
                            min_corr=config.min_related_return_corr,
                        )

        group_tickers = sorted(group_rows["ticker"].unique())
        for ticker_a, ticker_b in combinations(group_tickers, 2):
            add_pair_row(
                rows=rows,
                seen_pairs=seen_pairs,
                price_matrix=price_matrix,
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                asset_class=asset_class,
                group=group,
                subgroup="same_group_corr",
                candidate_tier="same_group_corr",
                config=config,
                min_corr=config.min_same_group_return_corr,
            )

    add_cross_theme_pairs(
        rows=rows,
        seen_pairs=seen_pairs,
        liquid=liquid,
        price_matrix=price_matrix,
        config=config,
    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker_a",
                "ticker_b",
                "asset_class",
                "group",
                "subgroup",
                "candidate_tier",
                "overlap_days",
                "return_corr",
                "passes_pair_filter",
            ]
        )

    pairs = pd.DataFrame(rows)
    return pairs.sort_values(
        [
            "passes_pair_filter",
            "candidate_tier",
            "asset_class",
            "group",
            "subgroup",
            "return_corr",
            "ticker_a",
            "ticker_b",
        ],
        ascending=[False, True, True, True, True, False, True, True],
    )


def write_filtered_outputs(
    universe_csv: str | Path,
    price_history_csv: str | Path,
    output_dir: str | Path,
    config: FilterConfig = FilterConfig(),
) -> tuple[Path, Path]:
    """Write liquid ETF and same-subgroup candidate pair CSVs."""
    universe = pd.read_csv(universe_csv)
    price_history = pd.read_csv(
        price_history_csv,
        usecols=["date", "ticker", "adj_close", "volume"],
        dtype={
            "date": "string",
            "ticker": "category",
            "adj_close": "float32",
            "volume": "float32",
        },
    )
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
