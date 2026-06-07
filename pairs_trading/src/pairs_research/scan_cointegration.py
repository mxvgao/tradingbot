"""Run a first-pass cointegration screen on candidate ETF pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class CointegrationConfig:
    max_coint_pvalue: float = 0.05
    max_adf_pvalue: float = 0.05
    min_half_life_days: float = 1
    max_half_life_days: float = 45
    min_hedge_ratio: float = 0.25
    max_hedge_ratio: float = 4.0
    min_spread_std: float = 0.002
    max_return_corr: float = 0.995


def load_price_matrix(price_history: pd.DataFrame) -> pd.DataFrame:
    """Return adjusted closes as date-indexed columns by ticker."""
    required = {"date", "ticker", "adj_close"}
    missing = required.difference(price_history.columns)
    if missing:
        raise ValueError(f"Missing price history columns: {sorted(missing)}")

    prices = price_history.copy(deep=False)
    prices["date"] = pd.to_datetime(prices["date"])
    if prices["ticker"].dtype.name != "category":
        prices["ticker"] = prices["ticker"].astype("category")
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    return prices.pivot_table(
        index="date",
        columns="ticker",
        values="adj_close",
        aggfunc="last",
        observed=True,
    ).sort_index()


def estimate_hedge_ratio(log_y: pd.Series, log_x: pd.Series) -> tuple[float, float]:
    """Estimate y = alpha + beta * x and return alpha, beta."""
    model = sm.OLS(log_y, sm.add_constant(log_x)).fit()
    alpha = float(model.params["const"])
    beta = float(model.params[log_x.name])
    return alpha, beta


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life in trading days."""
    spread_lag = spread.shift(1)
    spread_delta = spread - spread_lag
    regression_data = pd.concat([spread_lag, spread_delta], axis=1).dropna()
    regression_data.columns = ["spread_lag", "spread_delta"]

    if regression_data.empty:
        return np.nan

    model = sm.OLS(
        regression_data["spread_delta"],
        sm.add_constant(regression_data["spread_lag"]),
    ).fit()
    speed = float(model.params["spread_lag"])
    if speed >= 0:
        return np.nan

    return float(-np.log(2) / speed)


def analyze_pair(price_matrix: pd.DataFrame, ticker_a: str, ticker_b: str) -> dict[str, float | int]:
    """Calculate full-sample cointegration metrics for one pair."""
    pair_prices = price_matrix[[ticker_a, ticker_b]].dropna()
    log_prices = np.log(pair_prices)
    log_a = log_prices[ticker_a]
    log_b = log_prices[ticker_b]

    coint_stat, coint_pvalue, _ = coint(log_a, log_b)
    alpha, hedge_ratio = estimate_hedge_ratio(log_a, log_b)
    spread = log_a - alpha - hedge_ratio * log_b
    adf_stat, adf_pvalue, *_ = adfuller(spread)
    half_life_days = estimate_half_life(spread)
    return_corr = float(log_a.diff().corr(log_b.diff()))

    return {
        "n_obs": int(pair_prices.shape[0]),
        "coint_stat": float(coint_stat),
        "coint_pvalue": float(coint_pvalue),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "hedge_ratio": float(hedge_ratio),
        "intercept": float(alpha),
        "half_life_days": half_life_days,
        "return_corr": return_corr,
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std()),
    }


def scan_pairs(
    candidate_pairs: pd.DataFrame,
    price_history: pd.DataFrame,
    config: CointegrationConfig = CointegrationConfig(),
) -> pd.DataFrame:
    """Run the statistical filter on candidate pairs."""
    price_matrix = load_price_matrix(price_history)
    candidates = candidate_pairs[candidate_pairs["passes_pair_filter"]].copy()
    rows: list[dict[str, object]] = []

    for row in candidates.itertuples(index=False):
        ticker_a = row.ticker_a.upper()
        ticker_b = row.ticker_b.upper()
        if ticker_a not in price_matrix or ticker_b not in price_matrix:
            continue

        metrics = analyze_pair(price_matrix, ticker_a, ticker_b)
        result = row._asdict()
        result.update(metrics)
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    results["passes_coint_filter"] = (
        results["coint_pvalue"].lt(config.max_coint_pvalue)
        & results["adf_pvalue"].lt(config.max_adf_pvalue)
        & results["half_life_days"].ge(config.min_half_life_days)
        & results["half_life_days"].le(config.max_half_life_days)
        & results["hedge_ratio"].abs().ge(config.min_hedge_ratio)
        & results["hedge_ratio"].abs().le(config.max_hedge_ratio)
        & results["spread_std"].ge(config.min_spread_std)
        & results["return_corr"].le(config.max_return_corr)
    ).fillna(False)

    results["rank_score"] = (
        results["coint_pvalue"].rank(method="min")
        + results["adf_pvalue"].rank(method="min")
        + (results["half_life_days"] - 10).abs().rank(method="min")
        + (results["spread_std"] - 0.02).abs().rank(method="min")
    )

    return results.sort_values(
        ["passes_coint_filter", "rank_score", "coint_pvalue", "adf_pvalue"],
        ascending=[False, True, True, True],
    )


def write_cointegration_scan(
    candidate_pairs_csv: str | Path,
    price_history_csv: str | Path,
    output_csv: str | Path,
    config: CointegrationConfig = CointegrationConfig(),
) -> Path:
    """Read candidate pairs and prices, then write ranked scan results."""
    candidate_pairs = pd.read_csv(candidate_pairs_csv)
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
    results = scan_pairs(candidate_pairs, price_history, config)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    written_path = write_cointegration_scan(
        candidate_pairs_csv=base_dir / "data" / "candidate_pairs.csv",
        price_history_csv=base_dir / "data" / "etf_price_history.csv",
        output_csv=base_dir / "outputs" / "cointegration_scan.csv",
    )
    print(f"Wrote {written_path}")
