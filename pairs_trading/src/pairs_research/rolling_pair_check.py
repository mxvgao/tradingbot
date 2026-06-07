"""Rolling stability and first-pass trading viability checks for one ETF pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scan_cointegration import analyze_pair, load_price_matrix
except ImportError:
    from pairs_research.scan_cointegration import analyze_pair, load_price_matrix


@dataclass(frozen=True)
class RollingConfig:
    ticker_a: str = "SCHQ"
    ticker_b: str = "SPTL"
    window_days: int = 504
    step_days: int = 21
    max_coint_pvalue: float = 0.05
    max_adf_pvalue: float = 0.05
    min_half_life_days: float = 1
    max_half_life_days: float = 45
    entry_z: float = 2.0
    exit_z: float = 0.5
    one_way_cost_bps_per_leg: float = 1.0


def rolling_pair_metrics(price_history: pd.DataFrame, config: RollingConfig) -> pd.DataFrame:
    """Run pair diagnostics across rolling windows."""
    ticker_a = config.ticker_a.upper()
    ticker_b = config.ticker_b.upper()
    pair_history = price_history[
        price_history["ticker"].isin([ticker_a, ticker_b])
    ].copy()
    price_matrix = load_price_matrix(pair_history)
    pair_prices = price_matrix[[ticker_a, ticker_b]].dropna()

    rows: list[dict[str, object]] = []
    for end_idx in range(config.window_days, len(pair_prices) + 1, config.step_days):
        window = pair_prices.iloc[end_idx - config.window_days : end_idx]
        metrics = analyze_pair(window, ticker_a, ticker_b)
        rows.append(
            {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "window_days": config.window_days,
                "step_days": config.step_days,
                "start_date": window.index.min().date(),
                "end_date": window.index.max().date(),
                **metrics,
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results["passes_window_filter"] = (
        results["coint_pvalue"].lt(config.max_coint_pvalue)
        & results["adf_pvalue"].lt(config.max_adf_pvalue)
        & results["half_life_days"].ge(config.min_half_life_days)
        & results["half_life_days"].le(config.max_half_life_days)
    ).fillna(False)
    return results


def summarize_rolling_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize rolling stability into one row."""
    if results.empty:
        return pd.DataFrame()

    latest = results.iloc[-1]
    summary = {
        "ticker_a": latest["ticker_a"],
        "ticker_b": latest["ticker_b"],
        "window_days": int(latest["window_days"]),
        "step_days": int(latest["step_days"]),
        "n_windows": int(len(results)),
        "window_pass_rate": float(results["passes_window_filter"].mean()),
        "coint_pass_rate": float(results["coint_pvalue"].lt(0.05).mean()),
        "adf_pass_rate": float(results["adf_pvalue"].lt(0.05).mean()),
        "half_life_pass_rate": float(results["half_life_days"].between(1, 45).mean()),
        "hedge_ratio_min": float(results["hedge_ratio"].min()),
        "hedge_ratio_median": float(results["hedge_ratio"].median()),
        "hedge_ratio_max": float(results["hedge_ratio"].max()),
        "half_life_min": float(results["half_life_days"].min()),
        "half_life_median": float(results["half_life_days"].median()),
        "half_life_max": float(results["half_life_days"].max()),
        "latest_coint_pvalue": float(latest["coint_pvalue"]),
        "latest_adf_pvalue": float(latest["adf_pvalue"]),
        "latest_half_life_days": float(latest["half_life_days"]),
        "latest_hedge_ratio": float(latest["hedge_ratio"]),
        "latest_passes_window_filter": bool(latest["passes_window_filter"]),
    }
    return pd.DataFrame([summary])


def estimate_trading_viability(
    price_history: pd.DataFrame,
    config: RollingConfig,
    lookback_days: int = 504,
) -> pd.DataFrame:
    """Estimate whether spread moves are large enough versus rough costs."""
    ticker_a = config.ticker_a.upper()
    ticker_b = config.ticker_b.upper()
    pair_history = price_history[
        price_history["ticker"].isin([ticker_a, ticker_b])
    ].copy()
    price_matrix = load_price_matrix(pair_history)
    pair_prices = price_matrix[[ticker_a, ticker_b]].dropna().tail(lookback_days)
    metrics = analyze_pair(pair_prices, ticker_a, ticker_b)

    log_prices = np.log(pair_prices)
    spread = (
        log_prices[ticker_a]
        - metrics["intercept"]
        - metrics["hedge_ratio"] * log_prices[ticker_b]
    )
    zscore = (spread - spread.mean()) / spread.std()
    entry_events = int((zscore.abs() >= config.entry_z).sum())
    spread_std_bps = float(spread.std() * 10_000)
    gross_convergence_bps = float((config.entry_z - config.exit_z) * spread_std_bps)

    # Entry and exit both trade two legs. This is a rough daily-data proxy.
    round_trip_cost_bps = float(4 * config.one_way_cost_bps_per_leg)
    cost_coverage_ratio = gross_convergence_bps / round_trip_cost_bps

    return pd.DataFrame(
        [
            {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "lookback_days": int(len(pair_prices)),
                "entry_z": config.entry_z,
                "exit_z": config.exit_z,
                "entry_signal_days": entry_events,
                "spread_std_bps": spread_std_bps,
                "gross_convergence_bps": gross_convergence_bps,
                "round_trip_cost_bps": round_trip_cost_bps,
                "cost_coverage_ratio": cost_coverage_ratio,
                "viability_note": (
                    "likely too tight after costs"
                    if cost_coverage_ratio < 3
                    else "large enough to investigate"
                ),
            }
        ]
    )


def write_rolling_check(
    price_history_csv: str | Path,
    output_dir: str | Path,
    config: RollingConfig = RollingConfig(),
) -> tuple[Path, Path, Path]:
    """Write rolling windows, summary, and viability CSV files."""
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

    pair_name = f"{config.ticker_a.upper()}_{config.ticker_b.upper()}"
    rolling = rolling_pair_metrics(price_history, config)
    summary = summarize_rolling_results(rolling)
    viability = estimate_trading_viability(price_history, config)

    rolling_path = output_dir / f"rolling_{pair_name}.csv"
    summary_path = output_dir / f"rolling_{pair_name}_summary.csv"
    viability_path = output_dir / f"viability_{pair_name}.csv"
    rolling.to_csv(rolling_path, index=False)
    summary.to_csv(summary_path, index=False)
    viability.to_csv(viability_path, index=False)
    return rolling_path, summary_path, viability_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    paths = write_rolling_check(
        price_history_csv=base_dir / "data" / "etf_price_history.csv",
        output_dir=base_dir / "outputs",
    )
    for path in paths:
        print(f"Wrote {path}")
