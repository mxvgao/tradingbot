"""Run the barebones pairs backtest across cointegration-filtered pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest_pair import BacktestConfig, write_backtest_outputs


def run_batch_backtests(
    price_history_csv: str | Path,
    cointegration_scan_csv: str | Path,
    output_dir: str | Path,
    base_config: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """Backtest all pairs that passed the cointegration filter."""
    scan = pd.read_csv(cointegration_scan_csv)
    pairs = scan[scan["passes_coint_filter"]].copy()
    output_dir = Path(output_dir)
    summaries: list[pd.DataFrame] = []

    for row in pairs.itertuples(index=False):
        config = BacktestConfig(
            ticker_a=row.ticker_a,
            ticker_b=row.ticker_b,
            formation_days=base_config.formation_days,
            entry_z=base_config.entry_z,
            exit_z=base_config.exit_z,
            round_trip_cost_bps=base_config.round_trip_cost_bps,
            initial_capital=base_config.initial_capital,
            gross_notional_per_trade=base_config.gross_notional_per_trade,
            max_holding_days=base_config.max_holding_days,
            stop_z=base_config.stop_z,
        )
        _, _, summary_path = write_backtest_outputs(
            price_history_csv=price_history_csv,
            output_dir=output_dir,
            config=config,
        )
        summary = pd.read_csv(summary_path)
        summary["coint_pvalue"] = row.coint_pvalue
        summary["adf_pvalue"] = row.adf_pvalue
        summary["half_life_days"] = row.half_life_days
        summary["return_corr"] = row.return_corr
        summary["group"] = row.group
        summary["subgroup"] = row.subgroup
        summaries.append(summary)

    if not summaries:
        return pd.DataFrame()

    comparison = pd.concat(summaries, ignore_index=True)
    comparison["enough_trades"] = comparison["completed_trades"].ge(8)
    comparison["positive_return"] = comparison["total_return"].gt(0)
    comparison["ranking_score"] = (
        comparison["sharpe"].fillna(0)
        + comparison["total_return"].fillna(0) * 100
        + comparison["completed_trades"].clip(upper=20) / 20
        + comparison["enough_trades"].astype(int)
    )
    return comparison.sort_values(
        ["positive_return", "enough_trades", "ranking_score", "sharpe"],
        ascending=[False, False, False, False],
    )


def write_batch_backtest_comparison(
    price_history_csv: str | Path,
    cointegration_scan_csv: str | Path,
    output_dir: str | Path,
    base_config: BacktestConfig = BacktestConfig(),
) -> Path:
    """Run batch backtests and write a sorted comparison CSV."""
    comparison = run_batch_backtests(
        price_history_csv=price_history_csv,
        cointegration_scan_csv=cointegration_scan_csv,
        output_dir=output_dir,
        base_config=base_config,
    )
    output_path = Path(output_dir) / "backtest_comparison.csv"
    comparison.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    written_path = write_batch_backtest_comparison(
        price_history_csv=base_dir / "data" / "etf_price_history.csv",
        cointegration_scan_csv=base_dir / "outputs" / "cointegration_scan.csv",
        output_dir=base_dir / "outputs",
    )
    print(f"Wrote {written_path}")
