"""Parameter robustness checks for the current test-set pairs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest_pair import BacktestConfig, compute_walk_forward_signals, run_backtest


TOP_PAIR_LIMIT = 5
FORMATIONS = (252, 504, 756)
ENTRY_ZS = (1.5, 2.0, 2.5)
EXIT_ZS = (0.0, 0.5, 1.0)
COSTS_BPS = (4.0, 8.0, 12.0, 20.0)
RISK_SCENARIOS = (
    ("none", None, None),
    ("max_hold_30", 30, None),
    ("max_hold_60", 60, None),
    ("stop_4", None, 4.0),
    ("max_hold_60_stop_4", 60, 4.0),
)
ROLLING_GATES = (False, True)


def choose_top_pairs(backtest_comparison: pd.DataFrame, limit: int = TOP_PAIR_LIMIT) -> pd.DataFrame:
    """Choose a small set of pairs for robustness testing."""
    candidates = backtest_comparison[
        (backtest_comparison["total_return"] > 0)
        & (backtest_comparison["completed_trades"] >= 8)
    ].copy()
    if candidates.empty:
        candidates = backtest_comparison.copy()

    return candidates.sort_values(
        ["sharpe", "total_return", "completed_trades"],
        ascending=[False, False, False],
    ).head(limit)


def add_rolling_gate(signals: pd.DataFrame, rolling_csv: Path) -> pd.DataFrame:
    """Attach latest rolling pass/fail state to each signal date."""
    signals = signals.copy()
    signals["date"] = pd.to_datetime(signals["date"])

    if not rolling_csv.exists():
        signals["rolling_pass"] = True
        return signals

    rolling = pd.read_csv(rolling_csv, parse_dates=["end_date"])
    rolling = rolling[["end_date", "passes_window_filter"]].sort_values("end_date")
    merged = pd.merge_asof(
        signals.sort_values("date"),
        rolling,
        left_on="date",
        right_on="end_date",
        direction="backward",
    )
    merged["rolling_pass"] = merged["passes_window_filter"].fillna(False).astype(bool)
    return merged.drop(columns=["end_date", "passes_window_filter"])


def run_parameter_sweep(
    price_history_csv: str | Path,
    backtest_comparison_csv: str | Path,
    output_dir: str | Path,
    base_config: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """Run parameter sweeps for the current top pairs."""
    price_history = pd.read_csv(price_history_csv)
    comparison = pd.read_csv(backtest_comparison_csv)
    top_pairs = choose_top_pairs(comparison)
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []

    for pair in top_pairs.itertuples(index=False):
        ticker_a = pair.ticker_a
        ticker_b = pair.ticker_b
        pair_name = f"{ticker_a}_{ticker_b}"
        signal_cache: dict[int, pd.DataFrame] = {}

        for formation_days in FORMATIONS:
            signal_config = replace(
                base_config,
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                formation_days=formation_days,
            )
            signals = compute_walk_forward_signals(price_history, signal_config)
            rolling_csv = output_dir / f"rolling_{pair_name}.csv"
            signals = add_rolling_gate(signals, rolling_csv)
            signal_cache[formation_days] = signals

        for formation_days in FORMATIONS:
            signals = signal_cache[formation_days]
            for entry_z in ENTRY_ZS:
                for exit_z in EXIT_ZS:
                    if exit_z >= entry_z:
                        continue
                    for cost_bps in COSTS_BPS:
                        for risk_name, max_hold, stop_z in RISK_SCENARIOS:
                            for require_rolling_pass in ROLLING_GATES:
                                config = replace(
                                    base_config,
                                    ticker_a=ticker_a,
                                    ticker_b=ticker_b,
                                    formation_days=formation_days,
                                    entry_z=entry_z,
                                    exit_z=exit_z,
                                    round_trip_cost_bps=cost_bps,
                                    max_holding_days=max_hold,
                                    stop_z=stop_z,
                                    require_rolling_pass=require_rolling_pass,
                                )
                                trades, daily, summary = run_backtest(signals, config)
                                row = summary.iloc[0].to_dict()
                                row["risk_scenario"] = risk_name
                                row["baseline_rank_pair"] = pair_name
                                row["n_signal_days"] = len(signals)
                                rows.append(row)

    sweep = pd.DataFrame(rows)
    if sweep.empty:
        return sweep

    sweep["robust_pass"] = (
        sweep["total_return"].gt(0)
        & sweep["completed_trades"].ge(5)
        & sweep["max_drawdown"].gt(-0.02)
    )
    return sweep.sort_values(
        ["robust_pass", "sharpe", "total_return", "completed_trades"],
        ascending=[False, False, False, False],
    )


def summarize_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sweep results by pair."""
    if sweep.empty:
        return sweep

    grouped = sweep.groupby(["ticker_a", "ticker_b"], as_index=False)
    summary = grouped.agg(
        parameter_sets=("ticker_a", "size"),
        robust_pass_rate=("robust_pass", "mean"),
    )
    positive = grouped["total_return"].apply(lambda x: (x > 0).mean()).rename(
        columns={"total_return": "positive_return_rate"}
    )
    enough_trades = grouped["completed_trades"].apply(lambda x: (x >= 5).mean()).rename(
        columns={"completed_trades": "enough_trades_rate"}
    )
    medians = grouped.agg(
        median_total_return=("total_return", "median"),
        median_sharpe=("sharpe", "median"),
        median_trades=("completed_trades", "median"),
        worst_max_drawdown=("max_drawdown", "min"),
        best_total_return=("total_return", "max"),
        worst_total_return=("total_return", "min"),
    )
    summary = summary.merge(positive, on=["ticker_a", "ticker_b"])
    summary = summary.merge(enough_trades, on=["ticker_a", "ticker_b"])
    summary = summary.merge(medians, on=["ticker_a", "ticker_b"])
    return summary.sort_values(
        ["robust_pass_rate", "median_sharpe", "median_total_return"],
        ascending=[False, False, False],
    )


def write_research_summary(
    sweep_summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write a short markdown research summary."""
    output_path = Path(output_path)
    lines = [
        "# Test-Set Research Summary",
        "",
        "This summarizes the current curated ETF test set after cointegration scanning, rolling checks, portfolio backtests, and parameter sweeps.",
        "",
        "## Parameter Robustness",
        "",
    ]
    if sweep_summary.empty:
        lines.append("No sweep results were generated.")
    else:
        display_cols = [
            "ticker_a",
            "ticker_b",
            "robust_pass_rate",
            "positive_return_rate",
            "median_total_return",
            "median_sharpe",
            "median_trades",
            "worst_max_drawdown",
        ]
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
        for row in sweep_summary[display_cols].itertuples(index=False):
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        best = sweep_summary.iloc[0]
        lines.extend(
            [
                "",
                "## Current Read",
                "",
                f"Best robustness candidate: {best['ticker_a']} / {best['ticker_b']}.",
                "",
                "Caveats: this is still a simple daily-data backtest with rough costs, fixed gross notional, no borrow constraints, and no intraday execution modeling.",
                "",
                "Next action: promote robust pairs into a multi-pair portfolio test and compare against higher transaction-cost assumptions.",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_parameter_sweep_outputs(
    price_history_csv: str | Path,
    backtest_comparison_csv: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Run the sweep and write detailed, summary, and markdown outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = run_parameter_sweep(price_history_csv, backtest_comparison_csv, output_dir)
    sweep_path = output_dir / "parameter_sweep_comparison.csv"
    sweep.to_csv(sweep_path, index=False)

    sweep_summary = summarize_sweep(sweep)
    summary_path = output_dir / "parameter_sweep_summary.csv"
    research_path = output_dir / "research_summary.md"
    sweep_summary.to_csv(summary_path, index=False)
    write_research_summary(sweep_summary, research_path)
    return sweep_path, summary_path, research_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    paths = write_parameter_sweep_outputs(
        price_history_csv=base_dir / "data" / "etf_price_history.csv",
        backtest_comparison_csv=base_dir / "outputs" / "backtest_comparison.csv",
        output_dir=base_dir / "outputs",
    )
    for path in paths:
        print(f"Wrote {path}")
