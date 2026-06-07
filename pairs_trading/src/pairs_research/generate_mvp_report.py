"""Generate a concise MVP report and charts for the ETF pairs project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest_pair import BacktestConfig, compute_walk_forward_signals


def setup_matplotlib() -> None:
    """Use a non-GUI matplotlib backend for report generation."""
    import matplotlib

    matplotlib.use("Agg")


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pct(value: object) -> str:
    """Format a decimal return as a percent string."""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def num(value: object, digits: int = 2) -> str:
    """Format a number for the markdown report."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def ensure_chart_dir(output_dir: Path) -> Path:
    """Create and return the chart output directory."""
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    return chart_dir


def save_recent_opportunities_chart(recent: pd.DataFrame, chart_dir: Path) -> str | None:
    """Save a bar chart of the top recent opportunity scores."""
    if recent.empty or "recent_tradability_score" not in recent:
        return None

    import matplotlib.pyplot as plt

    top = recent.head(8).copy()
    top["pair"] = top["ticker_a"] + "/" + top["ticker_b"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top["pair"], top["recent_tradability_score"], color="#2f6f73")
    ax.set_title("Top Recent Opportunity Scores")
    ax.set_ylabel("Tradability score")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = chart_dir / "recent_opportunity_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return "charts/recent_opportunity_scores.png"


def save_equity_chart(daily: pd.DataFrame, chart_dir: Path) -> str | None:
    """Save walk-forward equity curve chart."""
    if daily.empty or "equity" not in daily:
        return None

    import matplotlib.pyplot as plt

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame["date"], frame["equity"], color="#334e68", linewidth=2)
    ax.set_title("Latest Walk-Forward Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = chart_dir / "walk_forward_equity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return "charts/walk_forward_equity.png"


def save_pair_returns_chart(pair_results: pd.DataFrame, chart_dir: Path) -> str | None:
    """Save a bar chart of selected pair test returns."""
    if pair_results.empty or "total_return" not in pair_results:
        return None

    import matplotlib.pyplot as plt

    frame = pair_results.copy()
    frame["pair"] = frame["ticker_a"] + "/" + frame["ticker_b"]
    frame["total_return"] = pd.to_numeric(frame["total_return"], errors="coerce")
    frame = frame.sort_values("total_return", ascending=False)
    colors = ["#2f6f73" if value >= 0 else "#b24745" for value in frame["total_return"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(frame["pair"], frame["total_return"] * 100, color=colors)
    ax.set_title("Selected Pair Test Returns")
    ax.set_ylabel("Return (%)")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = chart_dir / "selected_pair_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return "charts/selected_pair_returns.png"


def save_best_pair_spread_chart(
    price_csv: Path,
    pair_results: pd.DataFrame,
    chart_dir: Path,
) -> str | None:
    """Save a spread/z-score chart for the best selected test pair."""
    if pair_results.empty or "total_return" not in pair_results:
        return None

    best = pair_results.copy()
    best["total_return"] = pd.to_numeric(best["total_return"], errors="coerce")
    best = best.sort_values("total_return", ascending=False).iloc[0]
    ticker_a = str(best["ticker_a"]).upper()
    ticker_b = str(best["ticker_b"]).upper()

    prices = pd.read_csv(
        price_csv,
        usecols=["date", "ticker", "adj_close", "volume"],
        dtype={
            "date": "string",
            "ticker": "category",
            "adj_close": "float32",
            "volume": "float32",
        },
    )
    pair_prices = prices[prices["ticker"].isin([ticker_a, ticker_b])].copy()
    if pair_prices.empty:
        return None

    signals = compute_walk_forward_signals(
        pair_prices,
        BacktestConfig(ticker_a=ticker_a, ticker_b=ticker_b, formation_days=252),
    )
    if signals.empty:
        return None

    import matplotlib.pyplot as plt

    signals["date"] = pd.to_datetime(signals["date"])
    signals = signals.tail(504)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(signals["date"], signals["spread"], color="#334e68")
    axes[0].set_title(f"{ticker_a}/{ticker_b} Spread")
    axes[0].grid(alpha=0.25)
    axes[1].plot(signals["date"], signals["zscore"], color="#2f6f73")
    axes[1].axhline(2, color="#b24745", linestyle="--", linewidth=1)
    axes[1].axhline(-2, color="#b24745", linestyle="--", linewidth=1)
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_title("Walk-Forward Z-Score")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path = chart_dir / "best_pair_spread_zscore.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return "charts/best_pair_spread_zscore.png"


def top_pairs_table(frame: pd.DataFrame, columns: list[str], limit: int = 5) -> str:
    """Render a compact markdown table."""
    if frame.empty:
        return "_No rows available._"

    available = [column for column in columns if column in frame.columns]
    shown = frame.head(limit)[available].copy()
    for column in shown.columns:
        if "return" in column or "drawdown" in column:
            shown[column] = shown[column].map(pct)
        elif "sharpe" in column or "score" in column or "pnl_bps" in column:
            shown[column] = shown[column].map(lambda value: num(value, 2))
    headers = list(shown.columns)
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in shown.itertuples(index=False):
        values = [str(value) for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_report(base_dir: Path) -> tuple[str, list[str]]:
    """Build markdown report text and charts."""
    setup_matplotlib()
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"
    chart_dir = ensure_chart_dir(output_dir)

    universe = read_csv(data_dir / "etf_universe_seed.csv")
    liquid = read_csv(data_dir / "etf_universe_liquid.csv")
    candidates = read_csv(data_dir / "candidate_pairs.csv")
    cointegration = read_csv(output_dir / "cointegration_scan.csv")
    recent = read_csv(output_dir / "recent_opportunity_rankings.csv")
    portfolio_summary = read_csv(output_dir / "portfolio_summary.csv")
    walk_summary = read_csv(output_dir / "walk_forward_summary.csv")
    selected = read_csv(output_dir / "walk_forward_selected_pairs.csv")
    pair_results = read_csv(output_dir / "walk_forward_pair_results.csv")
    daily = read_csv(output_dir / "walk_forward_portfolio_daily.csv")

    charts = [
        save_recent_opportunities_chart(recent, chart_dir),
        save_equity_chart(daily, chart_dir),
        save_pair_returns_chart(pair_results, chart_dir),
        save_best_pair_spread_chart(data_dir / "etf_price_history.csv", pair_results, chart_dir),
    ]
    charts = [chart for chart in charts if chart is not None]

    coint_passed = (
        int(cointegration["passes_coint_filter"].sum())
        if "passes_coint_filter" in cointegration
        else 0
    )
    liquid_count = (
        int(liquid["passes_etf_filter"].sum())
        if "passes_etf_filter" in liquid
        else len(liquid)
    )
    walk = walk_summary.iloc[0].to_dict() if not walk_summary.empty else {}
    portfolio = portfolio_summary.iloc[0].to_dict() if not portfolio_summary.empty else {}

    best_pair = {}
    if not pair_results.empty and "total_return" in pair_results:
        ranked_results = pair_results.copy()
        ranked_results["total_return"] = pd.to_numeric(
            ranked_results["total_return"], errors="coerce"
        )
        best_pair = ranked_results.sort_values("total_return", ascending=False).iloc[0].to_dict()

    lines = [
        "# ETF Pairs Trading MVP Report",
        "",
        "## Executive Summary",
        "",
        "This project is an ETF pairs-trading research engine. It builds a liquid ETF universe, generates economically related candidate pairs, tests cointegration, ranks recent opportunities by train/validation trading quality, and runs walk-forward out-of-sample checks.",
        "",
        "The current MVP does not claim a production-ready trading strategy. Its main finding is that ETF pair relationships are regime-dependent: full-sample cointegration is very selective, while recent validation winners can still fail in the next window. The pipeline is useful because it makes those failures visible instead of hiding them.",
        "",
        "The most promising direction from the current run is not a broader model search. It is a focused follow-up on energy commodity/producer relationships, because BNO/XOP was the only selected pair with a positive walk-forward result and has a clear economic link between crude oil exposure and oil producer equities.",
        "",
        "## Latest Run Snapshot",
        "",
        f"- Universe ETFs: {len(universe):,}",
        f"- Liquid ETFs: {liquid_count:,}",
        f"- Candidate pairs: {len(candidates):,}",
        f"- Full-sample cointegration pass count: {coint_passed:,}",
        f"- Recent opportunity rows: {len(recent):,}",
        f"- Latest walk-forward return: {pct(walk.get('total_return'))}",
        f"- Latest walk-forward Sharpe: {num(walk.get('sharpe'))}",
        f"- Latest walk-forward max drawdown: {pct(walk.get('max_drawdown'))}",
        f"- Full-history portfolio return: {pct(portfolio.get('total_return'))}",
        f"- Full-history portfolio Sharpe: {num(portfolio.get('sharpe'))}",
        "",
        "## Top Recent Opportunities",
        "",
        top_pairs_table(
            recent,
            [
                "ticker_a",
                "ticker_b",
                "candidate_tier",
                "group",
                "recent_tradability_score",
                "recent_validation_completed_trades",
                "recent_validation_total_return",
                "recent_validation_sharpe",
                "recent_validation_avg_net_pnl_bps",
            ],
            limit=8,
        ),
        "",
        "## Latest Walk-Forward Selected Pairs",
        "",
        top_pairs_table(
            selected,
            [
                "ticker_a",
                "ticker_b",
                "group",
                "candidate_tier",
                "validation_tradability_score",
                "train_validation_total_return",
                "train_validation_sharpe",
            ],
            limit=10,
        ),
        "",
        "## Latest Test Results By Pair",
        "",
        top_pairs_table(
            pair_results.sort_values("total_return", ascending=False)
            if not pair_results.empty and "total_return" in pair_results
            else pair_results,
            ["ticker_a", "ticker_b", "group", "completed_trades", "total_return", "sharpe", "max_drawdown"],
            limit=10,
        ),
        "",
        "## Best Current Test Pair",
        "",
    ]

    if best_pair:
        lines.extend(
            [
                f"- Pair: {best_pair.get('ticker_a')}/{best_pair.get('ticker_b')}",
                f"- Test return: {pct(best_pair.get('total_return'))}",
                f"- Test Sharpe: {num(best_pair.get('sharpe'))}",
                f"- Max drawdown: {pct(best_pair.get('max_drawdown'))}",
                "",
            ]
        )
    else:
        lines.extend(["_No selected test pair available._", ""])

    if charts:
        lines.extend(["## Charts", ""])
        for chart in charts:
            lines.append(f"![{chart}]({chart})")
            lines.append("")

    lines.extend(
        [
            "## Methodology",
            "",
            "1. Build a broad ETF universe from Nasdaq Trader listings.",
            "2. Filter for liquidity, price, history length, missing data, and non-leveraged exposure.",
            "3. Generate candidate pairs within same subgroups and related economic themes.",
            "4. Run Engle-Granger cointegration and ADF tests as statistical screens.",
            "5. Rank recent opportunities using subtrain/validation trading quality rather than only full-sample p-values.",
            "6. Test selected pair/rule combinations out-of-sample in a walk-forward window.",
            "7. Report portfolio-level and pair-level results.",
            "",
            "## Key Finding",
            "",
        "The research funnel found some promising individual relationships, especially in energy producer/commodity-style pairs such as BNO/XOP. However, forcing a broader portfolio can dilute those winners with weaker pairs. The next research step is to improve selection confidence, likely by choosing fewer higher-conviction pairs and adding entry confirmation so the strategy avoids fading spreads that are still trending.",
        "",
        "A practical next research branch is an energy-focused experiment: build a smaller universe of oil, broad commodity, energy producer, oil services, and exploration/production ETFs; walk-forward test all economically related pairs in that subset; and evaluate whether BNO/XOP is an isolated winner or part of a repeatable theme.",
            "",
            "## MVP Limitations",
            "",
            "- yfinance daily data is suitable for research, not live execution.",
            "- Bid/ask spreads, intraday fills, financing, borrow constraints, and taxes are simplified.",
            "- Validation windows still have small trade counts.",
            "- HMM filtering reduces bad trades but can become too selective.",
            "- Current profitability is not robust enough for production trading.",
            "",
            "## Suggested Next Steps",
            "",
            "1. Run a focused energy/commodity-producer experiment around BNO/XOP-like relationships.",
            "2. Walk-forward test all pairs in that smaller energy universe instead of relying only on the broad ETF selector.",
            "3. Test smaller portfolios, such as top 1-3 high-conviction pairs, instead of forcing 5.",
            "4. Add entry confirmation: wait for z-score to start reverting before entering.",
            "5. Run more rolling folds on the 10-year dataset.",
            "6. Improve pair metadata with richer ETF categories, AUM, fees, and issuer data.",
            "7. Add realistic slippage and bid/ask cost modeling before any live trading.",
            "",
        ]
    )
    return "\n".join(lines), charts


def write_mvp_report(base_dir: str | Path) -> Path:
    """Write MVP markdown report and charts."""
    base_path = Path(base_dir)
    report, _ = build_report(base_path)
    output_path = base_path / "outputs" / "mvp_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    path = write_mvp_report(base_dir)
    print(f"Wrote {path}")
