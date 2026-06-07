"""Run the full pairs research pipeline with one command."""

from __future__ import annotations

import argparse
from pathlib import Path

from backtest_pair import BacktestConfig
from batch_backtest_pairs import write_batch_backtest_comparison
from build_universe import write_universe
from download_prices import write_price_history as write_yfinance_prices
from filter_universe import write_filtered_outputs
from generate_mvp_report import write_mvp_report
from parameter_sweep_pairs import write_parameter_sweep_outputs
from portfolio_backtest import write_portfolio_outputs
from recent_opportunity_ranker import write_recent_opportunity_rankings
from rolling_pair_check import RollingConfig, write_rolling_check
from scan_cointegration import write_cointegration_scan
from walk_forward_portfolio import WalkForwardConfig, write_walk_forward_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETF pairs research pipeline.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing data/etf_price_history.csv instead of downloading prices.",
    )
    parser.add_argument(
        "--skip-universe",
        action="store_true",
        help="Use existing data/etf_universe_seed.csv instead of rebuilding universe.",
    )
    parser.add_argument(
        "--price-period",
        default="5y",
        help="yfinance price period, e.g. 5y or 10y.",
    )
    parser.add_argument(
        "--top-rolling-pairs",
        type=int,
        default=10,
        help="Number of cointegration-passing pairs to run rolling validation for.",
    )
    parser.add_argument(
        "--walk-forward-max-folds",
        type=int,
        default=None,
        help="Limit walk-forward validation to the first N folds for faster test runs.",
    )
    parser.add_argument(
        "--walk-forward-latest",
        action="store_true",
        help="Run limited walk-forward folds starting from the most recent fold.",
    )
    parser.add_argument(
        "--disable-hmm",
        action="store_true",
        help="Disable the HMM regime filter in walk-forward validation.",
    )
    parser.add_argument(
        "--recent-max-candidates",
        type=int,
        default=75,
        help="Maximum recent cointegration candidates to rank by train/validation tradability.",
    )
    return parser.parse_args()


def run_pipeline(
    base_dir: Path,
    skip_download: bool = False,
    skip_universe: bool = False,
    price_period: str = "5y",
    top_rolling_pairs: int = 10,
    walk_forward_max_folds: int | None = None,
    walk_forward_latest: bool = False,
    use_hmm_filter: bool = True,
    recent_max_candidates: int = 75,
) -> dict[str, Path]:
    """Run universe -> data -> signals -> rolling -> backtest -> ranked results."""
    data_dir = base_dir / "data"
    output_dir = base_dir / "outputs"
    universe_csv = data_dir / "etf_universe_seed.csv"
    price_csv = data_dir / "etf_price_history.csv"

    if not skip_universe:
        print("Building ETF universe...")
        write_universe(universe_csv)

    if not skip_download:
        print("Downloading yfinance price history...")
        write_yfinance_prices(
            universe_csv=universe_csv,
            output_csv=price_csv,
            period=price_period,
        )

    print("Filtering liquid ETFs and generating candidate pairs...")
    liquid_csv, candidate_pairs_csv = write_filtered_outputs(
        universe_csv=universe_csv,
        price_history_csv=price_csv,
        output_dir=data_dir,
    )

    print("Running cointegration scan...")
    cointegration_csv = write_cointegration_scan(
        candidate_pairs_csv=candidate_pairs_csv,
        price_history_csv=price_csv,
        output_csv=output_dir / "cointegration_scan.csv",
    )

    print("Ranking recent train/validation opportunities...")
    recent_opportunities_csv = write_recent_opportunity_rankings(
        price_history_csv=price_csv,
        candidate_pairs_csv=candidate_pairs_csv,
        output_csv=output_dir / "recent_opportunity_rankings.csv",
        config=WalkForwardConfig(
            max_folds=walk_forward_max_folds,
            latest_folds_first=walk_forward_latest,
            use_hmm_filter=use_hmm_filter,
            max_train_candidates=recent_max_candidates,
        ),
        latest=walk_forward_latest,
        max_candidates=recent_max_candidates,
    )

    print("Running rolling validation for top pairs...")
    import pandas as pd

    scan = pd.read_csv(cointegration_csv)
    rolling_pairs = scan[scan["passes_coint_filter"]].head(top_rolling_pairs)
    for row in rolling_pairs.itertuples(index=False):
        write_rolling_check(
            price_history_csv=price_csv,
            output_dir=output_dir,
            config=RollingConfig(ticker_a=row.ticker_a, ticker_b=row.ticker_b),
        )

    print("Running pair backtests and ranking results...")
    backtest_comparison_csv = write_batch_backtest_comparison(
        price_history_csv=price_csv,
        cointegration_scan_csv=cointegration_csv,
        output_dir=output_dir,
        base_config=BacktestConfig(),
    )

    print("Building portfolio allocation from ranked pairs...")
    _, portfolio_daily_csv, portfolio_summary_csv = write_portfolio_outputs(
        backtest_comparison_csv=backtest_comparison_csv,
        output_dir=output_dir,
    )

    print("Running walk-forward portfolio validation...")
    _, _, _, walk_forward_daily_csv, walk_forward_summary_csv = write_walk_forward_outputs(
        price_history_csv=price_csv,
        candidate_pairs_csv=candidate_pairs_csv,
        output_dir=output_dir,
        config=WalkForwardConfig(
            max_folds=walk_forward_max_folds,
            latest_folds_first=walk_forward_latest,
            use_hmm_filter=use_hmm_filter,
            max_train_candidates=recent_max_candidates,
        ),
    )

    print("Running parameter robustness sweep...")
    _, parameter_summary_csv, research_summary_md = write_parameter_sweep_outputs(
        price_history_csv=price_csv,
        backtest_comparison_csv=backtest_comparison_csv,
        output_dir=output_dir,
    )

    print("Generating MVP report...")
    mvp_report_md = write_mvp_report(base_dir)

    return {
        "universe": universe_csv,
        "prices": price_csv,
        "liquid_universe": liquid_csv,
        "candidate_pairs": candidate_pairs_csv,
        "cointegration_scan": cointegration_csv,
        "recent_opportunities": recent_opportunities_csv,
        "backtest_comparison": backtest_comparison_csv,
        "portfolio_daily": portfolio_daily_csv,
        "portfolio_summary": portfolio_summary_csv,
        "walk_forward_daily": walk_forward_daily_csv,
        "walk_forward_summary": walk_forward_summary_csv,
        "parameter_summary": parameter_summary_csv,
        "research_summary": research_summary_md,
        "mvp_report": mvp_report_md,
    }


if __name__ == "__main__":
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[2]
    outputs = run_pipeline(
        base_dir=base_dir,
        skip_download=args.skip_download,
        skip_universe=args.skip_universe,
        price_period=args.price_period,
        top_rolling_pairs=args.top_rolling_pairs,
        walk_forward_max_folds=args.walk_forward_max_folds,
        walk_forward_latest=args.walk_forward_latest,
        use_hmm_filter=not args.disable_hmm,
        recent_max_candidates=args.recent_max_candidates,
    )
    print("\nPipeline complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")
