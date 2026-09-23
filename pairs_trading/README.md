# ETF Pairs Trading Research MVP

This project is an ETF pairs-trading research engine. It builds a liquid ETF universe, generates economically related pair candidates, runs statistical screens, ranks recent opportunities, and tests selected pairs out-of-sample.

The MVP goal is not to claim a production-ready trading strategy. The goal is to show a credible quant research workflow that can find, test, reject, and explain ETF pair opportunities.

## Current reference run

The shared-engine historical rerun returned −1.70% in one test fold. See the [dated provenance record](reports/historical_rerun_2026-09-22.md) and [current report](reports/mvp_report.md). Earlier same-close findings are not evidence for this engine.

## Quick Run

First run `uv sync --frozen` from the repository root. See the [root README](../README.md) and [PRD](../docs/PRD.md) for setup, next-session execution, and test commands.

Use the existing downloaded 10-year yfinance data:

```text
uv run --frozen pairs-research --skip-download --skip-universe --walk-forward-max-folds 1 --walk-forward-latest --disable-hmm --recent-max-candidates 75
```

Fresh 10-year run:

```text
uv run --frozen pairs-research --price-period 10y --walk-forward-max-folds 1 --walk-forward-latest --disable-hmm --recent-max-candidates 75
```

The fresh run downloads a large price file and can take a while.

## Pipeline

```text
ETF universe
-> liquidity filter
-> candidate pair generation
-> full-sample cointegration scan
-> recent opportunity ranking
-> rolling diagnostics
-> pair backtests
-> portfolio allocation
-> walk-forward validation
-> MVP report
```

## Main Outputs

```text
data/etf_universe_seed.csv
data/etf_universe_liquid.csv
data/candidate_pairs.csv
outputs/cointegration_scan.csv
outputs/recent_opportunity_rankings.csv
outputs/backtest_comparison.csv
outputs/portfolio_summary.csv
outputs/walk_forward_selected_pairs.csv
outputs/walk_forward_pair_results.csv
outputs/walk_forward_summary.csv
reports/mvp_report.md
reports/research_summary.md
reports/charts/
```

## Methodology

Version 0.2.0 fixes quantities at signal close t and fills at t+1 close. Holding limits use trading sessions, all positions are marked daily and liquidated at each test boundary, and execution fill risk is reported separately. Current outputs were regenerated on 2026-09-22; legacy pair-count sensitivity is archived separately.

Universe construction uses Nasdaq Trader ETF listings with heuristic ETF-name classification. The current default universe includes broad equity, sector, industry, factor, country, bond, credit, commodity, metals, and energy ETFs.

Candidate generation avoids testing every ETF against every other ETF. It creates pairs from:

```text
same subgroup
related subgroup
same group with high return correlation
sector vs industry
commodity vs producer
country vs region
factor vs broad market
credit vs equity-sensitive themes
```

The statistical screen uses:

```text
Engle-Granger cointegration
ADF test on the spread
OLS hedge ratio
spread half-life
spread volatility
return correlation sanity checks
```

Recent opportunity ranking uses the long price history for context, but ranks pairs using a current subtrain/validation window. This is meant to answer:

```text
Which pairs look tradable now?
```

instead of:

```text
Which pairs looked cointegrated over the entire 10-year sample?
```

Walk-forward validation tests selected pair/rule combinations out-of-sample using only information available before the test window.

## Risk Controls

The current walk-forward engine supports:

```text
hedge-ratio-sized legs
transaction cost assumptions
max holding period
stop-z exit
no immediate same-direction re-entry after failed max-hold exits
duplicate ticker exposure limits
group concentration limits
optional HMM regime filter
```

The optional HMM uses spread change, z-score change, rolling spread volatility, absolute z-score, and hedge-ratio change to filter regimes. It is fit only on subtrain data and then applied to validation/test windows without lookahead.

## Limitations

```text
yfinance daily data is research-grade, not execution-grade
bid/ask spreads and slippage are simplified
validation windows still have small trade counts
ETF relationships can break quickly during regime changes
the strategy is not production-ready yet
```

## Next milestone

Freeze the research-engine baseline and build a reproducible rolling-OLS experiment with frozen data, an explicit parameter grid, held-out windows, costs and a provenance manifest. More bot infrastructure is deferred.
