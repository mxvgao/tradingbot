# ETF Pairs Trading Research MVP

This project is an ETF pairs-trading research engine. It builds a liquid ETF universe, generates economically related pair candidates, runs statistical screens, ranks recent opportunities, and tests selected pairs out-of-sample.

The MVP goal is not to claim a production-ready trading strategy. The goal is to show a credible quant research workflow that can find, test, reject, and explain ETF pair opportunities.

## Current Finding

The project found that ETF pair relationships are highly regime-dependent.

Full-sample 10-year cointegration is very selective, while recent validation winners can still fail in the next walk-forward test window. The strongest current individual result is `BNO/XOP`, but broader selected portfolios are not yet robust enough for production trading.

## Quick Run

Use the existing downloaded 10-year yfinance data:

```text
python pairs_trading/src/pairs_research/run_pipeline.py --skip-download --skip-universe --walk-forward-max-folds 1 --walk-forward-latest --disable-hmm --recent-max-candidates 75
```

Fresh 10-year run:

```text
python pairs_trading/src/pairs_research/run_pipeline.py --price-period 10y --walk-forward-max-folds 1 --walk-forward-latest --disable-hmm --recent-max-candidates 75
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

## Latest MVP Result

In the latest one-window 10-year run:

```text
Universe ETFs:             1,666
Candidate pairs:           791
Full-sample coint passed:  7
Recent opportunity rows:   8
Latest walk-forward return: negative
Best selected test pair:   BNO/XOP
```

The best current individual pair result:

```text
BNO/XOP
test return: about +0.95%
test Sharpe: about 2.18
```

The selected portfolio was dragged down by weaker pairs, which suggests the next improvement is not more breadth. It is better selection confidence and smaller, higher-conviction portfolios.

The most promising follow-up is a focused energy/commodity-producer experiment. `BNO/XOP` has a clear economic relationship between crude oil exposure and oil producer equities, and it was the only selected pair with a positive walk-forward result in the latest MVP run. A natural next research branch is to build a smaller energy universe and walk-forward test all related oil, energy producer, oil services, exploration/production, and broad commodity pairs.

## Limitations

```text
yfinance daily data is research-grade, not execution-grade
bid/ask spreads and slippage are simplified
validation windows still have small trade counts
ETF relationships can break quickly during regime changes
the strategy is not production-ready yet
```

## Next Steps

```text
run a focused energy/commodity-producer experiment around BNO/XOP-like pairs
walk-forward test all pairs in that smaller energy universe
test top 1-3 high-conviction portfolios instead of forcing 5 pairs
add entry confirmation before trading z-score extremes
run more walk-forward folds on the 10-year dataset
add richer ETF metadata such as AUM, issuer, fees, and benchmark
add more realistic execution cost modeling
```
