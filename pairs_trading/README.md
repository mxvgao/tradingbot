# ETF Pairs Trading Research

Goal: find liquid ETF pairs with stable cointegration, then test whether the relationship survives realistic out-of-sample trading assumptions.

## Suggested Workflow

### 1. Build the ETF universe

Start with a broad ETF list, then enrich it with metadata:

- ticker
- fund name
- issuer
- asset class
- category/theme
- expense ratio
- AUM
- average dollar volume
- inception date

Good first universe sources:

- ETF.com or VettaFi exports for ETF metadata
- Nasdaq/NYSE ETF listings
- yfinance for quick price and volume history

Do not start by testing every ETF against every other ETF. That creates too many weak statistical discoveries.

### 2. Filter for tradability

Keep ETFs that are easy to trade:

- sufficient daily history, ideally 3-5 years or more
- average dollar volume above a chosen threshold
- no obvious stale pricing
- no major missing-data gaps
- avoid ETNs, leveraged ETFs, inverse ETFs, and ultra-niche products at first

Reasonable first filters:

- at least 756 trading days of history
- average dollar volume above $10M
- median daily volume above 100k shares
- price above $5

### 3. Group ETFs before testing pairs

Cointegration is more plausible within related economic exposures:

- broad US equity: SPY, IVV, VOO, VTI
- sectors: XLF, VFH, IYF
- international regions: EFA, IEFA, VEA
- bonds by duration/credit: IEF, GOVT, VGIT
- commodities themes
- factor ETFs: value, growth, momentum, low volatility

Use strict same-category pairs first. Later, test related groups like regional equity vs country ETFs.

### 4. Run the statistical screen

For each candidate pair:

1. Align adjusted-close prices.
2. Work in log prices.
3. Run Engle-Granger cointegration test.
4. Estimate hedge ratio with OLS.
5. Compute spread:

   ```text
   spread = log_price_y - hedge_ratio * log_price_x
   ```

6. Run ADF test on the spread.
7. Estimate half-life from mean reversion regression.

Keep only pairs where:

- Engle-Granger p-value < 0.05, ideally < 0.01
- ADF p-value on spread < 0.05
- half-life is between 3 and 30 trading days
- hedge ratio is economically reasonable
- price history has enough overlapping data

### 5. Require rolling stability

A pair that only works over the full sample is usually not enough. Re-run the tests over rolling windows:

- 252 trading days
- 504 trading days
- 756 trading days

Track:

- percent of windows passing the cointegration test
- hedge ratio stability
- half-life stability
- spread volatility stability

This is where many attractive-looking pairs should get rejected.

### 6. Backtest out-of-sample

Use a walk-forward structure:

- formation window: estimate hedge ratio and spread stats
- trading window: trade using only previously-known estimates
- rebalance estimates on a schedule

Example:

- 504-day formation window
- 63-day trading window
- enter when z-score exceeds 2.0
- exit near 0.5 or 0.0
- stop if z-score blows out or half-life deteriorates

Include:

- commissions
- bid/ask slippage
- borrow/short assumptions if shorting is involved
- position sizing
- max exposure limits

## First Implementation Milestones

1. Create a manually curated starter ETF universe.
2. Download adjusted prices and volume.
3. Calculate liquidity filters.
4. Group tickers by theme.
5. Run same-group cointegration scan.
6. Export ranked candidate pairs to CSV.
7. Add rolling-window validation.
8. Add walk-forward backtest.

## Suggested Folder Layout

```text
pairs_trading/
  data/                 raw and cached ETF data
  notebooks/            exploratory research
  outputs/              scans, charts, backtest results
  src/pairs_research/   reusable research code
```

## Practical Notes

Use this project as a research funnel. The statistical tests should find candidates, not final trades. The backtest should be treated as an attempt to disprove each pair before trusting it.

## First Filter Pass

The simple filter expects a price history CSV with:

```text
date,ticker,adj_close,volume
```

Create it with:

```text
python -m pip install -r pairs_trading/requirements.txt
python pairs_trading/src/pairs_research/download_prices.py
```

It writes:

```text
data/etf_universe_liquid.csv
data/candidate_pairs.csv
```

ETF-level filters:

```text
history_days >= 756
median_dollar_volume_60d >= 10000000
median_price_60d >= 5
missing_price_pct <= 0.02
```

Pair-level filters:

```text
same subgroup
overlap_days >= 756
```

Run the first statistical screen with:

```text
python pairs_trading/src/pairs_research/scan_cointegration.py
```

It writes:

```text
outputs/cointegration_scan.csv
```

First-pass cointegration filters:

```text
coint_pvalue < 0.05
adf_pvalue < 0.05
1 <= half_life_days <= 45
0.25 <= abs(hedge_ratio) <= 4.0
```

Run a rolling stability and rough viability check for the leading pair with:

```text
python pairs_trading/src/pairs_research/rolling_pair_check.py
```

It writes:

```text
outputs/rolling_ITOT_VTI.csv
outputs/rolling_ITOT_VTI_summary.csv
outputs/viability_ITOT_VTI.csv
```

Run the barebones walk-forward pairs backtest with:

```text
python pairs_trading/src/pairs_research/backtest_pair.py
```

Default pair:

```text
MUB / VTEB
```

It writes:

```text
outputs/backtest_MUB_VTEB_trades.csv
outputs/backtest_MUB_VTEB_daily.csv
outputs/backtest_MUB_VTEB_summary.csv
```

Run the same backtest across all cointegration-filtered pairs with:

```text
python pairs_trading/src/pairs_research/batch_backtest_pairs.py
```

It writes:

```text
outputs/backtest_comparison.csv
```

Run robustness checks on the top test-set pairs with:

```text
python pairs_trading/src/pairs_research/parameter_sweep_pairs.py
```

It writes:

```text
outputs/parameter_sweep_comparison.csv
outputs/parameter_sweep_summary.csv
outputs/research_summary.md
```
