# Test-Set Research Summary

This summarizes the current curated ETF test set after cointegration scanning, rolling checks, portfolio backtests, and parameter sweeps.

## Parameter Robustness

| ticker_a | ticker_b | robust_pass_rate | positive_return_rate | median_total_return | median_sharpe | median_trades | worst_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CEF | GDX | 0.4389 | 0.6593 | 0.0028 | 0.0805 | 11.0000 | -0.0284 |
| IXJ | IYH | 0.3250 | 0.4167 | -0.0003 | -0.0565 | 7.0000 | -0.0105 |

## Current Read

Best robustness candidate: CEF / GDX.

Caveats: this is still a simple daily-data backtest with rough costs, fixed gross notional, no borrow constraints, and no intraday execution modeling.

Next action: validate read-only paper-account reconciliation over several actual dry runs, then implement a separate submission worker and paper trade for several weeks before adding models. These in-sample sweeps are not independent profitability evidence.