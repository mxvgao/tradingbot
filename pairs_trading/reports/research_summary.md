# Test-Set Research Summary

This summarizes the current curated ETF test set after cointegration scanning, rolling checks, portfolio backtests, and parameter sweeps.

## Parameter Robustness

| ticker_a | ticker_b | robust_pass_rate | positive_return_rate | median_total_return | median_sharpe | median_trades | worst_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| JNK | PFXF | 0.8167 | 0.9278 | 0.0095 | 0.4542 | 13.0000 | -0.0129 |
| ANGL | PFXF | 0.5000 | 0.6130 | 0.0038 | 0.1712 | 6.0000 | -0.0094 |
| CEF | GDX | 0.4806 | 0.7380 | 0.0050 | 0.1285 | 12.0000 | -0.0293 |
| IJH | SPMD | 0.2259 | 0.2398 | -0.0004 | -0.3564 | 13.0000 | -0.0219 |
| MDY | SPMD | 0.1824 | 0.2611 | -0.0002 | -0.2076 | 4.0000 | -0.0206 |

## Current Read

Best robustness candidate: JNK / PFXF.

Caveats: this is still a simple daily-data backtest with rough costs, fixed gross notional, no borrow constraints, and no intraday execution modeling.

Next action: promote robust pairs into a multi-pair portfolio test and compare against higher transaction-cost assumptions.