# Historical rerun — 2026-09-22

The existing broad-universe pipeline completed under `session-engine-v1`, using fixed signal-time quantities, next-session-close fills, trading-session holding periods and terminal liquidation. No new model was introduced.

One latest complete test fold, **2025-10-01 through 2026-04-01**, selected USO/XOP and HYD/HYMB. Portfolio return was **−1.70%**, Sharpe **−0.77**, maximum drawdown **−3.44%**. This is limited out-of-sample evidence, not a profitability claim. The complete downloaded history extends through 2026-09-21; the pipeline's fixed fold grid leaves a trailing incomplete test window unused.

The saved seed contained 1,666 ETFs; 162 passed liquidity checks, yielding 459 candidate pairs. HMM was disabled. Training ran 2023-09-27 through 2025-09-30, with validation beginning 2025-04-01. Full-history screens and parameter sweeps are in-sample.

[Manifest and checksums](historical_rerun_2026-09-22.json), [report](mvp_report.md), and [supporting tables](historical_snapshot_2026-09-22/) retain the run's provenance. The source download remains in the ignored local price CSV (about 112 MB); it is not included in Git. Re-downloading may change adjusted prices. The forthcoming frozen research experiment stores its smaller input snapshot in Git to make exact reruns portable.

Reproduction against the retained input:

```sh
uv run --frozen pairs-research --skip-universe --skip-download --price-period 10y --walk-forward-max-folds 1 --walk-forward-latest --disable-hmm --recent-max-candidates 75
```

The initial attempt downloaded prices, then was restarted against the same file after a state-copy performance optimization. Source hashes identify that engine; the 81 foundation tests and golden economic fixtures still pass. Report wording was regenerated after removing obsolete hardcoded legacy conclusions. Old pair-count sensitivity results are archived under `outputs/legacy/` and are excluded from this rerun.

Limitations include the saved present-day universe, missing download histories, heuristic categories, simplified costs/borrow, overlapping ticker exposures, and only one test fold. These outputs establish a corrected reference run, not an independent strategy-selection result.
