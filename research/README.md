# Reproducible baseline experiments

The foundation is frozen at annotated tag `research-engine-v1`, commit `287c41d`. Its exact committed checkout passed **81 tests on Python 3.12.12**, using a separate environment installed from the root lockfile. Later adapter work is a separate commit; no model experiments are mixed into the foundation.

Experiment 001 is a rolling log-price OLS reference run, not a new trading model. It uses the same signal and session engine as batch backtests and durable replay. Its complete price snapshot is committed, so rerunning requires no download, credentials or broker access.

```sh
uv sync --frozen
uv run --frozen pairs-experiment --config research/experiments/001_rolling_ols_baseline/config.json
```

Paths inside a configuration are relative to that configuration. The command works from any directory when given an absolute config path. A matching installed wheel also works against this Git checkout. Outputs are written beside the config:

```text
research/
  universes/energy_etfs.csv
  experiments/001_rolling_ols_baseline/
    config.json
    data/energy_adjusted_close.csv
    manifest.json
    results/
      validation_grid.csv
      selected_parameters.csv
      test_metrics.csv
      test_daily.csv
      test_trades.csv
      test_results.png
    report.md
```

To compare a rerun without overwriting the saved report:

```sh
uv run --frozen pairs-experiment \
  --config research/experiments/001_rolling_ols_baseline/config.json \
  --output-dir /tmp/pairs-baseline-rerun
```

The configuration pins SHA-256 checksums of the universe and prices, the data/session range, windows, model, grid, fixed parameters, costs, random seeds, selection policy and execution convention. The manifest adds the executing Git commit, foundation commit, every package source checksum, lock checksum, runtime/dependency versions, actual window counts and every generated artifact checksum. The exact source/input commit is recorded before generated results are committed in a later commit; a subsequent rerun naturally records its new executing commit. Numerical tables should remain identical in the locked environment. Platform-dependent rendering and floating-point differences require comparison at the numerical level across platforms.

Source, configuration and input changes must be committed before an official run. `--allow-dirty` exists only for development and marks the manifest as uncommitted; do not present those outputs as the frozen baseline. Installed code must match the checkout. Checksums, dates, duplicate rows, complete pair/session coverage and window ordering are validated before publishing. The source session grid is frozen by checksum and count; the harness does not independently reconstruct exchange holidays. Missing entire sessions cannot be detected if someone deliberately creates a new configuration endorsing an incomplete grid.

## Experiment 001 protocol

- Universe fixed before this run: BNO, OIH, USO, XES, XLE, XOP. Test all 15 unordered pairs; no outcome-based pair exclusion or cointegration/HMM filter.
- Initial formation: 2022-01-03 through 2023-12-29. Validation: 2024-01-02 through 2024-12-31. Test: 2025-01-02 through 2025-12-31.
- Grid: formation 126/252 sessions × entry z 1.5/2.0. Fixed exit z 0.5, stop z 3.0, maximum hold 20 sessions. Choose the greatest validation net return per pair; canonical parameter JSON resolves ties. Include zero-trade candidates.
- OLS refits on each session's prior window. Earlier test closes can enter later formation estimates; future closes cannot. Parameters are never retuned on test results.
- Each evaluation window begins flat, decides from close t, fills fixed signed shares at t+1 close, marks daily, and liquidates at its declared endpoint. Four basis points round trip means two basis points on actual entry turnover plus two on exit turnover. Independent pair capital is $100,000 with a $10,000 entry gross budget.
- Python and NumPy seeds are 0; the baseline estimator itself is deterministic. Metrics, daily marks and the trade ledger come from the existing engine.

The six-symbol snapshot was downloaded on 2026-09-23 using yfinance `Adj Close`, `auto_adjust=False`, start 2022-01-03, exclusive end 2026-01-01. The earlier broad download lacked BNO/OIH histories, so all six series were freshly fetched together for this separate frozen experiment. Re-downloading is not part of the experiment command and may produce revised values. Create a new experiment ID for changed inputs, windows, universe or model decisions; preserve this baseline.

This is a retrospective research baseline. The selected universe is not point-in-time, adjusted prices can reflect later revisions, and earlier project work inspected overlapping test dates. Its test window is held out from this particular parameter choice, not an untouched prospective sample. Independent pair results are not a portfolio and do not account for common ticker exposure, borrow, financing or variable slippage. More infrastructure and paper submission are outside this milestone.
