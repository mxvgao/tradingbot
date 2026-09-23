# ETF pairs research

A Python 3.12 research system for ETF pairs, with a causal next-session-close
backtest and a separate execution-fill simulator. The product requirements and
acceptance evidence live in [docs/PRD.md](docs/PRD.md).

## Reproducible setup

Use uv 0.10.12 and the checked-in `.python-version` and `uv.lock`:

```sh
uv sync --frozen
uv run --frozen pairs-check-environment
uv run --frozen pytest
uv run --frozen pairs-research --help
```

On this workstation the same manager is available as `.tools/bin/uv`; the
prepared environment is `.venv` and its interpreter is Python 3.12.12.
You can run `.venv/bin/python -m pytest` directly. The system Python is not used.
For dependency changes, edit the root `pyproject.toml`, run `uv lock`, and commit
both files. Do not install separate pairs requirements or use `.python_packages`.
All research and test dependencies are in the root project configuration.

## Research pipeline

From the repository root, using already downloaded data:

```sh
uv run --frozen pairs-research --base-dir pairs_trading --skip-download --skip-universe --walk-forward-max-folds 1 --walk-forward-latest --disable-hmm --recent-max-candidates 75
```

Omit `--skip-download` for a fresh download. The command is also available as
`python -m pairs_research.run_pipeline`. It accepts an explicit data/output base
directory and works from an installed wheel outside this checkout. All other
research modules can be imported as `pairs_research.<module>`; do not invoke
source files by path or modify `sys.path`.

## Timing and accounting contract

- Signals use completed session t closes; parameters use earlier formation data.
- Fixed signed shares are sized at t, filled at t+1 close. Filters and exits
  obey the same delay. Costs apply to each leg's actual turnover.
- `max_holding_sessions` counts close-to-close intervals after entry, excluding
  weekends/holidays absent from the supplied calendar. This replaces the old
  `max_holding_days` configuration and CSV field names. `days_held` in trade
  output remains a compatibility alias for `sessions_held`.
- Batch test windows start flat, mark open trades daily, suppress final-close
  entries, and liquidate at their predeclared endpoint with exit costs.
- `replay_fills` in `pairs_research.execution` independently reports partial or
  absent fills, residual quantities, unpaired exposure, fees, fill-price P&L,
  legging P&L and execution shortfall. It never changes benchmark strategy P&L.
- The benchmark uses adjusted daily price units and assumes simultaneous full
  fills. Broker prices, corporate actions, auction rules and actual fill
  reconciliation still require a broker adapter; none is connected here.

Missing/invalid prices within a pair's active history are errors. Input rows
represent supplied trading sessions; callers must provide a complete exchange
calendar. Optional HMM evaluation uses forward filtering with frozen fitted
parameters, and attributes trades to the regime on the decision date.

## Project layout

- `pairs_trading/src/pairs_research`: installed application package.
- `pairs_trading/data`, `outputs`, `reports`: research inputs and results.
- `tests`: deterministic offline regression and integration checks.
- `experiments/stochcontrol`: isolated older execution experiment, scripts and results.

Historical outputs were regenerated under the shared engine; see the [dated rerun](pairs_trading/reports/historical_rerun_2026-09-22.md). Tests validate mechanics, not profits.

## Resumable replay

The existing t-close decision / t+1-close fill contract is preserved. Batch
`run_backtest` and durable replay now call the same `engine.process_session`.
The pure `strategy.generate_target` returns a target and reason; the order
planner fixes quantities, and the execution source supplies fills. SQLite
commits the resulting signals, orders, fills, marked positions and checkpoint
in a single transaction.

Replay a single pair from a long-format `date,ticker,adj_close` price CSV:

```sh
uv run --frozen pairs-replay --database research.sqlite --run-id energy-v1 \
  --prices pairs_trading/data/etf_price_history.csv \
  --ticker-a BNO --ticker-b XOP --formation-days 252
```

The input must contain both legs for every supplied session. Raw-price replay
records the formation sessions as flat warmup observations. Alternatively,
`--signals signals.csv` accepts the same causal signal rows as `run_backtest`,
including optional `rolling_pass` and `regime_allowed` flags. Raw replay does
not fit these external gates; gate-enabled configurations require `--signals`.
Precomputed inputs must themselves be produced causally.

Pause with `--stop-after 100`, then run the same command again without it.
Committed sessions are no-ops; later sessions continue from the saved positions,
pending order, holding count, reentry block and rolling formation history.
The source may contain the full range or just the consecutive unprocessed
suffix. Use `--config config.json` for any `BacktestConfig` parameters; explicit
CLI parameters override the file. A different config, engine/strategy/execution
version, input mode or liquidation policy requires a new `--run-id`.

Replay **preserves positions and pending orders by default**. Add
`--liquidate-at-end --end YYYY-MM-DD` to declare a fixed endpoint, or omit
`--end` to use the last supplied session. Keep that endpoint unchanged when
resuming. `--stop-after` pauses before the endpoint without liquidating.
Terminal liquidation cancels pending orders, suppresses final-close entries,
and closes holdings at the predeclared endpoint with costs. It is a runner
policy, not a signal decision. No later sessions can be appended to a liquidated
run. Batch backtests retain their previous `liquidate_at_end=True` default.

Database uniqueness constraints protect event identities, and a database
trigger rejects normal fills on or before their signal session. Each successful
session creates a marked position snapshot and advances the checkpoint only
once every related write succeeds. Restart equivalence means identical economic
and ledger contents; it does not require identical SQLite file bytes. Replaying
changed prices for an already committed date leaves the original ledger intact;
use a new run ID for corrected historical data.

The current engine uses complete simultaneous pair fills. A fill source can
supply other complete-fill prices/fees through the same workflow; asynchronous
partial fills and broker order submission remain future work. The separate
`execution.replay_fills` remains the partial-fill and legging-risk model. A
read-only adapter is documented in [docs/alpaca-dry-run.md](docs/alpaca-dry-run.md);
no external order submission is implemented.
