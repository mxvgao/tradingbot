# ETF pairs research core PRD

Updated: 2026-09-22. Scope: preserve the established next-session contract while extracting a resumable shared session engine, before broker integration.

## Product objective

Produce repeatable ETF pair research whose decisions can be replayed without using future observations. Separate ideal strategy returns from execution outcomes. Existing saved results use the previous model and must be regenerated before making strategy conclusions.

## Decisions

- Observe the completed adjusted close at session t. Fit hedge parameters using only earlier observations.
- Fix signed share quantities using prices and hedge ratio known at t. Execute at the next supplied trading session's close, t+1. This is a next-close research benchmark; live implementation must submit before the closing-auction cutoff and record actual fills.
- Entry filters and all price-dependent exits use completed observations only. No same-close reaction to a new signal.
- Maximum holding is the number of close-to-close trading intervals after entry. An entry at Friday close has held one session at Monday close (or Tuesday if Monday is a holiday). Schedule the time exit one session in advance.
- A test window begins flat. Its first in-window close can generate an order for the next in-window close. Do not carry validation positions or pending orders into test.
- Endpoint policy belongs to the runner. Replay marks and preserves final positions/pending orders by default. When liquidation is explicitly enabled, the endpoint is fixed in run metadata in advance: cancel pending orders, suppress final-close entries, and liquidate remaining holdings there with exit costs. Batch backtests retain their previous liquidation default. Boundary liquidation is distinct from a signal exit.
- Mark every open position to each supplied close. Report gross price P&L, costs, net P&L, realized and unrealized P&L, and actual marked exposure. Final equity must reconcile to cash and marked holdings; after terminal liquidation, it must also reconcile to completed trades.
- Treat input rows as the supplied exchange-session calendar. Never silently drop a missing leg inside an active pair history. Missing/invalid prices fail validation; no forward-filled execution prices.
- Partial fills, different fill times, residual orders, fees, slippage and legging exposure belong in a separate execution ledger. The strategy benchmark assumes complete simultaneous fills.

## Acceptance criteria and implementation status

| ID | Requirement | Status |
| --- | --- | --- |
| R1 | One Python 3.12 environment, locked dependencies and CI | Complete |
| R2 | Installable pairs_research package, package imports and CLI | Complete |
| R3 | Separate stochcontrol experiment and remove duplicate run_single_path | Complete |
| R4 | Delayed entries/exits, fixed quantities and decision timestamps | Complete |
| R5 | Session holding limits, daily marks and boundary liquidation | Complete |
| R6 | Separate partial-fill and legging-risk model | Complete |
| R7 | Automated deterministic regression tests and offline integration checks | Complete |

## Acceptance coverage

Test timing using price gaps; Friday/holiday holding intervals; delayed mean-reversion and stop exits; no last-session entry; terminal costs and P&L reconciliation; empty/no-trade paths; invalid inputs; fixed signal-time sizing; causal signals and regime inference; rejected/partial/delayed fills and residual exposures. Build and install a wheel outside the source directory. Exercise a small offline walk-forward case.

## Exclusions and remaining gates

No broker connection or live orders in this phase. Adjusted daily prices are research units, not broker share prices; corporate-action mapping, borrow availability/fees, dividends, auction eligibility/cutoffs, exchange-calendar completeness and live reconciliation remain broker-phase requirements. Stored historical performance is not revalidated by code tests. Universe survivorship and selection bias also require a separate research audit.


## Implementation record

- R1: Root `pyproject.toml`, `uv.lock` and `.python-version` define Python 3.12.12 and one environment. Runtime and test dependencies are centralized; the old pairs requirements file is removed. NumPy is constrained to the tested 2.2 series. `.github/workflows/tests.yml` restores the lock, checks imports, runs tests and builds/installs the wheel.
- R2: All active modules use package imports. `pairs-research --base-dir ...` and `pairs-check-environment` are installed entry points. No source-directory or user-package path injection remains.
- R3: `experiments/stochcontrol` contains the older package, scripts, fixture and saved outputs. The main wheel excludes it. Its duplicate imports and shadowed `run_single_path` were removed while retaining the previously active implementation.
- R4/R5: `backtest_pair.py` provides delayed execution, decision/fill dates, signal-time share sizing, cost on actual turnover, daily accounting, session limits and boundary closes. `max_holding_sessions` replaces the old configuration field across callers. Empty exports retain their schemas. Input checks reject duplicate dates and missing leg prices, including sessions where both prices are missing.
- R6: `execution.py` replays explicit per-leg fill events against simultaneous full fills. It retains unfilled signed quantities and open exposure at the replay endpoint. Invalid directions, overfills, duplicate fill IDs and fills without contemporaneous marks are rejected. Empty fills represent an unfilled/rejected order; cancellation/retry policy is deferred to the broker adapter.
- R7: Deterministic tests cover the signal-to-fill lifecycle, accounting, input errors, HMM causality and actual signal/backtest/export/portfolio integration. HMM states now use forward filtering and trades are attributed to the signal-date regime. Walk-forward cash sessions are retained, selected folds are processed chronologically, and overlapping test windows are rejected to avoid double-counting P&L.

## Detailed accounting conventions

Entry cost is charged when shares fill; exit cost is charged when they are closed. Each side uses actual gross turnover multiplied by half the configured round-trip basis-point rate. Unrealized P&L includes the open trade's entry cost; realized P&L is the net P&L of completed trades. Their sum equals cumulative net P&L. Daily returns divide net daily P&L by previous equity. After an exit, a new entry decision is considered from the following session's close.

The execution ledger compares actual net P&L with full-fill gross benchmark P&L. `execution_shortfall = actual_net_pnl - benchmark_gross_pnl`; negative means worse execution. It reconciles to `legging_pnl + fill_price_pnl - fees`. Legging P&L measures incomplete holdings between marks, including opportunity cost when both legs are unfilled. Unpaired gross exposure measures holdings beyond the proportionally completed pair, rather than assuming the desired hedge is dollar-neutral. This deterministic replay does not estimate fill probabilities.

## Initial stabilization verification — 2026-09-21

- 52 automated tests pass in Python 3.12.12, including a real fitted HMM and synthetic walk-forward portfolio cases. Tests are offline and require no broker credentials.
- Critical-error lint checks pass; `git diff --check` passes.
- Frozen environment synchronization and actual runtime dependency imports pass.
- Source distribution and wheel build successfully. A separate clean temporary environment installs the wheel with locked dependencies; every module imports and the CLI works from outside the repository. `stochcontrol` is absent from that environment.
- CI configuration is added; remote CI has not been run in this local task.
- Existing saved reports are labeled legacy, and new summaries identify the execution model version. The full historical market-data pipeline has not been rerun: the downloaded price-history CSV is not present in this checkout. The mechanics are verified; earlier performance claims remain unvalidated.


## Incremental engine requirements — implemented

The next-session timing rule already existed before this extraction. The change
preserves it and makes batch and replay use one transition path:

`session source → process_session → pure target → planner / eligible fills → ledger transaction → marked position + checkpoint`

Prior-session orders fill before the current close can generate a new order.
`generate_target(history, state, config)` has no database, clock, filesystem or
broker access. It returns the target, reason and current causal observation.
The planner alone creates signed quantities and eligibility. Hold, long/short
entry, mean reversion, stop and maximum holding decisions share this path.

### Architecture and ownership

| Component | Responsibility |
| --- | --- |
| `config.py`, `models.py` | Versioned configuration and serializable state/events |
| `strategy.py` | Prior-window estimation and pure target decisions |
| `engine.py` | One `process_session` workflow; eligibility, fill accounting, marking, order planning |
| `MemoryLedger` | Fast batch adapter; no parallel backtest implementation |
| `SQLiteLedger` | Atomic durable ledger, uniqueness, checkpoints and canonical snapshots |
| `replay.py` / `pairs-replay` | Source iteration, pause/resume, optional endpoint policy |
| `execution.py` | Existing independent partial-fill and legging-risk simulation |

`run_backtest` now only validates/adapts rows, invokes the common workflow and
formats the existing output tables. Raw-price replay retains the latest
formation window in the checkpoint; precomputed replay accepts existing causal
HMM/rolling flags. Warmup raw-price sessions are recorded as flat holds. Batch
and raw replay match economically after that warmup; precomputed replay matches
the batch output directly.

### Durable schema and atomicity

Each strategy instance currently owns one pair and one capital account. `runs`
stores the pair, full configuration, engine/strategy/execution version, input
mode, liquidation endpoint and a deterministic fingerprint. Existing run IDs
reject any incompatible metadata. New IDs can coexist in the same database.

- `signals`: unique `(strategy_id, pair_id, signal_session)`; target, reason and observation.
- `orders`: unique `(strategy_id, pair_id, signal_session, order_purpose)` plus deterministic order ID, immutable order contents and lifecycle status.
- `fills`: unique `fill_id`, linked order, signed quantities, prices, fees and fill session.
- `positions`: unique `(strategy_id, pair_id, session)`; cash, holdings, marks, realized/unrealized P&L and daily accounting.
- `trades`: completed-trade accounting linked to the closing position snapshot.
- `checkpoints`: current strategy state, pending order, reentry block, holding index, formation history and last committed session.

SQLite uses foreign keys, WAL, full synchronous durability and `BEGIN IMMEDIATE`.
The transaction encompasses reading the checkpoint, checking for duplicate
processing, transition computation and all writes. The checkpoint is written
last. An exception rolls back every change for that session. Database unique
constraints, rather than only Python checks, enforce event identity. A SQL
trigger disallows ordinary fills on or before their signal session. Boundary
fills are an explicit exception only at the run's predeclared endpoint.

### Acceptance criteria and evidence

| Requirement | Verification |
| --- | --- |
| Preserve existing execution timing/economics | Frozen pre-extraction next-close fixture and original lifecycle tests |
| Signals use no future prices | Alter every price after t; through-t decision/ledger unchanged; history-order validation |
| Normal fills strictly follow their signal session | Engine eligibility checks, ledger queries, SQL timing trigger and rejected early-fill tests |
| Duplicate committed sessions are no-ops | Replay twice with no additional rows; concurrent duplicate processing commits once |
| Resume equals uninterrupted execution | Restart after every session in both endpoint modes; canonical ledger snapshots match |
| Configuration/version mismatch rejected | Config, engine, strategy, execution, input and endpoint mismatch tests |
| Failed data or write leaves checkpoint intact | Missing-leg recovery and injected failure immediately before checkpoint write |
| Terminal marking preserves live-like as-of state | Long/short final marked positions, pending entry continuation and cash/P&L reconciliation |
| Optional terminal liquidation ends flat | Long/short boundary fills, canceled last-close entries, costs and cash/trade reconciliation |
| One decision and accounting implementation | Memory batch versus SQLite output comparison; raw replay versus batch observations/economics |

Equivalence concerns economic values and ledger events, not SQLite file layout,
row storage order, WAL bytes or incidental metadata. Event IDs and serialized
state are deterministic in the current implementation, which permits stronger
snapshot equality in the restart tests.

### Replay operations and explicit limits

Use the root README for executable replay examples. `--stop-after` is a pause,
not a terminal event. `--liquidate-at-end` pins the selected final session in the
run metadata; keep this endpoint unchanged when restarting. Default as-of runs
may extend with consecutive later sessions. Liquidated runs cannot extend.
Committed historical rows are immutable on replay, including when the source
prices change; corrected data requires a new run ID.

The caller must supply a complete exchange-session stream; the engine does not
infer missing entire sessions from an exchange calendar. Missing legs on a
supplied session fail before its commit. No broker adapter is included. The
current fill-source contract requires complete pair fills, and rejects partial
batches; independent partial-fill simulation is retained. Asynchronous broker
fills, reconciliation, external order submission and a transactional outbox
remain explicit paper-trading gates. Fill providers invoked inside the session
transaction must have no external side effects.


### Extraction verification — 2026-09-22

- 81 tests pass, including all 52 original tests, pre-extraction economic fixtures,
  session-by-session restart equivalence, SQL uniqueness/timing constraints,
  transaction failure injection, concurrent duplicate sessions, future-price
  perturbation and optional terminal policies.
- Critical-error lint, frozen lock validation and whitespace checks pass.
- The rebuilt wheel imports independently of the checkout. Its installed replay
  command pauses, resumes, liquidates and replays idempotently against SQLite.
- README and example configuration document setup and replay operations. CI
  includes the replay CLI in its installed-wheel check. No broker is connected.

## Historical rerun and offline daily adapter — 2026-09-22

The replay foundation was checkpointed separately in `287c41d`. The broad-universe pipeline was rerun using that shared economic transition (with an equivalent state-copy optimization). One latest complete test fold returned −1.70%; the [dated record](../pairs_trading/reports/historical_rerun_2026-09-22.md) documents inputs, checksums and limits. Earlier implementation-record statements about missing data/report regeneration describe the initial phase and are superseded by this rerun.

A read-only Alpaca adapter and transactional outbox now exist; no account-connected validation or order submission occurred. The [adapter contract](alpaca-dry-run.md) documents the command, report, failures and operational gates. Tests cover duplicate/restart behavior, exact engine quantities, long/short and exit proposals, missing/stale data, account/position/order conflicts, auction cutoff, fractional-share rejection, and rollback of both engine and outbox on database faults. Run metadata pins the broker/account/data policy separately from the research configuration. All network reads precede the transaction; fixed GET-only paper/data endpoints cannot submit orders.

Paper submission remains deferred. Offline validation does not satisfy the successful actual dry-run gate. Existing simultaneous strategy fills remain a benchmark; broker fill reconciliation and legging accounting must precede any worker.

## Next milestone — reproducible baseline research, 2026-09-23

Freeze and tag the 81-test replay foundation as `research-engine-v1`. Keep later infrastructure and model experiments in separate commits. Add a frozen energy ETF universe and rolling-OLS experiment with an executable configuration, committed input snapshot, git/source/dependency provenance, explicit formation/validation/test windows, parameter grid, costs, seeds, execution convention, metrics and figures. Choose parameters using validation only; report the held-out test without retuning. Do not add more bot infrastructure in this milestone.
