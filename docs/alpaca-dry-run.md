# Read-only daily reconciliation (offline validated)

`pairs-alpaca-dry-run` fetches completed raw daily bars, the exchange calendar, account status, paper positions and open orders. It calls the shared session engine, compares expected current holdings with the broker, and records the exact next-session intents plus a JSON report. Only fixed paper/data hosts and GET requests are implemented; redirects and URL overrides are disabled. There is no order-submission or cancellation method.

```sh
uv run --frozen pairs-alpaca-dry-run --database paper-dry.sqlite \
  --run-id paper-dry-v1 --config docs/replay-config.example.json \
  --feed iex --max-gross-exposure 10000 --report daily-report.json
```

A future account-connected check needs `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in an ignored `.env`. Credentials are not needed for `--snapshot snapshot.json`, which reads an `AlpacaSnapshot` fixture and uses its recorded clock. Fixtures are labeled `offline_fixture` and never qualify as operational paper-account evidence. All validation in this change used offline fixtures; no account was accessed.

The JSON report contains the signal/reason, broker/expected/desired positions, exact required orders, approved proposals, marked gross exposure, bar timestamps, completed close, and failures. Exit status 2 means blocked or failed. Blocked reports persist for diagnosis but create no new intents or engine checkpoints. Previously recorded intents remain immutable audit records.

SQLite commits engine sessions, both outbox legs and the successful report in one outer transaction. Unique proposal IDs prevent duplicates across repeats/restarts. Each intent has `dry_run_only` status, its signal date, future execution close, and closing-auction submission window. Database errors propagate; they cannot commit one leg or an advanced engine checkpoint alone. Network reads all finish before this transaction.

Freshness follows the exchange calendar, including weekends/early closes, with a 20-minute delay after the completed close. Missing/invalid bars, revised committed prices, slow/stale account snapshots, outside-pair or mismatched quantities, unresolved orders, blocked accounts, insufficient buying power, or excessive desired exposure block proposals. Runs pin account, feed, source, configuration and risk policy. Use a dedicated paper account and a new run ID for a changed policy or price basis. A replay ledger cannot silently become a broker ledger.

The first invocation starts flat with a formation window ending at the latest close. Subsequent invocations must reconcile one session at a time; missed sessions require explicit investigation. Because this adapter never trades, its simulated next-session fill will normally diverge from the unchanged paper account after an entry proposal. It then fails closed; it never pretends a simulated fill occurred at the broker or automatically resets the strategy.

The existing research quantities are usually fractional. Alpaca CLS orders require whole shares, so these cases report exact `required_orders` and a sizing failure, with no outbox proposals. No rounding or execution-time change is hidden in the adapter. The fixed closing-auction window begins at 19:00 New York time after the signal session and ends ten minutes before the next calendar close; no new proposal is approved after that cutoff. See [Alpaca order rules](https://docs.alpaca.markets/us/docs/orders-at-alpaca).

A submission worker is deferred. Before it can be built/enabled: align shared whole-share sizing, verify auction access (including early closes), tradability/short borrow, IEX versus consolidated auction prices, corporate actions/raw versus adjusted units, and actual partial-fill accounting. The buying-power buffer is only a preliminary close-price check. Several successful account-connected dry runs and a separate worker design with idempotent external order IDs/recovery are required; weeks of paper operation should precede additional modeling. No offline fixture satisfies those gates. The next requested work is the reproducible baseline research experiment.
