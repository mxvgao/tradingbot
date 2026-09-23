"""Frozen-input rolling OLS experiments using the shared research engine."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
from itertools import combinations, product
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile

import numpy as np
import pandas as pd

from .backtest_pair import BacktestConfig, compute_walk_forward_signals, run_backtest
from .config import ENGINE_VERSION, STRATEGY_VERSION
from .models import encode

CONVENTION = "signal_close_t_fill_close_t_plus_1"
GRID_KEYS = {"formation_days", "entry_z", "exit_z", "max_holding_sessions", "stop_z"}


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_inputs(config_path):
    """Reject changed snapshots and incomplete session grids before any output."""
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    if cfg["schema_version"] != 1 or cfg["model"] != "rolling_ols_log_prices":
        raise ValueError("Unsupported experiment schema/model")
    expected_execution = {
        "convention": CONVENTION,
        "liquidate_at_end": True,
        "start_each_window_flat": True,
        "sizing": "fixed_signal_time_fractional_shares",
        "price_basis": "adjusted_research_units",
        "fill_model": "simultaneous_complete_pair",
    }
    if cfg["execution"] != expected_execution:
        raise ValueError("This baseline requires the declared shared next-close execution contract")
    if cfg["selection"] != {
        "metric": "total_return",
        "direction": "maximize",
        "scope": "per_pair",
        "tie_break": "canonical_parameter_json_ascending",
        "minimum_validation_trades": 0,
    }:
        raise ValueError("Unsupported validation selection policy")
    paths = {}
    for key in ("universe", "data"):
        paths[key] = (config_path.parent / cfg[key]["path"]).resolve()
        if checksum(paths[key]) != cfg[key]["sha256"]:
            raise ValueError(f"{key} checksum mismatch; create a new experiment for changed inputs")
    universe = pd.read_csv(paths["universe"])
    symbols = universe["ticker"].tolist()
    if (
        len(symbols) < 2
        or len(set(symbols)) != len(symbols)
        or any(not isinstance(s, str) or not s or s != s.upper() for s in symbols)
    ):
        raise ValueError("Universe requires distinct uppercase tickers")
    prices = pd.read_csv(paths["data"], float_precision="round_trip")
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    if prices["date"].isna().any() or prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Invalid or duplicate price sessions")
    if set(prices.ticker) != set(symbols):
        raise ValueError("Price symbols must exactly match the frozen universe")
    matrix = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    if not np.isfinite(matrix.to_numpy()).all() or (matrix <= 0).any().any():
        raise ValueError("Missing or invalid prices; no filling or dropping sessions is permitted")
    if any(d.tzinfo is not None or d != d.normalize() for d in matrix.index):
        raise ValueError("Price dates must be timezone-free daily sessions")
    if (
        matrix.index[0] != pd.Timestamp(cfg["data"]["start"])
        or matrix.index[-1] != pd.Timestamp(cfg["data"]["end"])
        or len(matrix) != cfg["data"]["sessions"]
    ):
        raise ValueError("Data date range/session count differs from the frozen contract")
    windows = cfg["windows"]
    if set(windows) != {"formation", "validation", "test"}:
        raise ValueError("Formation, validation and test windows are required")
    previous = None
    for name in ("formation", "validation", "test"):
        start, end = (pd.Timestamp(windows[name][k]) for k in ("start", "end"))
        if start not in matrix.index or end not in matrix.index or start > end:
            raise ValueError(f"Invalid {name} window boundaries")
        if previous is not None:
            if matrix.index.get_loc(start) != matrix.index.get_loc(previous) + 1:
                raise ValueError(
                    "Windows must be ordered, disjoint and cover consecutive supplied sessions"
                )
        previous = end
    if (
        windows["formation"]["start"] != cfg["data"]["start"]
        or windows["test"]["end"] != cfg["data"]["end"]
    ):
        raise ValueError("Windows must cover the complete frozen data range")
    grid = cfg["parameter_grid"]
    if not grid or set(grid) - GRID_KEYS or set(grid).intersection(cfg["fixed_parameters"]):
        raise ValueError("Invalid or overlapping parameter grid keys")
    if any(
        not isinstance(v, list) or not v or len({encode(x) for x in v}) != len(v)
        for v in grid.values()
    ):
        raise ValueError("Grid values must be nonempty unique lists")
    if {"ticker_a", "ticker_b"}.intersection(cfg["fixed_parameters"]):
        raise ValueError("The universe determines tickers")
    params = sorted(
        [dict(zip(sorted(grid), values)) for values in product(*(grid[k] for k in sorted(grid)))],
        key=encode,
    )
    formation_count = len(matrix.loc[windows["formation"]["start"] : windows["formation"]["end"]])
    for param in params:
        backtest = BacktestConfig(**(cfg["fixed_parameters"] | param))
        if backtest.require_regime_allowed or backtest.require_rolling_pass:
            raise ValueError("The baseline excludes external HMM/rolling gates")
        if backtest.formation_days > formation_count:
            raise ValueError("Formation window is too short for the grid")
    for value in cfg["random_seeds"].values():
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32:
            raise ValueError("Seeds must be unsigned 32-bit integers")
    return cfg, prices.sort_values(["date", "ticker"]), sorted(symbols), params, paths


def provenance(config_path, input_paths, cfg, allow_dirty):
    git_directory = config_path.parent

    def git(*args):
        return subprocess.check_output(["git", "-C", str(git_directory), *args], text=True).strip()

    root = Path(git("rev-parse", "--show-toplevel"))
    git_directory = root
    package = Path(__file__).resolve().parent
    source_hashes = {}
    for source in sorted(package.glob("*.py")):
        relative = Path("pairs_trading/src/pairs_research") / source.name
        checkout_source = root / relative
        if not checkout_source.exists() or checksum(source) != checksum(checkout_source):
            raise ValueError(
                "Installed package differs from the experiment checkout; reinstall its wheel"
            )
        source_hashes[str(relative)] = checksum(source)
    relevant = ["pairs_trading/src/pairs_research", "pyproject.toml", "uv.lock"]
    relevant += [str(p.relative_to(root)) for p in [config_path, *input_paths.values()]]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *relevant)
    if dirty and not allow_dirty:
        raise ValueError(
            "Experiment source/config/inputs are uncommitted; commit them first (or use --allow-dirty for development only)"
        )
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "baseline_tag": cfg["baseline_tag"],
        "baseline_commit": git("rev-parse", cfg["baseline_tag"] + "^{commit}"),
        "relevant_inputs_clean": not bool(dirty),
        "uncommitted_paths": dirty.splitlines(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            p: version(p) for p in ("numpy", "pandas", "statsmodels", "scipy", "matplotlib")
        },
        "uv_lock_sha256": checksum(root / "uv.lock"),
        "config_sha256": checksum(config_path),
        "source_sha256": source_hashes,
        "engine_version": ENGINE_VERSION,
        "strategy_version": STRATEGY_VERSION,
    }


def evaluate(prices, a, b, cfg, params, window):
    """The estimator receives no rows beyond the evaluation window."""
    backtest = BacktestConfig(ticker_a=a, ticker_b=b, **(cfg["fixed_parameters"] | params))
    past = prices[(prices.ticker.isin([a, b])) & (prices.date <= window["end"])]
    signals = compute_walk_forward_signals(past, backtest)
    signals = signals[signals.date.between(window["start"], window["end"])]
    expected = past.loc[past.date.between(window["start"], window["end"]), "date"].nunique()
    if signals.empty or len(signals) != expected:
        raise ValueError("Every evaluation session must have a formed signal")
    trades, daily, summary = run_backtest(signals, backtest, liquidate_at_end=True)
    return trades, daily, summary.iloc[0].to_dict(), asdict(backtest)


def save_csv(frame, path):
    frame.to_csv(
        path, index=False, float_format="%.17g", date_format="%Y-%m-%d", lineterminator="\n"
    )


def write_figure(daily, test, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "figure.dpi": 120}):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
        for pair, frame in daily.groupby("pair_id", sort=True):
            capital = test.loc[test.pair_id == pair, "initial_capital"].iloc[0]
            axes[0].plot(frame.date, (frame.equity / capital - 1) * 100, label=pair, linewidth=1)
        axes[0].set(
            title="Held-out pair tests · independent capital accounts", ylabel="Net return (%)"
        )
        axes[0].legend(ncol=5, fontsize=7)
        axes[0].grid(alpha=0.2)
        axes[1].bar(
            test.pair_id,
            test.total_return * 100,
            color=["#26806c" if r >= 0 else "#ba4b46" for r in test.total_return],
        )
        axes[1].axhline(0, color="#555555", linewidth=0.7)
        axes[1].set(
            ylabel="Test return (%)",
            title="Every predeclared pair, including losses and no-trade cases",
        )
        axes[1].tick_params(axis="x", rotation=45, labelsize=8)
        fig.savefig(output, metadata={"Software": "pairs-experiment-v1"})
        plt.close(fig)


def run_experiment(config_path, output_dir=None, *, allow_dirty=False):
    config_path = Path(config_path).resolve()
    cfg, prices, symbols, grid, input_paths = read_inputs(config_path)
    metadata = provenance(config_path, input_paths, cfg, allow_dirty)
    output = Path(output_dir).resolve() if output_dir else config_path.parent
    protected = [config_path, *input_paths.values()]
    if any(
        p.is_relative_to(output / "results")
        or p in {output / "manifest.json", output / "report.md"}
        for p in protected
    ):
        raise ValueError("Output paths would overwrite an experiment input")
    random.seed(cfg["random_seeds"]["python"])
    np.random.seed(cfg["random_seeds"]["numpy"])
    validation_rows, selected_rows, tests, daily_rows, trade_rows = [], [], [], [], []
    for a, b in combinations(symbols, 2):
        pair = f"{a}/{b}"
        candidates = []
        for params in grid:
            _, _, metrics, resolved = evaluate(
                prices, a, b, cfg, params, cfg["windows"]["validation"]
            )
            row = {"pair_id": pair, "parameter_id": encode(params), **metrics}
            if not np.isfinite(row["total_return"]):
                raise ValueError("Nonfinite validation score")
            validation_rows.append(row)
            candidates.append((params, row, resolved))
        chosen, validation, resolved = sorted(
            candidates, key=lambda c: (-c[1]["total_return"], encode(c[0]))
        )[0]
        selected_rows.append(
            {
                "pair_id": pair,
                "parameter_id": encode(chosen),
                "validation_total_return": validation["total_return"],
                "resolved_config": encode(resolved),
            }
        )
        # Test data is evaluated only after the validation-only choice is fixed.
        trades, daily, metrics, _ = evaluate(prices, a, b, cfg, chosen, cfg["windows"]["test"])
        tests.append({"pair_id": pair, "parameter_id": encode(chosen), **metrics})
        daily_rows.append(daily.assign(pair_id=pair))
        trade_rows.append(trades.assign(pair_id=pair))
    validation, selection, test = map(pd.DataFrame, (validation_rows, selected_rows, tests))
    daily = pd.concat(daily_rows, ignore_index=True)
    trades = pd.concat(trade_rows, ignore_index=True)
    # All computation and rendering finish in staging before publishing a run.
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".experiment-", dir=output) as staging:
        stage = Path(staging)
        results = stage / "results"
        results.mkdir()
        for name, frame in [
            ("validation_grid", validation),
            ("selected_parameters", selection),
            ("test_metrics", test),
            ("test_daily", daily),
            ("test_trades", trades),
        ]:
            save_csv(frame, results / f"{name}.csv")
        write_figure(daily, test, results / "test_results.png")
        windows = {
            name: {
                **window,
                "sessions": int(
                    prices.loc[
                        prices.date.between(window["start"], window["end"]), "date"
                    ].nunique()
                ),
            }
            for name, window in cfg["windows"].items()
        }
        lines = [
            f"# {cfg['experiment_id']}",
            "",
            "Rolling log-price OLS with validation-selected parameters and the shared next-session engine.",
            "",
            f"Code commit: `{metadata['git_commit']}`. Foundation: `{metadata['baseline_tag']}` (`{metadata['baseline_commit']}`).",
            f"Committed source/inputs: {metadata['relevant_inputs_clean']}. See [manifest](manifest.json) for checksums, dependencies and the complete configuration.",
            "",
            f"Universe: {', '.join(symbols)}. All {len(test)} unordered pairs were tested; {len(grid)} parameter combinations per pair.",
            f"Validation: {cfg['windows']['validation']['start']} to {cfg['windows']['validation']['end']}. Test: {cfg['windows']['test']['start']} to {cfg['windows']['test']['end']}.",
            f"Positive test returns: {int(test.total_return.gt(0).sum())}/{len(test)}; median pair return: {test.total_return.median():.2%}. These are separate pair accounts, not portfolio performance.",
            "",
            "| Pair | Formation | Entry z | Test return | Sharpe | Max drawdown | Trades |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in test.itertuples():
            lines.append(
                f"| {row.pair_id} | {row.formation_days} | {row.entry_z:g} | {row.total_return:.2%} | {row.sharpe:.2f} | {row.max_drawdown:.2%} | {row.completed_trades} |"
            )
        lines += [
            "",
            "![Held-out results](results/test_results.png)",
            "",
            "## Protocol",
            "",
            "OLS refits each session using only its prior formation window. Initial formation precedes validation; rolling estimates may use earlier validation/test closes as they become observable. Hyperparameters are chosen per pair by maximum validation net return, with deterministic canonical-JSON tie breaking. No test result changes that choice.",
            "",
            "Validation and test start flat and liquidate at their own declared endpoints. Fixed shares use signal-close prices and fill at the following supplied session close. Open trades are marked daily. Whole-pair simultaneous fills, fractional adjusted research units and the configured turnover cost are assumed; this experiment makes no broker calls.",
            "",
            f"Round-trip cost: {cfg['fixed_parameters']['round_trip_cost_bps']:g} bps (half on entry turnover, half on exit turnover). Initial capital per independent pair: ${cfg['fixed_parameters']['initial_capital']:,.0f}; entry gross budget: ${cfg['fixed_parameters']['gross_notional_per_trade']:,.0f}.",
            "",
            "## Artifacts",
            "",
            "- [Validation grid](results/validation_grid.csv)",
            "- [Selected parameters](results/selected_parameters.csv)",
            "- [Test metrics](results/test_metrics.csv)",
            "- [Daily marks](results/test_daily.csv)",
            "- [Trade ledger](results/test_trades.csv)",
            "",
            "## Limits",
            "",
        ]
        lines.extend(f"- {s}" for s in cfg["limitations"])
        (stage / "report.md").write_text("\n".join(lines) + "\n")
        manifest = {
            "manifest_version": 1,
            **metadata,
            "experiment_id": cfg["experiment_id"],
            "configuration": cfg,
            "universe": symbols,
            "pair_count": len(test),
            "parameter_combinations_per_pair": len(grid),
            "windows": windows,
            "artifacts": {
                str(p.relative_to(stage)): checksum(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (output / "results").mkdir(exist_ok=True)
        for path in sorted(results.iterdir()):
            os.replace(path, output / "results" / path.name)
        os.replace(stage / "report.md", output / "report.md")
        # Manifest is the final completion marker; it authenticates each output.
        os.replace(stage / "manifest.json", output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, help="Write a comparison rerun outside the original experiment"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development only; manifest explicitly records uncommitted inputs",
    )
    args = parser.parse_args()
    try:
        manifest = run_experiment(args.config, args.output_dir, allow_dirty=args.allow_dirty)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Experiment failed: {exc}\n")
    print(
        json.dumps(
            {
                "experiment_id": manifest["experiment_id"],
                "git_commit": manifest["git_commit"],
                "pairs": manifest["pair_count"],
                "artifacts": len(manifest["artifacts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
