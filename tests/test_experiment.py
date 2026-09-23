import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs_research import experiment
from pairs_research.models import encode

BASELINE = (
    Path(__file__).resolve().parents[1]
    / "research/experiments/001_rolling_ols_baseline/config.json"
)


@pytest.fixture
def small_experiment(tmp_path, monkeypatch):
    cfg = json.loads(BASELINE.read_text())
    days = pd.bdate_range("2023-01-02", periods=90)
    dates = days.strftime("%Y-%m-%d")
    rng = np.random.default_rng(9)
    base = 4.5 + np.cumsum(rng.normal(0, 0.02, len(days)))
    spread = 0.025 * np.sin(np.arange(len(days)) / 2)
    frame = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["A"] * len(days) + ["B"] * len(days),
            "adj_close": np.r_[np.exp(base + spread), np.exp(base)],
        }
    )
    prices = tmp_path / "prices.csv"
    frame.to_csv(prices, index=False)
    universe = tmp_path / "universe.csv"
    universe.write_text("ticker\nA\nB\n")
    cfg["data"].update(
        path="prices.csv",
        start=dates[0],
        end=dates[-1],
        sessions=len(days),
        sha256=experiment.checksum(prices),
    )
    cfg["universe"] = {"path": "universe.csv", "sha256": experiment.checksum(universe)}
    cfg["windows"] = {
        "formation": {"start": dates[0], "end": dates[19]},
        "validation": {"start": dates[20], "end": dates[49]},
        "test": {"start": dates[50], "end": dates[-1]},
    }
    cfg["parameter_grid"] = {"formation_days": [5, 10], "entry_z": [1.5, 2.0]}
    cfg["fixed_parameters"]["max_holding_sessions"] = 3
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(
        experiment,
        "provenance",
        lambda *args: {
            "git_commit": "fixture",
            "baseline_tag": "research-engine-v1",
            "baseline_commit": "fixture-base",
            "relevant_inputs_clean": True,
        },
    )
    return config_path, cfg, frame


def update_config(path, cfg):
    path.write_text(json.dumps(cfg))


def test_repeated_experiment_has_identical_artifacts_and_accounting(small_experiment, tmp_path):
    path, cfg, _ = small_experiment
    first = experiment.run_experiment(path, tmp_path / "first")
    second = experiment.run_experiment(path, tmp_path / "second")
    assert first == second
    for relative, sha in first["artifacts"].items():
        assert experiment.checksum(tmp_path / "first" / relative) == sha
    assert first["windows"]["test"]["sessions"] == 40
    out = tmp_path / "first/results"
    trades, daily, metrics = (
        pd.read_csv(out / f"{name}.csv") for name in ("test_trades", "test_daily", "test_metrics")
    )
    assert not trades.empty
    assert (trades.entry_date > trades.entry_signal_date).all()
    ordinary = trades[trades.exit_reason != "test_boundary"]
    assert (ordinary.exit_date > ordinary.exit_signal_date).all()
    assert daily.iloc[-1].position == 0 and daily.iloc[-1].gross_exposure == 0
    assert daily.net_daily_pnl.sum() == pytest.approx(metrics.iloc[0].total_pnl_dollars)
    assert trades.pnl_dollars.sum() == pytest.approx(metrics.iloc[0].total_pnl_dollars)
    assert set(daily.date).isdisjoint(set([cfg["windows"]["validation"]["end"]]))
    selected = pd.read_csv(out / "selected_parameters.csv").iloc[0]
    grid = pd.read_csv(out / "validation_grid.csv").sort_values(
        ["total_return", "parameter_id"], ascending=[False, True]
    )
    assert selected.parameter_id == grid.iloc[0].parameter_id


def test_test_prices_cannot_change_validation_or_selected_parameters(small_experiment, tmp_path):
    path, cfg, frame = small_experiment
    experiment.run_experiment(path, tmp_path / "before")
    mask = (frame.date >= cfg["windows"]["test"]["start"]) & (frame.ticker == "A")
    frame.loc[mask, "adj_close"] *= 2
    frame.to_csv(path.parent / "prices.csv", index=False)
    cfg["data"]["sha256"] = experiment.checksum(path.parent / "prices.csv")
    update_config(path, cfg)
    experiment.run_experiment(path, tmp_path / "after")
    for name in ("validation_grid.csv", "selected_parameters.csv"):
        assert (tmp_path / "before/results" / name).read_bytes() == (
            tmp_path / "after/results" / name
        ).read_bytes()
    assert (tmp_path / "before/results/test_daily.csv").read_bytes() != (
        tmp_path / "after/results/test_daily.csv"
    ).read_bytes()


@pytest.mark.parametrize(
    "bad",
    [
        "checksum",
        "missing_leg",
        "missing_session",
        "overlap",
        "short_formation",
        "execution",
        "universe",
    ],
)
def test_invalid_inputs_leave_existing_results_intact(small_experiment, tmp_path, bad):
    path, cfg, frame = small_experiment
    if bad in {"checksum", "missing_leg", "missing_session"}:
        if bad == "missing_session":
            frame = frame[frame.date != frame.date.iloc[4]]
        else:
            frame = frame.iloc[1:]
        frame.to_csv(path.parent / "prices.csv", index=False)
        if bad != "checksum":
            cfg["data"]["sha256"] = experiment.checksum(path.parent / "prices.csv")
    elif bad == "overlap":
        cfg["windows"]["test"]["start"] = cfg["windows"]["validation"]["end"]
    elif bad == "short_formation":
        cfg["parameter_grid"]["formation_days"] = [21]
    elif bad == "execution":
        cfg["execution"]["convention"] = "same_close"
    else:
        (path.parent / "universe.csv").write_text("ticker\nA\nA\n")
        cfg["universe"]["sha256"] = experiment.checksum(path.parent / "universe.csv")
    update_config(path, cfg)
    output = tmp_path / "protected"
    output.mkdir()
    sentinel = output / "manifest.json"
    sentinel.write_text("previous results")
    with pytest.raises(ValueError):
        experiment.run_experiment(path, output)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "previous results"


def test_zero_trade_ties_are_included_and_break_deterministically(small_experiment, tmp_path):
    path, cfg, frame = small_experiment
    frame["adj_close"] = 100
    frame.to_csv(path.parent / "prices.csv", index=False)
    cfg["data"]["sha256"] = experiment.checksum(path.parent / "prices.csv")
    update_config(path, cfg)
    experiment.run_experiment(path, tmp_path / "flat")
    chosen = pd.read_csv(tmp_path / "flat/results/selected_parameters.csv").iloc[0]
    assert chosen.parameter_id == min(
        encode({"formation_days": f, "entry_z": z}) for f in [5, 10] for z in [1.5, 2.0]
    )
    metrics = pd.read_csv(tmp_path / "flat/results/test_metrics.csv")
    assert len(metrics) == 1 and metrics.iloc[0].completed_trades == 0
    assert metrics.iloc[0].total_return == 0


def test_committed_baseline_snapshot_has_complete_frozen_grid():
    cfg, prices, symbols, grid, _ = experiment.read_inputs(BASELINE)
    assert len(symbols) == 6 and len(grid) == 4
    assert len(prices) == cfg["data"]["sessions"] * 6
    assert cfg["execution"]["convention"] == experiment.CONVENTION


def test_provenance_checks_dirty_paths_from_repository_root(monkeypatch):
    root = BASELINE.parents[3]
    commands = []

    def git(command, **kwargs):
        commands.append(command)
        if command[3:] == ["rev-parse", "--show-toplevel"]:
            return str(root)
        if "status" in command:
            assert command[2] == str(root)
            assert "research/experiments/001_rolling_ols_baseline/config.json" in command
            return "?? pairs_trading/src/pairs_research/experiment.py\n"
        return "fixture-commit\n"

    monkeypatch.setattr(experiment.subprocess, "check_output", git)
    monkeypatch.setattr(experiment.platform, "platform", lambda: "fixture-platform")
    cfg, _, _, _, paths = experiment.read_inputs(BASELINE)
    with pytest.raises(ValueError, match="uncommitted"):
        experiment.provenance(BASELINE, paths, cfg, False)
    metadata = experiment.provenance(BASELINE, paths, cfg, True)
    assert metadata["relevant_inputs_clean"] is False
    assert metadata["uncommitted_paths"]
    assert metadata["config_sha256"] == experiment.checksum(BASELINE)
