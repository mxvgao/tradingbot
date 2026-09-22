import ast
import importlib
import pkgutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import pairs_research
from pairs_research.backtest_pair import BacktestConfig, write_backtest_outputs
from pairs_research.hmm_regime_filter import causal_hmm_states, infer_allowed_states_from_trades
from pairs_research.walk_forward_portfolio import WalkForwardConfig, backtest_pair_on_prices
from test_backtest import price_history


def test_every_package_module_imports():
    for module in pkgutil.iter_modules(pairs_research.__path__):
        importlib.import_module(f"pairs_research.{module.name}")


def test_frozen_hmm_future_observations_do_not_rewrite_states():
    model = SimpleNamespace(
        startprob_=np.array([0.5, 0.5]),
        transmat_=np.array([[0.95, 0.05], [0.05, 0.95]]),
        means_=np.array([[-1.0], [1.0]]),
        covars_=np.array([[[1.0]], [[1.0]]]),
    )
    prefix = np.array([[-2.0], [0.0], [0.1]])
    before = causal_hmm_states(model, prefix)
    after = causal_hmm_states(model, np.vstack([prefix, [[10.0], [10.0]]]))
    np.testing.assert_array_equal(before, after[: len(prefix)])


def test_regime_attribution_uses_decision_date():
    trades = pd.DataFrame(
        {"entry_signal_date": ["2026-01-05"], "entry_date": ["2026-01-06"], "net_pnl_bps": [10.0]}
    )
    regimes = pd.DataFrame({"date": ["2026-01-05", "2026-01-06"], "hmm_state": [0, 1]})
    assert infer_allowed_states_from_trades(trades, regimes) == {0}


def test_offline_signal_backtest_export_and_window(tmp_path):
    prices = price_history()
    path = tmp_path / "prices.csv"
    prices.to_csv(path, index=False)
    cfg = BacktestConfig(
        ticker_a="A", ticker_b="B", formation_days=20, entry_z=1, exit_z=0.5, max_holding_sessions=3
    )
    trade_path, daily_path, summary_path = write_backtest_outputs(path, tmp_path, cfg)
    trades, daily, summary = [pd.read_csv(p) for p in (trade_path, daily_path, summary_path)]
    assert len(trades) > 0
    assert daily.iloc[-1].position == 0
    assert np.isclose(trades.pnl_dollars.sum(), summary.total_pnl_dollars.iloc[0])
    dates = sorted(prices.date.unique())
    window_trades, window_daily, _ = backtest_pair_on_prices(
        prices,
        "A",
        "B",
        WalkForwardConfig(formation_days=20, entry_z=1),
        start_date=dates[40],
        end_date=dates[60],
        max_holding_sessions=3,
    )
    assert len(window_trades) > 0
    assert window_daily.iloc[0].position == window_daily.iloc[-1].position == 0
    assert (
        pd.to_datetime(window_trades.entry_date) > pd.to_datetime(window_trades.entry_signal_date)
    ).all()


def test_legacy_has_one_run_single_path_and_is_not_main_package():
    root = Path(__file__).resolve().parents[1]
    source = root / "experiments/stochcontrol/src/stochcontrol/backtest.py"
    tree = ast.parse(source.read_text())
    assert (
        sum(isinstance(n, ast.FunctionDef) and n.name == "run_single_path" for n in tree.body) == 1
    )
    assert not (root / "src/stochcontrol").exists()


def test_walk_forward_keeps_cash_sessions_and_orders_latest_folds(monkeypatch):
    from pairs_research import walk_forward_portfolio as wf

    monkeypatch.setattr(wf, "scan_pairs", lambda *args: pd.DataFrame())
    monkeypatch.setattr(wf, "select_fold_pairs", lambda *args: pd.DataFrame())
    cfg = WalkForwardConfig(
        train_days=40,
        validation_days=10,
        test_days=10,
        step_days=10,
        formation_days=10,
        max_folds=2,
        latest_folds_first=True,
        use_hmm_filter=False,
    )
    folds, _, _, daily, summary = wf.run_walk_forward(price_history(), pd.DataFrame(), cfg)
    assert len(folds) == 2
    assert len(daily) == 20
    assert daily.date.is_monotonic_increasing
    assert daily.equity.eq(cfg.initial_capital).all()
    assert summary.total_return.iloc[0] == 0


def test_walk_forward_pair_results_reconcile_to_portfolio(monkeypatch):
    from pairs_research import walk_forward_portfolio as wf

    selected = pd.DataFrame(
        [
            {
                "ticker_a": "A",
                "ticker_b": "B",
                "group": "test",
                "subgroup": "test",
                "selection_score": 1,
                "train_selected_entry_z": 1,
                "train_selected_exit_z": 0.5,
                "train_selected_max_holding_sessions": 3,
                "train_selected_stop_z": None,
            }
        ]
    )
    # Fix only selection; run real signals, fills, boundary closes and aggregation.
    monkeypatch.setattr(wf, "scan_pairs", lambda *args: pd.DataFrame())
    monkeypatch.setattr(wf, "select_fold_pairs", lambda *args: selected)
    cfg = WalkForwardConfig(
        train_days=40,
        validation_days=10,
        test_days=10,
        step_days=10,
        formation_days=10,
        max_folds=2,
        latest_folds_first=True,
        use_hmm_filter=False,
    )
    _, _, pairs, daily, summary = wf.run_walk_forward(price_history(), pd.DataFrame(), cfg)
    assert daily.date.is_unique and daily.date.is_monotonic_increasing
    assert daily.groupby("fold_id").tail(1).gross_exposure.eq(0).all()
    assert np.isclose(daily.net_daily_pnl.sum(), pairs.total_pnl_dollars.sum())
    assert np.isclose(summary.total_pnl_dollars.iloc[0], pairs.total_pnl_dollars.sum())
    expected = daily.net_daily_pnl / daily.equity.shift(1).fillna(cfg.initial_capital)
    np.testing.assert_allclose(daily.daily_return, expected)


def test_real_hmm_fit_and_filter_are_causal_out_of_sample():
    from pairs_research.hmm_regime_filter import HMMRegimeConfig, fit_predict_hmm_regimes

    rng = np.random.default_rng(1)
    signals = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=120),
            "spread": rng.normal(size=120),
            "zscore": rng.normal(size=120),
            "hedge_ratio": rng.normal(1, 0.1, size=120),
        }
    )
    cfg = HMMRegimeConfig(n_states=2, n_iter=20)
    before = fit_predict_hmm_regimes(signals, signals.date[89], cfg)
    signals.loc[110:, "spread"] *= 100
    after = fit_predict_hmm_regimes(signals, signals.date[89], cfg)
    pd.testing.assert_frame_equal(before.iloc[:110], after.iloc[:110])
