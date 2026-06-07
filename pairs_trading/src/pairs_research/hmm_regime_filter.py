"""HMM regime filter for pair spread signals."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_PACKAGES = Path(__file__).resolve().parents[2] / ".python_packages"
if LOCAL_PACKAGES.exists() and str(LOCAL_PACKAGES) not in sys.path:
    sys.path.append(str(LOCAL_PACKAGES))


@dataclass(frozen=True)
class HMMRegimeConfig:
    n_states: int = 3
    min_state_trades: int = 1
    random_state: int = 42
    covariance_type: str = "diag"
    n_iter: int = 200


def build_hmm_features(signals: pd.DataFrame) -> pd.DataFrame:
    """Build spread-regime features from walk-forward pair signals."""
    features = signals[["date", "spread", "zscore", "hedge_ratio"]].copy()
    features["date"] = pd.to_datetime(features["date"])
    features["spread_change"] = features["spread"].diff()
    features["zscore_change"] = features["zscore"].diff()
    features["spread_vol_20d"] = features["spread_change"].rolling(20).std()
    features["abs_zscore"] = features["zscore"].abs()
    features["hedge_ratio_change_20d"] = features["hedge_ratio"].diff(20).abs()
    return features.dropna().reset_index(drop=True)


def standardize_train_apply(
    train_features: pd.DataFrame,
    all_features: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize features using train-window moments only."""
    mean = train_features[feature_columns].mean()
    std = train_features[feature_columns].std().replace(0, np.nan).fillna(1.0)
    train_x = ((train_features[feature_columns] - mean) / std).to_numpy(dtype=float)
    all_x = ((all_features[feature_columns] - mean) / std).to_numpy(dtype=float)
    return train_x, all_x


def fit_predict_hmm_regimes(
    signals: pd.DataFrame,
    subtrain_end: pd.Timestamp,
    config: HMMRegimeConfig = HMMRegimeConfig(),
) -> pd.DataFrame:
    """Fit HMM on subtrain features and predict regimes for all signal dates."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise ImportError(
            "hmmlearn is required for HMM regime filtering. "
            "Install it with `python -m pip install hmmlearn`."
        ) from exc

    features = build_hmm_features(signals)
    feature_columns = [
        "spread_change",
        "zscore_change",
        "spread_vol_20d",
        "abs_zscore",
        "hedge_ratio_change_20d",
    ]
    train_features = features[features["date"] <= subtrain_end].copy()
    if train_features.shape[0] < config.n_states * 20:
        output = signals[["date"]].copy()
        output["hmm_state"] = np.nan
        return output

    train_x, all_x = standardize_train_apply(train_features, features, feature_columns)
    model = GaussianHMM(
        n_components=config.n_states,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        random_state=config.random_state,
    )
    model.fit(train_x)

    features["hmm_state"] = model.predict(all_x)
    return signals[["date"]].merge(
        features[["date", "hmm_state"]],
        on="date",
        how="left",
    )


def infer_allowed_states_from_trades(
    trades: pd.DataFrame,
    regimes: pd.DataFrame,
    config: HMMRegimeConfig = HMMRegimeConfig(),
) -> set[int]:
    """Choose HMM states where subtrain trade entries had positive edge."""
    if trades.empty:
        return set()

    trade_states = trades.copy()
    trade_states["entry_date"] = pd.to_datetime(trade_states["entry_date"])
    regimes = regimes.copy()
    regimes["date"] = pd.to_datetime(regimes["date"])
    trade_states = trade_states.merge(
        regimes.rename(columns={"date": "entry_date"}),
        on="entry_date",
        how="left",
    ).dropna(subset=["hmm_state"])
    if trade_states.empty:
        return set()

    state_stats = trade_states.groupby("hmm_state").agg(
        trades=("net_pnl_bps", "size"),
        avg_net_pnl_bps=("net_pnl_bps", "mean"),
    )
    allowed = state_stats[
        state_stats["trades"].ge(config.min_state_trades)
        & state_stats["avg_net_pnl_bps"].gt(0)
    ].index
    return {int(state) for state in allowed}


def add_regime_allowed(
    signals: pd.DataFrame,
    regimes: pd.DataFrame,
    allowed_states: set[int],
) -> pd.DataFrame:
    """Attach HMM state and allowed-entry flag to signals."""
    output = signals.copy()
    regimes = regimes.copy()
    regimes["date"] = pd.to_datetime(regimes["date"])
    output["date"] = pd.to_datetime(output["date"])
    output = output.merge(regimes, on="date", how="left")
    output["regime_allowed"] = output["hmm_state"].isin(allowed_states).fillna(False)
    return output
