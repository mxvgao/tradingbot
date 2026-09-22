"""Tools for finding correlated stock pairs and baskets."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class PairCandidate:
    """Diagnostics for a two-asset statistical-arbitrage candidate."""

    asset_1: str
    asset_2: str
    beta: float
    corr: float
    coint_pvalue: float
    adf_pvalue: float
    half_life: float
    spread_vol: float
    rolling_pass_rate: float
    score: float


@dataclass
class BasketCandidate:
    """Summary of a correlated basket candidate."""

    assets: tuple[str, ...]
    size: int
    mean_corr: float
    min_corr: float
    max_corr: float
    avg_vol: float
    first_pc_explained: float
    score: float


def _require_statsmodels():
    try:
        import statsmodels.api as sm
        from statsmodels.tsa.stattools import adfuller, coint
    except ImportError as exc:
        raise ImportError(
            "Pair cointegration diagnostics require statsmodels. "
            "Install the project with the optional/statistics dependency or run "
            "`pip install statsmodels`."
        ) from exc

    return sm, adfuller, coint


def clean_price_frame(prices: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    """Return a positive numeric price frame suitable for log-return analysis."""

    if tickers is not None:
        prices = prices[tickers]

    out = prices.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(out > 0)
    return out.dropna(how="all")


def fit_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """Regress ``y`` on ``x`` and return the hedge ratio beta."""

    sm, _, _ = _require_statsmodels()
    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X).fit()
    return float(model.params[1])


def compute_spread(y: pd.Series, x: pd.Series, beta: float) -> pd.Series:
    """Return the log-price spread ``y - beta * x``."""

    return y - beta * x


def compute_half_life(spread: pd.Series) -> float:
    """Approximate mean-reversion half-life from an AR(1)-style regression."""

    sm, _, _ = _require_statsmodels()
    s = spread.dropna()
    if len(s) < 20:
        return np.inf

    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    lagged = lagged.loc[delta.index]

    X = sm.add_constant(lagged.values)
    model = sm.OLS(delta.values, X).fit()
    b = float(model.params[1])

    if b >= 0:
        return np.inf

    half_life = -math.log(2) / b
    if not np.isfinite(half_life) or half_life <= 0:
        return np.inf
    return float(half_life)


def rolling_cointegration_pass_rate(
    y: pd.Series,
    x: pd.Series,
    window: int = 126,
    step: int = 21,
    max_pvalue: float = 0.05,
) -> float:
    """Return the fraction of rolling windows that pass an Engle-Granger test."""

    _, _, coint = _require_statsmodels()
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < window:
        return 0.0

    passes = 0
    total = 0

    for start in range(0, len(df) - window + 1, step):
        sub = df.iloc[start:start + window]
        y_sub = sub.iloc[:, 0]
        x_sub = sub.iloc[:, 1]

        try:
            _, pvalue, _ = coint(y_sub, x_sub)
        except Exception:
            continue

        total += 1
        if np.isfinite(pvalue) and pvalue < max_pvalue:
            passes += 1

    return passes / total if total > 0 else 0.0


def score_pair(
    corr: float,
    coint_pvalue: float,
    adf_pvalue: float,
    half_life: float,
    spread_vol: float,
    rolling_pass_rate: float,
) -> float:
    """Score a pair candidate. Bigger is better."""

    if not np.isfinite(half_life):
        half_life_penalty = -3.0
    else:
        half_life_penalty = -0.03 * min(half_life, 100)

    score = 0.0
    score += 2.0 * max(corr, 0.0)
    score += 3.0 * (-math.log(max(coint_pvalue, 1e-6)))
    score += 2.0 * (-math.log(max(adf_pvalue, 1e-6)))
    score += 3.0 * rolling_pass_rate
    score += 0.5 * spread_vol
    score += half_life_penalty
    return float(score)


def evaluate_pair(
    prices: pd.DataFrame,
    ticker_1: str,
    ticker_2: str,
    min_obs: int = 252,
) -> PairCandidate | None:
    """Evaluate a single pair with correlation, cointegration, and spread metrics."""

    _, adfuller, coint = _require_statsmodels()
    df = clean_price_frame(prices, [ticker_1, ticker_2]).dropna()
    if len(df) < min_obs:
        return None

    y = np.log(df[ticker_1])
    x = np.log(df[ticker_2])

    corr = float(y.diff().corr(x.diff()))
    if not np.isfinite(corr):
        return None

    try:
        beta = fit_hedge_ratio(y, x)
        spread = compute_spread(y, x, beta).dropna()

        _, coint_pvalue, _ = coint(y, x)
        adf_pvalue = float(adfuller(spread, maxlag=1, regression="c", autolag="AIC")[1])
        half_life = compute_half_life(spread)
        spread_vol = float(spread.std())
        rolling_pass_rate = rolling_cointegration_pass_rate(y, x)

        score = score_pair(
            corr=corr,
            coint_pvalue=float(coint_pvalue),
            adf_pvalue=adf_pvalue,
            half_life=half_life,
            spread_vol=spread_vol,
            rolling_pass_rate=rolling_pass_rate,
        )

        return PairCandidate(
            asset_1=ticker_1,
            asset_2=ticker_2,
            beta=float(beta),
            corr=corr,
            coint_pvalue=float(coint_pvalue),
            adf_pvalue=adf_pvalue,
            half_life=half_life,
            spread_vol=spread_vol,
            rolling_pass_rate=rolling_pass_rate,
            score=score,
        )
    except Exception:
        return None


def find_best_pairs(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    correlation_floor: float = 0.50,
    top_n: int = 25,
    min_obs: int = 252,
) -> pd.DataFrame:
    """Search across ticker pairs and return ranked pair candidates."""

    prices = clean_price_frame(prices, tickers)
    tickers = list(prices.columns) if tickers is None else tickers

    results: list[PairCandidate] = []
    log_prices = np.log(prices[tickers]).dropna(how="all")
    returns = log_prices.diff()

    for a, b in itertools.combinations(tickers, 2):
        pair_returns = returns[[a, b]].dropna()
        if len(pair_returns) < 100:
            continue

        corr = float(pair_returns[a].corr(pair_returns[b]))
        if not np.isfinite(corr) or corr < correlation_floor:
            continue

        candidate = evaluate_pair(prices, a, b, min_obs=min_obs)
        if candidate is not None:
            results.append(candidate)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame([asdict(r) for r in results])
    return out.sort_values("score", ascending=False).reset_index(drop=True).head(top_n)


def score_basket(
    mean_corr: float,
    min_corr: float,
    avg_vol: float,
    first_pc_explained: float,
    size: int,
) -> float:
    """Score a correlated basket candidate. Bigger is better."""

    score = 0.0
    score += 4.0 * mean_corr
    score += 2.0 * min_corr
    score += 1.5 * first_pc_explained
    score += 0.25 * avg_vol
    score += 0.10 * size
    return float(score)


def evaluate_basket(
    returns: pd.DataFrame,
    assets: tuple[str, ...],
    min_obs: int = 100,
) -> BasketCandidate | None:
    """Evaluate one basket using pairwise correlations and common-factor strength."""

    basket_returns = returns[list(assets)].dropna()
    if len(basket_returns) < min_obs:
        return None

    corr_matrix = basket_returns.corr()
    corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
    if corr_values.empty:
        return None

    mean_corr = float(corr_values.mean())
    min_corr = float(corr_values.min())
    max_corr = float(corr_values.max())
    avg_vol = float(basket_returns.std().mean())

    corr_array = corr_matrix.to_numpy(dtype=float)
    if not np.isfinite(corr_array).all():
        return None

    eigenvalues = np.linalg.eigvalsh(corr_array)
    first_pc_explained = float(eigenvalues[-1] / eigenvalues.sum())
    score = score_basket(mean_corr, min_corr, avg_vol, first_pc_explained, len(assets))

    return BasketCandidate(
        assets=assets,
        size=len(assets),
        mean_corr=mean_corr,
        min_corr=min_corr,
        max_corr=max_corr,
        avg_vol=avg_vol,
        first_pc_explained=first_pc_explained,
        score=score,
    )


def find_correlated_baskets(
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
    min_basket_size: int = 3,
    max_basket_size: int = 5,
    correlation_floor: float = 0.50,
    top_n: int = 25,
    min_obs: int = 100,
) -> pd.DataFrame:
    """Find ranked baskets where every pair clears a return-correlation floor.

    The input should be a wide price frame with dates on the index and one column
    per ticker. Ranking favors high average pairwise correlation, a strong common
    first principal component, and slightly larger baskets.
    """

    if min_basket_size < 2:
        raise ValueError("min_basket_size must be at least 2")
    if max_basket_size < min_basket_size:
        raise ValueError("max_basket_size must be >= min_basket_size")

    prices = clean_price_frame(prices, tickers)
    tickers = list(prices.columns) if tickers is None else tickers
    log_prices = np.log(prices[tickers]).dropna(how="all")
    returns = log_prices.diff()
    corr = returns.corr()

    candidates: list[BasketCandidate] = []
    for size in range(min_basket_size, max_basket_size + 1):
        for assets in itertools.combinations(tickers, size):
            sub_corr = corr.loc[list(assets), list(assets)]
            pair_corr = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool)).stack()
            if pair_corr.empty:
                continue
            if pair_corr.min() < correlation_floor:
                continue

            candidate = evaluate_basket(returns, assets, min_obs=min_obs)
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        return pd.DataFrame()

    out = pd.DataFrame([asdict(candidate) for candidate in candidates])
    return out.sort_values("score", ascending=False).reset_index(drop=True).head(top_n)
