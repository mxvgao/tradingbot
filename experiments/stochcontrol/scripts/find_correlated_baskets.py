"""Find correlated stock baskets from Yahoo Finance price history."""

from __future__ import annotations

import yfinance as yf

from stochcontrol import find_best_pairs, find_correlated_baskets


TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "NVDA",
    "AMD",
    "AVGO",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "XOM",
    "CVX",
]


def load_close_prices(tickers: list[str], start: str = "2021-01-01"):
    """Download adjusted close prices as a wide DataFrame."""

    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if "Close" in data:
        return data["Close"].dropna(how="all")
    return data.dropna(how="all")


def main():
    prices = load_close_prices(TICKERS)

    baskets = find_correlated_baskets(
        prices,
        min_basket_size=3,
        max_basket_size=5,
        correlation_floor=0.45,
        top_n=15,
    )

    print("\nTop correlated baskets")
    print("=" * 80)
    if baskets.empty:
        print("No baskets passed the filter.")
    else:
        view = baskets.copy()
        view["assets"] = view["assets"].apply(lambda assets: ", ".join(assets))
        print(
            view[
                [
                    "assets",
                    "size",
                    "mean_corr",
                    "min_corr",
                    "first_pc_explained",
                    "score",
                ]
            ].to_string(index=False)
        )

    print("\nTop cointegrated pairs")
    print("=" * 80)
    try:
        pairs = find_best_pairs(prices, correlation_floor=0.45, top_n=10)
    except ImportError as exc:
        print(exc)
        return

    if pairs.empty:
        print("No pairs passed the filter.")
        return

    print(
        pairs[
            [
                "asset_1",
                "asset_2",
                "corr",
                "beta",
                "coint_pvalue",
                "half_life",
                "score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
