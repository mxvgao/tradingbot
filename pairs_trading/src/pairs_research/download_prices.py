"""Download ETF price history with yfinance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_tickers(universe_csv: str | Path) -> list[str]:
    """Read tickers from a universe CSV."""
    universe = pd.read_csv(universe_csv)
    if "ticker" not in universe.columns:
        raise ValueError("Universe CSV must contain a ticker column")

    tickers = universe["ticker"].dropna().astype(str).str.upper().unique()
    return sorted(tickers)


def download_price_history(
    tickers: list[str],
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download daily adjusted close and volume for tickers."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to download prices. Install project dependencies "
            "or run: python -m pip install yfinance"
        ) from exc

    if not tickers:
        raise ValueError("No tickers provided")

    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError("Downloaded price history is empty")

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            ticker_data = raw[ticker]
        else:
            ticker_data = raw

        if "Adj Close" not in ticker_data.columns or "Volume" not in ticker_data.columns:
            continue

        frame = ticker_data[["Adj Close", "Volume"]].reset_index()
        frame.columns = ["date", "adj_close", "volume"]
        frame["ticker"] = ticker
        frames.append(frame[["date", "ticker", "adj_close", "volume"]])

    if not frames:
        raise ValueError("No usable adjusted close and volume data was downloaded")

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices


def write_price_history(
    universe_csv: str | Path,
    output_csv: str | Path,
    period: str = "5y",
    interval: str = "1d",
) -> Path:
    """Download price history for the universe and write it to CSV."""
    tickers = load_tickers(universe_csv)
    prices = download_price_history(tickers, period=period, interval=interval)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    written_path = write_price_history(
        universe_csv=base_dir / "data" / "etf_universe_seed.csv",
        output_csv=base_dir / "data" / "etf_price_history.csv",
    )
    print(f"Wrote {written_path}")
