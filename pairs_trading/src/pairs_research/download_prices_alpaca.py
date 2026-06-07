"""Download ETF daily bars from Alpaca Market Data."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from time import sleep

import pandas as pd
import requests


ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
KEY_ENV_NAMES = ("APCA_API_KEY_ID", "ALPACA_API_KEY", "ALPACA_API_KEY_ID")
SECRET_ENV_NAMES = ("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_SECRET_KEY")


def load_dotenv_if_present(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env without extra dependencies."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def first_env(names: tuple[str, ...]) -> str | None:
    """Return the first non-empty environment variable from names."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def alpaca_headers() -> dict[str, str]:
    """Build Alpaca auth headers from environment variables."""
    load_dotenv_if_present()
    api_key = first_env(KEY_ENV_NAMES)
    secret_key = first_env(SECRET_ENV_NAMES)
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY, or ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }


def load_tickers(universe_csv: str | Path) -> list[str]:
    """Read tickers from a universe CSV."""
    universe = pd.read_csv(universe_csv)
    if "ticker" not in universe.columns:
        raise ValueError("Universe CSV must contain a ticker column")

    tickers = universe["ticker"].dropna().astype(str).str.upper().unique()
    return sorted(tickers)


def chunks(items: list[str], size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_bars_batch(
    tickers: list[str],
    start: str,
    end: str,
    feed: str = "iex",
    adjustment: str = "all",
    limit: int = 10_000,
    pause_seconds: float = 0.25,
) -> list[dict[str, object]]:
    """Fetch daily bars for a batch of tickers, following Alpaca pagination."""
    headers = alpaca_headers()
    rows: list[dict[str, object]] = []
    next_page_token: str | None = None

    while True:
        params = {
            "symbols": ",".join(tickers),
            "timeframe": "1Day",
            "start": start,
            "end": end,
            "adjustment": adjustment,
            "feed": feed,
            "sort": "asc",
            "limit": limit,
        }
        if next_page_token:
            params["page_token"] = next_page_token

        response = requests.get(
            ALPACA_BARS_URL,
            headers=headers,
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        for ticker, bars in payload.get("bars", {}).items():
            for bar in bars:
                rows.append(
                    {
                        "date": pd.to_datetime(bar["t"]).date(),
                        "ticker": ticker.upper(),
                        "adj_close": bar["c"],
                        "volume": bar["v"],
                    }
                )

        next_page_token = payload.get("next_page_token")
        if not next_page_token:
            break
        sleep(pause_seconds)

    return rows


def download_price_history(
    tickers: list[str],
    years: int = 5,
    feed: str = "iex",
    batch_size: int = 50,
) -> pd.DataFrame:
    """Download adjusted daily close and volume from Alpaca."""
    if not tickers:
        raise ValueError("No tickers provided")

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years + 10)
    rows: list[dict[str, object]] = []

    for batch in chunks(tickers, batch_size):
        rows.extend(
            fetch_bars_batch(
                tickers=batch,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                feed=feed,
            )
        )
        sleep(0.25)

    if not rows:
        raise ValueError("No Alpaca bars returned")

    prices = pd.DataFrame(rows)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices["ticker"] = prices["ticker"].str.upper()
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def write_price_history(
    universe_csv: str | Path,
    output_csv: str | Path,
    years: int = 5,
    feed: str = "iex",
) -> Path:
    """Download Alpaca price history and write it to CSV."""
    tickers = load_tickers(universe_csv)
    prices = download_price_history(tickers, years=years, feed=feed)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    try:
        written_path = write_price_history(
            universe_csv=base_dir / "data" / "etf_universe_seed.csv",
            output_csv=base_dir / "data" / "etf_price_history.csv",
            feed=os.environ.get("ALPACA_DATA_FEED", "iex"),
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {written_path}")
