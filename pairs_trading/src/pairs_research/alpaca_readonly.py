"""Read-only Alpaca boundary: fixed paper/data URLs, GET only, no redirects."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import re

import requests

from .config import BacktestConfig
from .download_prices_alpaca import (
    load_dotenv_if_present,
    first_env,
    KEY_ENV_NAMES,
    SECRET_ENV_NAMES,
)

PAPER = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"
ENDPOINTS = {
    "clock": PAPER + "/v2/clock",
    "calendar": PAPER + "/v2/calendar",
    "positions": PAPER + "/v2/positions",
    "orders": PAPER + "/v2/orders",
    "account": PAPER + "/v2/account",
    "bars": DATA + "/v2/stocks/bars",
}


@dataclass(frozen=True)
class AlpacaSnapshot:
    observed_at: str
    calendar: list[dict]
    bars: dict[str, list[dict]]
    positions: list[dict]
    open_orders: list[dict]
    account: dict
    feed: str = "iex"
    adjustment: str = "raw"
    source: str = "paper_read_only"

    def to_dict(self):
        return asdict(self)


class AlpacaReadOnly:
    """There is intentionally no submit/cancel method or configurable base URL."""

    def __init__(self, api_key: str, secret: str):
        if not api_key or not secret:
            raise ValueError("Paper credentials are required")
        self._http = requests.Session()
        self._http.trust_env = False
        self._headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}

    @classmethod
    def from_environment(cls):
        load_dotenv_if_present()
        return cls(first_env(KEY_ENV_NAMES) or "", first_env(SECRET_ENV_NAMES) or "")

    def close(self):
        self._http.close()

    def _get(self, name: str, params: dict | None = None):
        # Select from a closed endpoint set. Environment/CLI URL overrides are
        # never read, and credentials can never follow an HTTP redirect.
        if name not in ENDPOINTS:
            raise ValueError("Read-only endpoint not allowed")
        response = self._http.get(
            ENDPOINTS[name], params=params, headers=self._headers, timeout=30, allow_redirects=False
        )
        if response.status_code != 200:
            # Never include response headers, bodies or request credentials.
            raise ValueError(f"Alpaca {name} read failed with HTTP {response.status_code}")
        return response.json()

    def snapshot(self, config: BacktestConfig, feed: str = "iex") -> AlpacaSnapshot:
        if feed not in {"iex", "sip"}:
            raise ValueError("Supported feeds are iex and sip")
        symbols = [config.ticker_a.upper(), config.ticker_b.upper()]
        if any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", s) for s in symbols):
            raise ValueError("Invalid equity ticker")
        clock = self._get("clock")
        now = datetime.fromisoformat(clock["timestamp"].replace("Z", "+00:00"))
        if now.tzinfo is None:
            raise ValueError("Alpaca clock must be timezone aware")
        start = (now - timedelta(days=max(60, config.formation_days * 3))).date().isoformat()
        calendar = self._get(
            "calendar", {"start": start, "end": (now + timedelta(days=10)).date().isoformat()}
        )
        bars = {s: [] for s in symbols}
        token, seen = None, set()
        while True:
            params = {
                "symbols": ",".join(symbols),
                "timeframe": "1Day",
                "start": start,
                "end": now.isoformat(),
                "feed": feed,
                "adjustment": "raw",
                "sort": "asc",
                "limit": 10000,
            }
            if token:
                params["page_token"] = token
            page = self._get("bars", params)
            for symbol, rows in page.get("bars", {}).items():
                if symbol not in bars or not isinstance(rows, list):
                    raise ValueError("Unexpected bars response")
                bars[symbol].extend(rows)
            token = page.get("next_page_token")
            if not token:
                break
            if token in seen:
                raise ValueError("Repeated Alpaca bars pagination token")
            seen.add(token)
        account = self._get("account")
        positions = self._get("positions")
        # Any nonempty page blocks the entire dedicated-pair account, so capped
        # pagination cannot hide an unresolved order behind a filtered symbol.
        orders = self._get("orders", {"status": "open", "limit": 500, "nested": "true"})
        final_clock = self._get("clock")
        finished = datetime.fromisoformat(final_clock["timestamp"].replace("Z", "+00:00"))
        if finished.tzinfo is None or not 0 <= (finished - now).total_seconds() <= 120:
            raise ValueError("Broker snapshot collection exceeded the freshness window")
        if (
            not isinstance(positions, list)
            or not isinstance(orders, list)
            or not isinstance(calendar, list)
        ):
            raise ValueError("Malformed Alpaca account/calendar response")
        return AlpacaSnapshot(
            final_clock["timestamp"], calendar, bars, positions, orders, account, feed=feed
        )
