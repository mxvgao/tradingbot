"""Build the default ETF universe for pairs-trading research.

The default universe is the larger free universe sourced from Nasdaq Trader
symbol directories and grouped with simple ETF-name heuristics.
"""

from __future__ import annotations

from pathlib import Path

from .build_large_universe import build_large_universe, write_large_universe


def write_universe(path: str | Path) -> Path:
    """Write the default large ETF universe CSV."""
    return write_large_universe(path)


if __name__ == "__main__":
    base_dir = Path.cwd() / "pairs_trading"
    written_path = write_universe(base_dir / "data" / "etf_universe_seed.csv")
    print(f"Wrote {written_path}")
