"""Build a larger free ETF universe from Nasdaq Trader symbol directories.

This is intentionally heuristic. It is meant to expand the research funnel for
yfinance-based screening, not to be a perfect ETF taxonomy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def read_pipe_file(url: str) -> pd.DataFrame:
    """Read a Nasdaq Trader pipe-delimited symbol file."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    lines = [
        line
        for line in response.text.splitlines()
        if line and not line.startswith("File Creation Time")
    ]
    rows = [line.split("|") for line in lines]
    header = rows[0]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)


def normalize_ticker(ticker: str) -> str:
    """Convert exchange symbol style to yfinance-compatible ticker style."""
    return ticker.strip().replace(".", "-").upper()


def classify_etf(name: str) -> tuple[str, str, str]:
    """Return asset_class, group, subgroup from ETF name keywords."""
    n = name.upper()

    if any(term in n for term in ["TREASURY", "T-BILL", "T BILL"]):
        if any(term in n for term in ["0-3", "1-3 MONTH", "BILL", "ULTRA SHORT"]):
            return "fixed_income", "us_bonds", "ultra_short_treasury"
        if any(term in n for term in ["SHORT", "1-3 YEAR", "1-5 YEAR"]):
            return "fixed_income", "us_bonds", "short_treasury"
        if any(term in n for term in ["20+", "LONG", "25+"]):
            return "fixed_income", "us_bonds", "long_treasury"
        return "fixed_income", "us_bonds", "treasury"

    if any(term in n for term in ["MUNICIPAL", "MUNI", "TAX-EXEMPT"]):
        return "fixed_income", "municipal", "national_muni"
    if "TIPS" in n or "INFLATION" in n:
        return "fixed_income", "inflation_linked", "tips"
    if "HIGH YIELD" in n or "HY CORP" in n:
        return "fixed_income", "credit", "high_yield"
    if "CORPORATE BOND" in n or "INVESTMENT GRADE" in n:
        return "fixed_income", "credit", "investment_grade"
    if "BOND" in n or "FIXED INCOME" in n:
        return "fixed_income", "bonds", "broad_bond"

    if any(term in n for term in ["GOLD", "SILVER", "METAL"]):
        if "GOLD" in n:
            return "commodity", "metals", "gold"
        if "SILVER" in n:
            return "commodity", "metals", "silver"
        return "commodity", "metals", "metals"
    if any(term in n for term in ["OIL", "NATURAL GAS", "ENERGY FUND"]):
        return "commodity", "energy", "energy_commodity"
    if "COMMOD" in n:
        return "commodity", "broad_commodities", "broad"

    sectors = {
        "TECHNOLOGY": "technology",
        "SEMICONDUCTOR": "semiconductors",
        "SOFTWARE": "software",
        "INTERNET": "internet",
        "CLOUD": "cloud",
        "CYBER": "cybersecurity",
        "ROBOTICS": "robotics",
        "ARTIFICIAL INTELLIGENCE": "artificial_intelligence",
        "FINANCIAL": "financials",
        "BANK": "banks",
        "INSURANCE": "insurance",
        "HEALTH": "healthcare",
        "BIOTECH": "biotechnology",
        "PHARMACEUTICAL": "pharmaceuticals",
        "MEDICAL DEVICE": "medical_devices",
        "ENERGY": "energy",
        "OIL & GAS": "oil_gas_equity",
        "OIL AND GAS": "oil_gas_equity",
        "EXPLORATION": "oil_gas_equity",
        "INDUSTRIAL": "industrials",
        "AEROSPACE": "aerospace_defense",
        "DEFENSE": "aerospace_defense",
        "UTILITY": "utilities",
        "UTILITIES": "utilities",
        "REAL ESTATE": "real_estate",
        "REIT": "real_estate",
        "MATERIAL": "materials",
        "MINING": "mining",
        "CONSUMER STAPLES": "consumer_staples",
        "CONSUMER DISCRETIONARY": "consumer_discretionary",
        "RETAIL": "retail",
        "COMMUNICATION": "communication_services",
    }
    for keyword, subgroup in sectors.items():
        if keyword in n:
            group = (
                "industry"
                if subgroup
                in {
                    "semiconductors",
                    "software",
                    "internet",
                    "cloud",
                    "cybersecurity",
                    "robotics",
                    "artificial_intelligence",
                    "biotechnology",
                    "pharmaceuticals",
                    "medical_devices",
                    "banks",
                    "insurance",
                    "oil_gas_equity",
                    "aerospace_defense",
                    "mining",
                    "retail",
                }
                else "sector"
            )
            return "equity", group, subgroup

    if any(term in n for term in ["JAPAN", "GERMANY", "UNITED KINGDOM", "CANADA", "CHINA", "INDIA", "BRAZIL", "KOREA", "TAIWAN", "MEXICO", "AUSTRALIA", "FRANCE", "SWITZERLAND", "SPAIN", "ITALY", "INDONESIA", "VIETNAM", "THAILAND"]):
        country_map = {
            "JAPAN": "japan",
            "GERMANY": "germany",
            "UNITED KINGDOM": "united_kingdom",
            "CANADA": "canada",
            "CHINA": "china",
            "INDIA": "india",
            "BRAZIL": "brazil",
            "KOREA": "south_korea",
            "TAIWAN": "taiwan",
            "MEXICO": "mexico",
            "AUSTRALIA": "australia",
            "FRANCE": "france",
            "SWITZERLAND": "switzerland",
            "SPAIN": "spain",
            "ITALY": "italy",
            "INDONESIA": "indonesia",
            "VIETNAM": "vietnam",
            "THAILAND": "thailand",
        }
        for keyword, subgroup in country_map.items():
            if keyword in n:
                return "equity", "single_country", subgroup

    if any(term in n for term in ["EMERGING", "EAFE", "INTERNATIONAL", "GLOBAL", "WORLD", "EX-US", "EX US"]):
        if "EMERGING" in n:
            return "equity", "international_equity", "emerging_markets"
        if "GLOBAL" in n or "WORLD" in n:
            return "equity", "global_equity", "global"
        return "equity", "international_equity", "developed_ex_us"

    if any(term in n for term in ["VALUE", "GROWTH", "QUALITY", "MOMENTUM", "LOW VOL", "MIN VOL", "DIVIDEND"]):
        if "VALUE" in n:
            return "equity", "factor", "value"
        if "GROWTH" in n:
            return "equity", "factor", "growth"
        if "QUALITY" in n:
            return "equity", "factor", "quality"
        if "MOMENTUM" in n:
            return "equity", "factor", "momentum"
        if "VOL" in n:
            return "equity", "factor", "minimum_volatility"
        return "equity", "factor", "dividend"

    if any(term in n for term in ["S&P 500", "LARGE CAP", "LARGE-CAP", "RUSSELL 1000"]):
        return "equity", "us_equity", "large_blend"
    if any(term in n for term in ["MID CAP", "MID-CAP", "S&P MIDCAP"]):
        return "equity", "us_equity", "mid_blend"
    if any(term in n for term in ["SMALL CAP", "SMALL-CAP", "RUSSELL 2000", "S&P SMALLCAP"]):
        return "equity", "us_equity", "small_blend"
    if any(term in n for term in ["TOTAL MARKET", "BROAD MARKET", "RUSSELL 3000"]):
        return "equity", "us_equity", "total_market"

    return "other", "other", "other"


def build_large_universe(max_per_subgroup: int = 60) -> pd.DataFrame:
    """Fetch ETF symbols and assign rough groups."""
    nasdaq = read_pipe_file(NASDAQ_LISTED_URL)
    nasdaq = nasdaq[nasdaq["ETF"] == "Y"].rename(
        columns={"Symbol": "ticker", "Security Name": "name"}
    )

    other = read_pipe_file(OTHER_LISTED_URL)
    other = other[other["ETF"] == "Y"].rename(
        columns={"ACT Symbol": "ticker", "Security Name": "name"}
    )

    universe = pd.concat(
        [nasdaq[["ticker", "name"]], other[["ticker", "name"]]],
        ignore_index=True,
    )
    universe["ticker"] = universe["ticker"].map(normalize_ticker)
    universe = universe.drop_duplicates("ticker")
    universe = universe[~universe["ticker"].str.contains(r"[$^/ ]", regex=True)]

    groups = universe["name"].map(classify_etf)
    universe[["asset_class", "group", "subgroup"]] = pd.DataFrame(
        groups.tolist(), index=universe.index
    )
    universe["leveraged_or_inverse"] = universe["name"].str.upper().str.contains(
        "2X|3X|ULTRA|INVERSE|BEAR|BULL|LEVERAGED|SHORT"
    )
    universe["notes"] = "nasdaq_trader_heuristic"

    universe = universe[
        (universe["asset_class"] != "other") & ~universe["leveraged_or_inverse"]
    ].copy()
    universe = universe.sort_values(["asset_class", "group", "subgroup", "ticker"])
    universe = universe.groupby(["asset_class", "group", "subgroup"]).head(max_per_subgroup)
    return universe.reset_index(drop=True)


def write_large_universe(path: str | Path, max_per_subgroup: int = 60) -> Path:
    """Write the larger universe CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    universe = build_large_universe(max_per_subgroup=max_per_subgroup)
    universe.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    written_path = write_large_universe(base_dir / "data" / "etf_universe_seed.csv")
    print(f"Wrote {written_path}")
