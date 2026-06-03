"""Build a curated ETF universe for pairs-trading research.

The first pass is intentionally manual. Cointegration scans are noisy, so the
universe should encode economic common sense before any statistical testing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ETF:
    ticker: str
    name: str
    asset_class: str
    group: str
    subgroup: str
    leveraged_or_inverse: bool = False
    notes: str = ""


CORE_ETFS: tuple[ETF, ...] = (
    ETF("SPY", "SPDR S&P 500 ETF Trust", "equity", "us_equity", "large_blend"),
    ETF("IVV", "iShares Core S&P 500 ETF", "equity", "us_equity", "large_blend"),
    ETF("VOO", "Vanguard S&P 500 ETF", "equity", "us_equity", "large_blend"),
    ETF("VTI", "Vanguard Total Stock Market ETF", "equity", "us_equity", "total_market"),
    ETF("ITOT", "iShares Core S&P Total U.S. Stock Market ETF", "equity", "us_equity", "total_market"),
    ETF("SCHB", "Schwab U.S. Broad Market ETF", "equity", "us_equity", "total_market"),
    ETF("QQQ", "Invesco QQQ Trust", "equity", "us_equity", "large_growth"),
    ETF("VUG", "Vanguard Growth ETF", "equity", "us_equity", "large_growth"),
    ETF("IWF", "iShares Russell 1000 Growth ETF", "equity", "us_equity", "large_growth"),
    ETF("VTV", "Vanguard Value ETF", "equity", "us_equity", "large_value"),
    ETF("IWD", "iShares Russell 1000 Value ETF", "equity", "us_equity", "large_value"),
    ETF("IWB", "iShares Russell 1000 ETF", "equity", "us_equity", "large_blend"),
    ETF("IWM", "iShares Russell 2000 ETF", "equity", "us_equity", "small_blend"),
    ETF("VTWO", "Vanguard Russell 2000 ETF", "equity", "us_equity", "small_blend"),
    ETF("VB", "Vanguard Small-Cap ETF", "equity", "us_equity", "small_blend"),
    ETF("IJR", "iShares Core S&P Small-Cap ETF", "equity", "us_equity", "small_blend"),
    ETF("XLF", "Financial Select Sector SPDR Fund", "equity", "sector", "financials"),
    ETF("VFH", "Vanguard Financials ETF", "equity", "sector", "financials"),
    ETF("IYF", "iShares U.S. Financials ETF", "equity", "sector", "financials"),
    ETF("XLK", "Technology Select Sector SPDR Fund", "equity", "sector", "technology"),
    ETF("VGT", "Vanguard Information Technology ETF", "equity", "sector", "technology"),
    ETF("IYW", "iShares U.S. Technology ETF", "equity", "sector", "technology"),
    ETF("XLE", "Energy Select Sector SPDR Fund", "equity", "sector", "energy"),
    ETF("VDE", "Vanguard Energy ETF", "equity", "sector", "energy"),
    ETF("IYE", "iShares U.S. Energy ETF", "equity", "sector", "energy"),
    ETF("XLV", "Health Care Select Sector SPDR Fund", "equity", "sector", "healthcare"),
    ETF("VHT", "Vanguard Health Care ETF", "equity", "sector", "healthcare"),
    ETF("IYH", "iShares U.S. Healthcare ETF", "equity", "sector", "healthcare"),
    ETF("XLI", "Industrial Select Sector SPDR Fund", "equity", "sector", "industrials"),
    ETF("VIS", "Vanguard Industrials ETF", "equity", "sector", "industrials"),
    ETF("IYJ", "iShares U.S. Industrials ETF", "equity", "sector", "industrials"),
    ETF("EFA", "iShares MSCI EAFE ETF", "equity", "international_equity", "developed_ex_us"),
    ETF("IEFA", "iShares Core MSCI EAFE ETF", "equity", "international_equity", "developed_ex_us"),
    ETF("VEA", "Vanguard FTSE Developed Markets ETF", "equity", "international_equity", "developed_ex_us"),
    ETF("EEM", "iShares MSCI Emerging Markets ETF", "equity", "international_equity", "emerging_markets"),
    ETF("IEMG", "iShares Core MSCI Emerging Markets ETF", "equity", "international_equity", "emerging_markets"),
    ETF("VWO", "Vanguard FTSE Emerging Markets ETF", "equity", "international_equity", "emerging_markets"),
    ETF("AGG", "iShares Core U.S. Aggregate Bond ETF", "fixed_income", "us_bonds", "aggregate"),
    ETF("BND", "Vanguard Total Bond Market ETF", "fixed_income", "us_bonds", "aggregate"),
    ETF("SCHZ", "Schwab U.S. Aggregate Bond ETF", "fixed_income", "us_bonds", "aggregate"),
    ETF("IEF", "iShares 7-10 Year Treasury Bond ETF", "fixed_income", "us_bonds", "intermediate_treasury"),
    ETF("VGIT", "Vanguard Intermediate-Term Treasury ETF", "fixed_income", "us_bonds", "intermediate_treasury"),
    ETF("GOVT", "iShares U.S. Treasury Bond ETF", "fixed_income", "us_bonds", "treasury"),
    ETF("TLT", "iShares 20+ Year Treasury Bond ETF", "fixed_income", "us_bonds", "long_treasury"),
    ETF("VGLT", "Vanguard Long-Term Treasury ETF", "fixed_income", "us_bonds", "long_treasury"),
    ETF("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF", "fixed_income", "credit", "high_yield"),
    ETF("JNK", "SPDR Bloomberg High Yield Bond ETF", "fixed_income", "credit", "high_yield"),
    ETF("LQD", "iShares iBoxx $ Investment Grade Corporate Bond ETF", "fixed_income", "credit", "investment_grade"),
    ETF("VCIT", "Vanguard Intermediate-Term Corporate Bond ETF", "fixed_income", "credit", "investment_grade"),
    ETF("GLD", "SPDR Gold Shares", "commodity", "metals", "gold"),
    ETF("IAU", "iShares Gold Trust", "commodity", "metals", "gold"),
    ETF("SLV", "iShares Silver Trust", "commodity", "metals", "silver"),
)


EXPANDED_ETFS: tuple[ETF, ...] = (
    ETF("SPLG", "SPDR Portfolio S&P 500 ETF", "equity", "us_equity", "large_blend"),
    ETF("SCHX", "Schwab U.S. Large-Cap ETF", "equity", "us_equity", "large_blend"),
    ETF("VV", "Vanguard Large-Cap ETF", "equity", "us_equity", "large_blend"),
    ETF("MGC", "Vanguard Mega Cap ETF", "equity", "us_equity", "large_blend"),
    ETF("IWV", "iShares Russell 3000 ETF", "equity", "us_equity", "total_market"),
    ETF("SPTM", "SPDR Portfolio S&P 1500 Composite Stock Market ETF", "equity", "us_equity", "total_market"),
    ETF("SCHG", "Schwab U.S. Large-Cap Growth ETF", "equity", "us_equity", "large_growth"),
    ETF("MGK", "Vanguard Mega Cap Growth ETF", "equity", "us_equity", "large_growth"),
    ETF("SPYG", "SPDR Portfolio S&P 500 Growth ETF", "equity", "us_equity", "large_growth"),
    ETF("IUSG", "iShares Core S&P U.S. Growth ETF", "equity", "us_equity", "large_growth"),
    ETF("SCHV", "Schwab U.S. Large-Cap Value ETF", "equity", "us_equity", "large_value"),
    ETF("MGV", "Vanguard Mega Cap Value ETF", "equity", "us_equity", "large_value"),
    ETF("SPYV", "SPDR Portfolio S&P 500 Value ETF", "equity", "us_equity", "large_value"),
    ETF("IUSV", "iShares Core S&P U.S. Value ETF", "equity", "us_equity", "large_value"),
    ETF("MDY", "SPDR S&P MidCap 400 ETF Trust", "equity", "us_equity", "mid_blend"),
    ETF("IJH", "iShares Core S&P Mid-Cap ETF", "equity", "us_equity", "mid_blend"),
    ETF("VO", "Vanguard Mid-Cap ETF", "equity", "us_equity", "mid_blend"),
    ETF("IWR", "iShares Russell Mid-Cap ETF", "equity", "us_equity", "mid_blend"),
    ETF("SCHM", "Schwab U.S. Mid-Cap ETF", "equity", "us_equity", "mid_blend"),
    ETF("SCHA", "Schwab U.S. Small-Cap ETF", "equity", "us_equity", "small_blend"),
    ETF("SPSM", "SPDR Portfolio S&P 600 Small Cap ETF", "equity", "us_equity", "small_blend"),
    ETF("SLY", "SPDR S&P 600 Small Cap ETF", "equity", "us_equity", "small_blend"),
    ETF("RSP", "Invesco S&P 500 Equal Weight ETF", "equity", "us_equity", "large_equal_weight"),
    ETF("EQAL", "Invesco Russell 1000 Equal Weight ETF", "equity", "us_equity", "large_equal_weight"),
    ETF("XLC", "Communication Services Select Sector SPDR Fund", "equity", "sector", "communication_services"),
    ETF("VOX", "Vanguard Communication Services ETF", "equity", "sector", "communication_services"),
    ETF("IYZ", "iShares U.S. Telecommunications ETF", "equity", "sector", "communication_services"),
    ETF("FCOM", "Fidelity MSCI Communication Services Index ETF", "equity", "sector", "communication_services"),
    ETF("XLY", "Consumer Discretionary Select Sector SPDR Fund", "equity", "sector", "consumer_discretionary"),
    ETF("VCR", "Vanguard Consumer Discretionary ETF", "equity", "sector", "consumer_discretionary"),
    ETF("IYC", "iShares U.S. Consumer Discretionary ETF", "equity", "sector", "consumer_discretionary"),
    ETF("FDIS", "Fidelity MSCI Consumer Discretionary Index ETF", "equity", "sector", "consumer_discretionary"),
    ETF("XLP", "Consumer Staples Select Sector SPDR Fund", "equity", "sector", "consumer_staples"),
    ETF("VDC", "Vanguard Consumer Staples ETF", "equity", "sector", "consumer_staples"),
    ETF("IYK", "iShares U.S. Consumer Staples ETF", "equity", "sector", "consumer_staples"),
    ETF("FSTA", "Fidelity MSCI Consumer Staples Index ETF", "equity", "sector", "consumer_staples"),
    ETF("XLB", "Materials Select Sector SPDR Fund", "equity", "sector", "materials"),
    ETF("VAW", "Vanguard Materials ETF", "equity", "sector", "materials"),
    ETF("IYM", "iShares U.S. Basic Materials ETF", "equity", "sector", "materials"),
    ETF("FMAT", "Fidelity MSCI Materials Index ETF", "equity", "sector", "materials"),
    ETF("XLU", "Utilities Select Sector SPDR Fund", "equity", "sector", "utilities"),
    ETF("VPU", "Vanguard Utilities ETF", "equity", "sector", "utilities"),
    ETF("IDU", "iShares U.S. Utilities ETF", "equity", "sector", "utilities"),
    ETF("FUTY", "Fidelity MSCI Utilities Index ETF", "equity", "sector", "utilities"),
    ETF("XLRE", "Real Estate Select Sector SPDR Fund", "equity", "sector", "real_estate"),
    ETF("VNQ", "Vanguard Real Estate ETF", "equity", "sector", "real_estate"),
    ETF("IYR", "iShares U.S. Real Estate ETF", "equity", "sector", "real_estate"),
    ETF("SCHH", "Schwab U.S. REIT ETF", "equity", "sector", "real_estate"),
    ETF("SMH", "VanEck Semiconductor ETF", "equity", "industry", "semiconductors"),
    ETF("SOXX", "iShares Semiconductor ETF", "equity", "industry", "semiconductors"),
    ETF("XSD", "SPDR S&P Semiconductor ETF", "equity", "industry", "semiconductors"),
    ETF("PSI", "Invesco Semiconductors ETF", "equity", "industry", "semiconductors"),
    ETF("IBB", "iShares Biotechnology ETF", "equity", "industry", "biotechnology"),
    ETF("XBI", "SPDR S&P Biotech ETF", "equity", "industry", "biotechnology"),
    ETF("FBT", "First Trust NYSE Arca Biotechnology Index Fund", "equity", "industry", "biotechnology"),
    ETF("KBE", "SPDR S&P Bank ETF", "equity", "industry", "banks"),
    ETF("KRE", "SPDR S&P Regional Banking ETF", "equity", "industry", "banks"),
    ETF("IAT", "iShares U.S. Regional Banks ETF", "equity", "industry", "banks"),
    ETF("ITA", "iShares U.S. Aerospace & Defense ETF", "equity", "industry", "aerospace_defense"),
    ETF("XAR", "SPDR S&P Aerospace & Defense ETF", "equity", "industry", "aerospace_defense"),
    ETF("PPA", "Invesco Aerospace & Defense ETF", "equity", "industry", "aerospace_defense"),
    ETF("QUAL", "iShares MSCI USA Quality Factor ETF", "equity", "factor", "quality"),
    ETF("SPHQ", "Invesco S&P 500 Quality ETF", "equity", "factor", "quality"),
    ETF("DGRW", "WisdomTree U.S. Quality Dividend Growth Fund", "equity", "factor", "quality"),
    ETF("MTUM", "iShares MSCI USA Momentum Factor ETF", "equity", "factor", "momentum"),
    ETF("PDP", "Invesco Dorsey Wright Momentum ETF", "equity", "factor", "momentum"),
    ETF("SPMO", "Invesco S&P 500 Momentum ETF", "equity", "factor", "momentum"),
    ETF("USMV", "iShares MSCI USA Min Vol Factor ETF", "equity", "factor", "minimum_volatility"),
    ETF("SPLV", "Invesco S&P 500 Low Volatility ETF", "equity", "factor", "minimum_volatility"),
    ETF("EFAV", "iShares MSCI EAFE Min Vol Factor ETF", "equity", "factor", "minimum_volatility"),
    ETF("VLUE", "iShares MSCI USA Value Factor ETF", "equity", "factor", "value_factor"),
    ETF("RPV", "Invesco S&P 500 Pure Value ETF", "equity", "factor", "value_factor"),
    ETF("VBR", "Vanguard Small-Cap Value ETF", "equity", "factor", "small_value"),
    ETF("IJS", "iShares S&P Small-Cap 600 Value ETF", "equity", "factor", "small_value"),
    ETF("SLYV", "SPDR S&P 600 Small Cap Value ETF", "equity", "factor", "small_value"),
    ETF("VBK", "Vanguard Small-Cap Growth ETF", "equity", "factor", "small_growth"),
    ETF("IJT", "iShares S&P Small-Cap 600 Growth ETF", "equity", "factor", "small_growth"),
    ETF("SLYG", "SPDR S&P 600 Small Cap Growth ETF", "equity", "factor", "small_growth"),
    ETF("ACWI", "iShares MSCI ACWI ETF", "equity", "global_equity", "global"),
    ETF("VT", "Vanguard Total World Stock ETF", "equity", "global_equity", "global"),
    ETF("URTH", "iShares MSCI World ETF", "equity", "global_equity", "developed_global"),
    ETF("VEU", "Vanguard FTSE All-World ex-US ETF", "equity", "international_equity", "ex_us"),
    ETF("VXUS", "Vanguard Total International Stock ETF", "equity", "international_equity", "ex_us"),
    ETF("IXUS", "iShares Core MSCI Total International Stock ETF", "equity", "international_equity", "ex_us"),
    ETF("SCHF", "Schwab International Equity ETF", "equity", "international_equity", "developed_ex_us"),
    ETF("EWJ", "iShares MSCI Japan ETF", "equity", "single_country", "japan"),
    ETF("FLJP", "Franklin FTSE Japan ETF", "equity", "single_country", "japan"),
    ETF("DXJ", "WisdomTree Japan Hedged Equity Fund", "equity", "single_country", "japan"),
    ETF("EWG", "iShares MSCI Germany ETF", "equity", "single_country", "germany"),
    ETF("FLGR", "Franklin FTSE Germany ETF", "equity", "single_country", "germany"),
    ETF("EWU", "iShares MSCI United Kingdom ETF", "equity", "single_country", "united_kingdom"),
    ETF("FLGB", "Franklin FTSE United Kingdom ETF", "equity", "single_country", "united_kingdom"),
    ETF("EWC", "iShares MSCI Canada ETF", "equity", "single_country", "canada"),
    ETF("FLCA", "Franklin FTSE Canada ETF", "equity", "single_country", "canada"),
    ETF("EWW", "iShares MSCI Mexico ETF", "equity", "single_country", "mexico"),
    ETF("FLMX", "Franklin FTSE Mexico ETF", "equity", "single_country", "mexico"),
    ETF("FXI", "iShares China Large-Cap ETF", "equity", "single_country", "china"),
    ETF("MCHI", "iShares MSCI China ETF", "equity", "single_country", "china"),
    ETF("ASHR", "Xtrackers Harvest CSI 300 China A-Shares ETF", "equity", "single_country", "china"),
    ETF("EWZ", "iShares MSCI Brazil ETF", "equity", "single_country", "brazil"),
    ETF("FLBR", "Franklin FTSE Brazil ETF", "equity", "single_country", "brazil"),
    ETF("INDA", "iShares MSCI India ETF", "equity", "single_country", "india"),
    ETF("FLIN", "Franklin FTSE India ETF", "equity", "single_country", "india"),
    ETF("EWT", "iShares MSCI Taiwan ETF", "equity", "single_country", "taiwan"),
    ETF("FLTW", "Franklin FTSE Taiwan ETF", "equity", "single_country", "taiwan"),
    ETF("EWY", "iShares MSCI South Korea ETF", "equity", "single_country", "south_korea"),
    ETF("FLKR", "Franklin FTSE South Korea ETF", "equity", "single_country", "south_korea"),
    ETF("SHY", "iShares 1-3 Year Treasury Bond ETF", "fixed_income", "us_bonds", "short_treasury"),
    ETF("VGSH", "Vanguard Short-Term Treasury ETF", "fixed_income", "us_bonds", "short_treasury"),
    ETF("SCHO", "Schwab Short-Term U.S. Treasury ETF", "fixed_income", "us_bonds", "short_treasury"),
    ETF("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", "fixed_income", "us_bonds", "ultra_short_treasury"),
    ETF("SGOV", "iShares 0-3 Month Treasury Bond ETF", "fixed_income", "us_bonds", "ultra_short_treasury"),
    ETF("SHV", "iShares Short Treasury Bond ETF", "fixed_income", "us_bonds", "ultra_short_treasury"),
    ETF("SPTS", "SPDR Portfolio Short Term Treasury ETF", "fixed_income", "us_bonds", "short_treasury"),
    ETF("SPTI", "SPDR Portfolio Intermediate Term Treasury ETF", "fixed_income", "us_bonds", "intermediate_treasury"),
    ETF("SPTL", "SPDR Portfolio Long Term Treasury ETF", "fixed_income", "us_bonds", "long_treasury"),
    ETF("EDV", "Vanguard Extended Duration Treasury ETF", "fixed_income", "us_bonds", "extended_treasury"),
    ETF("ZROZ", "PIMCO 25+ Year Zero Coupon U.S. Treasury Index ETF", "fixed_income", "us_bonds", "extended_treasury"),
    ETF("TIP", "iShares TIPS Bond ETF", "fixed_income", "inflation_linked", "tips"),
    ETF("SCHP", "Schwab U.S. TIPS ETF", "fixed_income", "inflation_linked", "tips"),
    ETF("VTIP", "Vanguard Short-Term Inflation-Protected Securities ETF", "fixed_income", "inflation_linked", "short_tips"),
    ETF("STIP", "iShares 0-5 Year TIPS Bond ETF", "fixed_income", "inflation_linked", "short_tips"),
    ETF("VCSH", "Vanguard Short-Term Corporate Bond ETF", "fixed_income", "credit", "short_investment_grade"),
    ETF("IGSB", "iShares 1-5 Year Investment Grade Corporate Bond ETF", "fixed_income", "credit", "short_investment_grade"),
    ETF("SPSB", "SPDR Portfolio Short Term Corporate Bond ETF", "fixed_income", "credit", "short_investment_grade"),
    ETF("IGIB", "iShares 5-10 Year Investment Grade Corporate Bond ETF", "fixed_income", "credit", "investment_grade"),
    ETF("USIG", "iShares Broad USD Investment Grade Corporate Bond ETF", "fixed_income", "credit", "investment_grade"),
    ETF("SPHY", "SPDR Portfolio High Yield Bond ETF", "fixed_income", "credit", "high_yield"),
    ETF("USHY", "iShares Broad USD High Yield Corporate Bond ETF", "fixed_income", "credit", "high_yield"),
    ETF("MUB", "iShares National Muni Bond ETF", "fixed_income", "municipal", "national_muni"),
    ETF("VTEB", "Vanguard Tax-Exempt Bond ETF", "fixed_income", "municipal", "national_muni"),
    ETF("TFI", "SPDR Nuveen Bloomberg Municipal Bond ETF", "fixed_income", "municipal", "national_muni"),
    ETF("EMB", "iShares J.P. Morgan USD Emerging Markets Bond ETF", "fixed_income", "em_bonds", "usd_em_sovereign"),
    ETF("VWOB", "Vanguard Emerging Markets Government Bond ETF", "fixed_income", "em_bonds", "usd_em_sovereign"),
    ETF("PCY", "Invesco Emerging Markets Sovereign Debt ETF", "fixed_income", "em_bonds", "usd_em_sovereign"),
    ETF("DBC", "Invesco DB Commodity Index Tracking Fund", "commodity", "broad_commodities", "broad"),
    ETF("PDBC", "Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF", "commodity", "broad_commodities", "broad"),
    ETF("GSG", "iShares S&P GSCI Commodity-Indexed Trust", "commodity", "broad_commodities", "broad"),
    ETF("USO", "United States Oil Fund", "commodity", "energy", "oil"),
    ETF("BNO", "United States Brent Oil Fund", "commodity", "energy", "oil"),
    ETF("UNG", "United States Natural Gas Fund", "commodity", "energy", "natural_gas"),
)


SEED_UNIVERSE: tuple[ETF, ...] = CORE_ETFS + EXPANDED_ETFS


def build_universe() -> pd.DataFrame:
    """Return the curated ETF universe as a normalized DataFrame."""
    universe = pd.DataFrame(asdict(etf) for etf in SEED_UNIVERSE)
    universe["ticker"] = universe["ticker"].str.upper()
    universe = universe.sort_values(["asset_class", "group", "subgroup", "ticker"])
    return universe.reset_index(drop=True)


def validate_universe(universe: pd.DataFrame) -> None:
    """Fail fast on issues that would poison downstream scans."""
    required = {"ticker", "name", "asset_class", "group", "subgroup"}
    missing_columns = required.difference(universe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    duplicate_tickers = universe.loc[universe["ticker"].duplicated(), "ticker"].tolist()
    if duplicate_tickers:
        raise ValueError(f"Duplicate tickers: {duplicate_tickers}")

    blank_tickers = universe["ticker"].isna() | universe["ticker"].eq("")
    if blank_tickers.any():
        raise ValueError("Universe contains blank tickers")


def write_universe(path: str | Path) -> Path:
    """Build, validate, and write the universe CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    universe = build_universe()
    validate_universe(universe)
    universe.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parents[2] / "data" / "etf_universe_seed.csv"
    written_path = write_universe(default_path)
    print(f"Wrote {written_path}")
