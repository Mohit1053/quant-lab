"""Asset universe definitions - ticker lists and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Universe:
    """Defines an asset universe with tickers and metadata."""

    name: str
    tickers: list[str]
    benchmark: str
    description: str = ""
    sector_map: dict[str, str] = field(default_factory=dict)

    @property
    def num_assets(self) -> int:
        return len(self.tickers)


# NIFTY 50 constituents (NSE tickers for yfinance use .NS suffix)
NIFTY50_TICKERS = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BPCL.NS",
    "BHARTIARTL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "ITC.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NTPC.NS",
    "NESTLEIND.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SHRIRAMFIN.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TCS.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
]


UNIVERSES = {
    "nifty50": Universe(
        name="nifty50",
        tickers=NIFTY50_TICKERS,
        benchmark="^NSEI",
        description="NIFTY 50 - Top 50 Indian large-cap equities",
    ),
    "nifty500": Universe(
        name="nifty500",
        tickers=[],  # Populated dynamically via yfinance or config override
        benchmark="^CRSLDX",
        description="NIFTY 500 - Broad Indian market (load tickers via config override)",
    ),
    "indian_market": Universe(
        name="indian_market",
        tickers=[],  # Populated dynamically via yfinance or config override
        benchmark="^NSEI",
        description="Full Indian market universe (load tickers via config override)",
    ),
}


def load_nifty500_tickers(
    cache_dir: str = "data/universe_cache",
    force_refresh: bool = False,
) -> list[str]:
    """Load NIFTY 500 tickers from NSE, with CSV caching.

    Can be used as tickers_override for get_universe("nifty500").
    """
    from quant_lab.data.sources.nse_source import (
        NSEConstituentFetcher,
        NSEFetchConfig,
    )

    config = NSEFetchConfig(
        cache_dir=cache_dir,
        cache_ttl_days=0 if force_refresh else 7,
    )
    fetcher = NSEConstituentFetcher(config)
    return fetcher.fetch_constituents("NIFTY 500")


def get_universe(name: str, tickers_override: list[str] | None = None) -> Universe:
    """Get a predefined universe by name, optionally overriding tickers."""
    if name not in UNIVERSES:
        available = ", ".join(UNIVERSES.keys())
        raise ValueError(f"Unknown universe '{name}'. Available: {available}")

    universe = UNIVERSES[name]
    if tickers_override:
        universe = Universe(
            name=universe.name,
            tickers=tickers_override,
            benchmark=universe.benchmark,
            description=universe.description,
            sector_map=universe.sector_map,
        )
    return universe
