"""Fetch index constituent lists from NSE India."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd
import requests
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class NSEFetchConfig:
    """Configuration for NSE fetching."""

    base_url: str = "https://www.nseindia.com"
    api_url: str = "https://www.nseindia.com/api/equity-stockIndices"
    cache_dir: str = "data/universe_cache"
    cache_ttl_days: int = 7
    max_retries: int = 3
    retry_delay: float = 2.0
    request_timeout: int = 30


class NSEConstituentFetcher:
    """Fetches index constituent tickers from NSE India website.

    NSE requires a session with cookies from the main page before
    API calls succeed. This class handles that handshake.
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.nseindia.com/",
    }

    def __init__(self, config: NSEFetchConfig | None = None):
        self.config = config or NSEFetchConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_constituents(self, index_name: str = "NIFTY 500") -> list[str]:
        """Fetch index constituents, using cache if fresh enough.

        Returns:
            Sorted list of tickers with .NS suffix for yfinance.
        """
        cache_path = self._cache_path(index_name)

        if self._is_cache_fresh(cache_path):
            logger.info(
                "using_cached_constituents",
                index=index_name,
                path=str(cache_path),
            )
            return self._load_cache(cache_path)

        try:
            tickers = self._fetch_from_nse(index_name)
            self._save_cache(tickers, cache_path, index_name)
            logger.info(
                "constituents_fetched",
                index=index_name,
                count=len(tickers),
            )
            return tickers
        except Exception as e:
            logger.warning("nse_fetch_failed", error=str(e))
            if cache_path.exists():
                logger.info("using_stale_cache", path=str(cache_path))
                return self._load_cache(cache_path)
            raise

    def _fetch_from_nse(self, index_name: str) -> list[str]:
        """Fetch constituents from NSE API with session handling."""
        session = requests.Session()
        session.headers.update(self.HEADERS)

        # Step 1: Hit main page to get cookies
        for attempt in range(self.config.max_retries):
            try:
                resp = session.get(
                    self.config.base_url,
                    timeout=self.config.request_timeout,
                )
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(self.config.retry_delay * (attempt + 1))

        # Step 2: Fetch index data from API
        time.sleep(2)  # Respectful delay for cookie propagation
        params = {"index": index_name}
        for attempt in range(self.config.max_retries):
            try:
                resp = session.get(
                    self.config.api_url,
                    params=params,
                    timeout=self.config.request_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, ValueError):
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(self.config.retry_delay * (attempt + 1))

        # Parse: data["data"] is a list of dicts with "symbol" key
        stocks = data.get("data", [])
        symbols = []
        for stock in stocks:
            symbol = stock.get("symbol", "")
            if symbol and symbol != index_name.replace(" ", ""):
                symbols.append(f"{symbol}.NS")

        if not symbols:
            raise ValueError(
                f"No constituents found for {index_name}. "
                f"API response keys: {list(data.keys())}"
            )

        return sorted(symbols)

    def _cache_path(self, index_name: str) -> Path:
        safe_name = index_name.lower().replace(" ", "_")
        return self.cache_dir / f"{safe_name}_constituents.csv"

    def _is_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.datetime.now() - mtime
        return age.days < self.config.cache_ttl_days

    def _load_cache(self, path: Path) -> list[str]:
        df = pd.read_csv(path)
        return df["ticker"].tolist()

    def _save_cache(
        self, tickers: list[str], path: Path, index_name: str
    ) -> None:
        df = pd.DataFrame({"ticker": tickers, "index": index_name})
        df.to_csv(path, index=False)
        logger.info("cache_saved", path=str(path), count=len(tickers))
