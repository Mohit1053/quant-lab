"""Yahoo Finance data source via yfinance."""

from __future__ import annotations

import time

import pandas as pd
import structlog
import yfinance as yf

from quant_lab.data.sources.base_source import BaseDataSource

logger = structlog.get_logger(__name__)


class YFinanceSource(BaseDataSource):
    """Fetches OHLCV data from Yahoo Finance."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        batch_size: int = 100,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for given tickers from Yahoo Finance.

        For large ticker lists (>batch_size), downloads in batches to
        avoid rate limits and memory issues, then concatenates results.
        """
        logger.info("fetching_data", num_tickers=len(tickers), start=start, end=end)

        if len(tickers) <= self.batch_size:
            raw = self._download_with_retry(tickers, start, end)
            if raw.empty:
                raise RuntimeError("No data returned from Yahoo Finance")
            df = self._reshape_to_long(raw, tickers)
        else:
            df = self._fetch_batched(tickers, start, end)

        df = self.validate_schema(df)

        logger.info(
            "fetch_complete",
            rows=len(df),
            tickers_found=df["ticker"].nunique(),
            date_range=f"{df['date'].min().date()} to {df['date'].max().date()}",
        )
        return df

    def _fetch_batched(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Download tickers in batches and concatenate results."""
        all_records = []
        n_batches = (len(tickers) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info(
                "batch_download",
                batch=batch_num,
                total_batches=n_batches,
                tickers_in_batch=len(batch),
            )

            try:
                raw = self._download_with_retry(batch, start, end)
                if not raw.empty:
                    batch_df = self._reshape_to_long(raw, batch)
                    all_records.append(batch_df)
                    logger.info(
                        "batch_complete",
                        batch=batch_num,
                        rows=len(batch_df),
                        tickers=batch_df["ticker"].nunique(),
                    )
                else:
                    logger.warning("batch_empty", batch=batch_num)
            except Exception as e:
                logger.warning(
                    "batch_failed",
                    batch=batch_num,
                    error=str(e),
                )

            # Respectful delay between batches
            if batch_num < n_batches:
                time.sleep(2)

        if not all_records:
            raise RuntimeError("No data returned from any batch")

        return pd.concat(all_records, ignore_index=True)

    def _download_with_retry(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """Download with retry logic for rate limits."""
        for attempt in range(1, self.max_retries + 1):
            try:
                data = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                if not data.empty:
                    return data
            except Exception as e:
                logger.warning(
                    "download_retry",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"Failed to download data after {self.max_retries} attempts"
        )

    def _reshape_to_long(
        self, raw: pd.DataFrame, tickers: list[str]
    ) -> pd.DataFrame:
        """Reshape yfinance multi-ticker output to standard long format."""
        records = []

        if len(tickers) == 1:
            # Single ticker: columns are just OHLCV
            ticker = tickers[0]
            df = raw.copy()
            df = df.reset_index()
            col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
            df = df.rename(columns=col_map)
            df["ticker"] = ticker
            if "adj_close" not in df.columns and "adj close" in raw.columns:
                df["adj_close"] = raw["Adj Close"].values
            elif "adj_close" not in df.columns:
                df["adj_close"] = df["close"]
            records.append(df)
        else:
            # Multi-ticker: MultiIndex columns (ticker, field)
            for ticker in tickers:
                try:
                    ticker_data = raw[ticker].copy()
                    if ticker_data.dropna(how="all").empty:
                        logger.warning("ticker_empty", ticker=ticker)
                        continue
                    ticker_data = ticker_data.reset_index()
                    col_map = {c: c.lower().replace(" ", "_") for c in ticker_data.columns}
                    ticker_data = ticker_data.rename(columns=col_map)
                    ticker_data["ticker"] = ticker
                    if "adj_close" not in ticker_data.columns:
                        ticker_data["adj_close"] = ticker_data.get("close", None)
                    records.append(ticker_data)
                except KeyError:
                    logger.warning("ticker_not_found", ticker=ticker)

        if not records:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        result = pd.concat(records, ignore_index=True)

        # Standardize column names
        rename_map = {"price": "date"}
        for old, new in rename_map.items():
            if old in result.columns and new not in result.columns:
                result = result.rename(columns={old: new})

        # Ensure required columns exist
        for col in self.REQUIRED_COLUMNS:
            if col not in result.columns:
                if col == "adj_close":
                    result[col] = result.get("close", None)
                else:
                    result[col] = None

        return result[self.REQUIRED_COLUMNS]
