"""
Event Extractor — tick-level feature extraction around economic/fundamental events.

Produces three outputs per event occurrence:
  1. event_orderflow: VPIN-bucketed candles (pre + post release windows)
  2. event_stats: Scalar features (max return, max drawdown, volume quintiles)
  3. event_surprise: expected - realized (if user provides the data)

Usage
-----
    from CTAFlow.features.event_extractor import EventExtractor, EventExtractorConfig
    from CTAFlow.screeners.event_presets import get_events_for_ticker

    config = EventExtractorConfig(ticker="CL", data_dir="/path/to/scid")
    events = get_events_for_ticker("CL")

    # event_dates: actual release dates per event code
    event_dates = {"EIA_CRUDE_INV": [date(2024,1,3), date(2024,1,10), ...]}

    extractor = EventExtractor(config, events, event_dates)
    results = extractor.extract_all(verbose=True)

    # results["event_stats"]  -> pd.DataFrame indexed by (date, event_code)
    # results["event_orderflow"] -> dict[(date, event_code, "pre"|"post")] -> pd.DataFrame
    # results["event_surprise"] -> pd.Series or None
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base_extractor import ScidBaseExtractor
from ..screeners.event_presets import (
    EventDefinition,
    event_release_dt_for_date,
    get_events_for_ticker,
    is_matching_event_slot,
)
from .volume.vpin import VPINExtractor

try:
    from ..config import DLY_DATA_PATH
except ImportError:
    DLY_DATA_PATH = ""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EventExtractorConfig:
    """Configuration for EventExtractor."""

    ticker: str
    data_dir: str = DLY_DATA_PATH
    tz: str = "America/Chicago"

    # Event windows (relative to release time)
    pre_minutes: int = 60
    post_minutes: int = 60

    # VPIN bucketing for orderflow output
    bucket_volume: Optional[int] = None  # None = use VPINExtractor default
    vpin_window: int = 20

    # Back-month contract
    include_back_month: bool = False
    back_month_offset: int = 1

    # Fixed output length per window (pad zeros / patch adjacent buckets)
    fixed_length: int = 128

    # Contract cache for faster startup (default: .contract_cache.pkl in data_dir)
    contract_cache: Optional[str] = None

    def __post_init__(self):
        if self.contract_cache is None and self.data_dir:
            default_cache = Path(self.data_dir) / ".contract_cache.pkl"
            if default_cache.exists():
                self.contract_cache = str(default_cache)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_event_dates_from_presets(
    events: List[EventDefinition],
    start_date: date,
    end_date: date,
) -> Dict[str, List[date]]:
    """Generate synthetic event dates using event_presets rules.

    For recurring events (weekly/monthly), iterates through the date range
    and uses ``is_matching_event_slot`` to find matching dates.  This is a
    convenience for testing — production should use actual release calendars.
    """
    out: Dict[str, List[date]] = {}
    for ev in events:
        dates: List[date] = []
        d = start_date
        while d <= end_date:
            # Skip weekends
            if d.weekday() < 5 and is_matching_event_slot(d, ev):
                dates.append(d)
            d += timedelta(days=1)
        out[ev.code] = dates
    return out


# ---------------------------------------------------------------------------
# EventExtractor
# ---------------------------------------------------------------------------

class EventExtractor(ScidBaseExtractor):
    """Extract tick-level features around economic/fundamental events.

    Parameters
    ----------
    config : EventExtractorConfig
        Ticker, paths, and window configuration.
    events : list of EventDefinition
        Event types to extract (from ``event_presets.py``).
    event_dates : dict
        ``{event_code: [date, ...]}`` of **actual** release dates.
        Use ``generate_event_dates_from_presets`` for synthetic dates.
    surprise_df : pd.DataFrame, optional
        Columns ``expected``, ``realized``.  Index or columns must contain
        ``date`` and ``event_code`` so we can look up per-event surprises.
    """

    def __init__(
        self,
        config: EventExtractorConfig,
        events: List[EventDefinition],
        event_dates: Dict[str, List[date]],
        surprise_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__(
            data_dir=config.data_dir,
            ticker=config.ticker,
            tz=config.tz,
            contract_cache=config.contract_cache,
        )
        self.config = config
        self.events = {ev.code: ev for ev in events}
        self.event_dates = event_dates
        self.surprise_df = self._normalize_surprise_df(surprise_df)

        # Lazy-init VPIN extractor (shares same base data dir / ticker)
        self._vpin_ext = VPINExtractor(
            config.data_dir,
            config.ticker,
            config.tz,
            bucket_volume=config.bucket_volume or 150,
            window=config.vpin_window,
        )

    # ------------------------------------------------------------------
    # Future integration stub
    # ------------------------------------------------------------------

    @classmethod
    def from_macrosint(cls, config: EventExtractorConfig, macrosint_path: Optional[str] = None):
        """Future: auto-load event_dates and surprise_df from macrOS-Int.

        Will use:
        - EIATable('NG'/'PET') for EIA release dates + actuals
        - NASSTable / QuickStatsClient for USDA release dates + actuals
        - FRED release calendar for macro events

        For now raises NotImplementedError.
        """
        raise NotImplementedError(
            "macrOS-Int integration not yet wired. Provide event_dates dict manually. "
            "See C:\\Users\\nicho\\PyCharmProjects\\macrOS-Int\\MacrOSINT\\data\\data_tables.py"
        )

    # ------------------------------------------------------------------
    # Surprise normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_surprise_df(
        df: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """Ensure surprise_df has a (date, event_code) MultiIndex."""
        if df is None:
            return None
        out = df.copy()
        # If not already a MultiIndex, try to set from columns
        if not isinstance(out.index, pd.MultiIndex):
            if "date" in out.columns and "event_code" in out.columns:
                out["date"] = pd.to_datetime(out["date"]).dt.date
                out = out.set_index(["date", "event_code"])
            else:
                logger.warning(
                    "surprise_df must have 'date' and 'event_code' columns or MultiIndex"
                )
                return None
        return out

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_event(
        self,
        dt: date,
        event_def: EventDefinition,
    ) -> Dict[str, Any]:
        """Extract features for a single event occurrence.

        Returns
        -------
        dict with keys:
            event_stats : dict of scalar features
            event_orderflow_pre : pd.DataFrame (VPIN buckets before release)
            event_orderflow_post : pd.DataFrame (VPIN buckets after release)
            surprise : float or None1
        """
        cfg = self.config

        # 1. Resolve exact release timestamp in instrument timezone
        release_dt = event_release_dt_for_date(event_def, dt, cfg.tz)

        # 2. Build fetch window
        pre_td = pd.Timedelta(minutes=cfg.pre_minutes)
        post_td = pd.Timedelta(minutes=cfg.post_minutes)
        fetch_start = release_dt - pre_td
        fetch_end = release_dt + post_td

        # 3. Fetch ticks
        try:
            ticks = self.get_stitched_data(
                start_time=fetch_start,
                end_time=fetch_end,
                columns=["Close", "TotalVolume", "BidVolume", "AskVolume"],
            )
        except Exception as e:
            logger.warning(f"[{event_def.code}] {dt}: tick fetch failed: {e}")
            return self._empty_result(event_def.code, dt)

        if ticks.empty:
            logger.debug(f"[{event_def.code}] {dt}: no ticks in window")
            return self._empty_result(event_def.code, dt)

        # Ensure tz-aware index for splitting
        if ticks.index.tz is None:
            ticks.index = ticks.index.tz_localize("UTC")
        ticks.index = ticks.index.tz_convert(cfg.tz)

        # 4. Split at release time
        release_ts = pd.Timestamp(release_dt)
        if release_ts.tz is None:
            release_ts = release_ts.tz_localize(cfg.tz)
        else:
            release_ts = release_ts.tz_convert(cfg.tz)

        pre_ticks = ticks[ticks.index < release_ts]
        post_ticks = ticks[ticks.index >= release_ts]

        # Reference price = last close before release (or first close in window)
        ref_price = float(pre_ticks["Close"].iloc[-1]) if len(pre_ticks) > 0 else (
            float(post_ticks["Close"].iloc[0]) if len(post_ticks) > 0 else np.nan
        )

        # 5. Compute stats
        pre_stats = self._compute_window_stats(pre_ticks, ref_price, prefix="pre")
        post_stats = self._compute_window_stats(post_ticks, ref_price, prefix="post")
        stats = {**pre_stats, **post_stats, "event_code": event_def.code, "date": dt}

        # 6. VPIN orderflow
        orderflow_pre = self._compute_orderflow(pre_ticks)
        orderflow_post = self._compute_orderflow(post_ticks)

        # 7. Back-month (optional)
        if cfg.include_back_month:
            back_stats = self._extract_back_month(dt, event_def, fetch_start, fetch_end, release_ts)
            stats.update(back_stats)

        # 8. Surprise
        surprise = self._lookup_surprise(dt, event_def.code)
        stats["surprise"] = surprise

        return {
            "event_stats": stats,
            "event_orderflow_pre": orderflow_pre,
            "event_orderflow_post": orderflow_post,
            "surprise": surprise,
        }

    # ------------------------------------------------------------------
    # Window stats
    # ------------------------------------------------------------------

    def _compute_window_stats(
        self,
        ticks: pd.DataFrame,
        ref_price: float,
        prefix: str = "pre",
    ) -> dict:
        """Compute scalar features for a tick window.

        Features (all in basis points relative to ref_price):
            {prefix}_max_return: (max_high - ref) / ref * 100
            {prefix}_max_dd: (ref - min_low) / ref * 100
            {prefix}_total_volume: sum of TotalVolume
            {prefix}_vol_lowest_quintile: volume in bottom 20% of price range
            {prefix}_vol_highest_quintile: volume in top 20% of price range
        """
        empty = {
            f"{prefix}_max_return": 0.0,
            f"{prefix}_max_dd": 0.0,
            f"{prefix}_total_volume": 0.0,
            f"{prefix}_vol_lowest_quintile": 0.0,
            f"{prefix}_vol_highest_quintile": 0.0,
        }
        if ticks.empty or not np.isfinite(ref_price) or ref_price == 0:
            return empty

        close = ticks["Close"].astype(float)
        vol_col = "TotalVolume" if "TotalVolume" in ticks.columns else None

        max_price = close.max()
        min_price = close.min()

        max_return = (max_price - ref_price) / ref_price * 100.0
        max_dd = (ref_price - min_price) / ref_price * 100.0

        total_volume = float(ticks[vol_col].sum()) if vol_col else 0.0

        # Quintile volumes: split price range into 5 equal bins
        vol_lo_q = 0.0
        vol_hi_q = 0.0
        if vol_col and (max_price - min_price) > 0:
            price_range = max_price - min_price
            q_size = price_range / 5.0
            lo_upper = min_price + q_size
            hi_lower = max_price - q_size
            volumes = ticks[vol_col].astype(float).values
            prices = close.values
            vol_lo_q = float(volumes[prices <= lo_upper].sum())
            vol_hi_q = float(volumes[prices >= hi_lower].sum())

        return {
            f"{prefix}_max_return": max_return,
            f"{prefix}_max_dd": max_dd,
            f"{prefix}_total_volume": total_volume,
            f"{prefix}_vol_lowest_quintile": vol_lo_q,
            f"{prefix}_vol_highest_quintile": vol_hi_q,
        }

    # ------------------------------------------------------------------
    # Orderflow (VPIN bucketing)
    # ------------------------------------------------------------------

    def _compute_orderflow(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Volume-bucket ticks into VPIN-style orderflow DataFrame."""
        if ticks.empty:
            return self._empty_orderflow()
        try:
            raw = self._vpin_ext.calculate_vpin(
                ticks,
                bucket_volume=self.config.bucket_volume or self._vpin_ext.bucket_volume,
                window=self.config.vpin_window,
                include_sequence_features=True,
            )
        except Exception as e:
            logger.debug(f"VPIN bucketing failed: {e}")
            return self._empty_orderflow()

        if raw.empty:
            return self._empty_orderflow()

        return self._normalize_length(raw)

    # ------------------------------------------------------------------
    # Pad / patch to fixed length
    # ------------------------------------------------------------------

    # Columns summed when merging adjacent buckets
    _SUM_COLS = {"buy", "sell", "vol", "imbalance", "max_buy_run", "max_sell_run"}
    # Columns averaged (volume-weighted) when merging
    _AVG_COLS = {"close", "imb_frac", "vpin", "bucket_return", "log_duration",
                 "signed_imbalance", "vol_ratio"}
    # Binary columns — take max (any-of)
    _MAX_COLS = {"buy_dom", "sell_dom"}

    def _normalize_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pad or patch *df* to ``self.config.fixed_length`` rows.

        - **Under**: zero-pad with ``bucket_volume=0`` so the model knows.
        - **Over**: iteratively merge the two adjacent rows with the
          smallest combined volume until we hit the target length.
          ``bucket_volume`` of the merged row = sum of the two originals.
        """
        target = self.config.fixed_length

        # Add bucket_volume tracking column (original bucket vol each)
        bv = self.config.bucket_volume or self._vpin_ext.bucket_volume
        if "bucket_volume" not in df.columns:
            df = df.copy()
            df["bucket_volume"] = bv

        n = len(df)

        if n == target:
            return df.reset_index(drop=True)

        if n < target:
            return self._pad(df, target)

        return self._patch(df, target)

    def _pad(self, df: pd.DataFrame, target: int) -> pd.DataFrame:
        """Zero-pad to *target* rows.  ``bucket_volume=0`` marks padding."""
        pad_n = target - len(df)
        pad = pd.DataFrame(0.0, index=range(pad_n), columns=df.columns)
        pad["bucket_volume"] = 0.0
        out = pd.concat([df, pad], ignore_index=True)
        out["bucket"] = range(target)
        return out

    def _patch(self, df: pd.DataFrame, target: int) -> pd.DataFrame:
        """Merge adjacent bucket pairs until length == *target*.

        At each step, pick the pair whose combined volume is smallest
        (least information loss).  Merged row aggregates:
          - sum cols: buy, sell, vol, imbalance, …
          - weighted-avg cols: close, vpin, bucket_return, …
          - max cols: buy_dom, sell_dom
          - bucket_volume: sum (tracks total volume represented)
        """
        data = df.reset_index(drop=True).copy()

        while len(data) > target:
            # Pairwise combined volume
            vols = data["bucket_volume"].values
            pair_vol = vols[:-1] + vols[1:]
            merge_idx = int(np.argmin(pair_vol))

            r1 = data.iloc[merge_idx]
            r2 = data.iloc[merge_idx + 1]
            w1 = r1["bucket_volume"]
            w2 = r2["bucket_volume"]
            w_total = w1 + w2

            merged = {}
            for col in data.columns:
                if col == "bucket":
                    merged[col] = r1["bucket"]
                elif col == "bucket_volume":
                    merged[col] = w_total
                elif col in self._SUM_COLS:
                    merged[col] = r1[col] + r2[col]
                elif col in self._MAX_COLS:
                    merged[col] = max(r1[col], r2[col])
                elif col in self._AVG_COLS and w_total > 0:
                    merged[col] = (r1[col] * w1 + r2[col] * w2) / w_total
                else:
                    merged[col] = r1[col]

            # Recompute derived ratios for merged row
            if merged.get("vol", 0) > 0:
                merged["imb_frac"] = abs(merged["buy"] - merged["sell"]) / merged["vol"]
                merged["signed_imbalance"] = (merged["buy"] - merged["sell"]) / merged["vol"]

            # Drop the two rows, insert merged
            data = pd.concat([
                data.iloc[:merge_idx],
                pd.DataFrame([merged]),
                data.iloc[merge_idx + 2:],
            ], ignore_index=True)

        data["bucket"] = range(target)
        return data

    def _empty_orderflow(self) -> pd.DataFrame:
        """Return a zero-padded DataFrame of fixed_length."""
        cols = ["bucket", "buy", "sell", "vol", "close", "imbalance",
                "imb_frac", "vpin", "bucket_return", "log_duration",
                "signed_imbalance", "buy_dom", "sell_dom",
                "max_buy_run", "max_sell_run", "vol_ratio", "bucket_volume"]
        df = pd.DataFrame(0.0, index=range(self.config.fixed_length), columns=cols)
        df["bucket"] = range(self.config.fixed_length)
        return df

    # ------------------------------------------------------------------
    # Back-month
    # ------------------------------------------------------------------

    def _extract_back_month(
        self,
        dt: date,
        event_def: EventDefinition,
        fetch_start,
        fetch_end,
        release_ts: pd.Timestamp,
    ) -> dict:
        """Extract back-month stats (spread change around event)."""
        prefix = "back"
        empty = {
            f"{prefix}_pre_max_return": 0.0,
            f"{prefix}_post_max_return": 0.0,
            f"{prefix}_spread_change": 0.0,
        }
        # Back-month ticker: same base ticker, offset contract
        # ScidBaseExtractor can fetch a different ticker via get_stitched_data(ticker=...)
        # but we don't have a clean "next month" ticker resolver yet.
        # For now, log and return zeros — this is the integration seam.
        logger.debug(f"Back-month extraction not yet implemented for {dt}")
        return empty

    # ------------------------------------------------------------------
    # Surprise lookup
    # ------------------------------------------------------------------

    def _lookup_surprise(self, dt: date, event_code: str) -> Optional[float]:
        """Look up expected - realized from surprise_df."""
        if self.surprise_df is None:
            return None
        try:
            row = self.surprise_df.loc[(dt, event_code)]
            expected = float(row["expected"])
            realized = float(row["realized"])
            return expected - realized
        except (KeyError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Empty result helper
    # ------------------------------------------------------------------

    def _empty_result(self, event_code: str, dt: date) -> Dict[str, Any]:
        stats = {
            **self._compute_window_stats(pd.DataFrame(), np.nan, "pre"),
            **self._compute_window_stats(pd.DataFrame(), np.nan, "post"),
            "event_code": event_code,
            "date": dt,
            "surprise": None,
        }
        return {
            "event_stats": stats,
            "event_orderflow_pre": pd.DataFrame(),
            "event_orderflow_post": pd.DataFrame(),
            "surprise": None,
        }

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_all(
        self,
        verbose: bool = False,
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """Extract features for all events across all dates.

        Returns
        -------
        dict with keys:
            event_stats : pd.DataFrame
                Indexed by (date, event_code) with scalar features.
            event_orderflow : dict
                ``{(date, event_code, "pre"|"post"): pd.DataFrame}``
            event_surprise : pd.Series or None
                Indexed by (date, event_code).
        """
        all_stats: List[dict] = []
        all_orderflow: Dict[tuple, pd.DataFrame] = {}
        total = sum(len(dates) for dates in self.event_dates.values())
        i = 0

        for event_code, dates in self.event_dates.items():
            event_def = self.events.get(event_code)
            if event_def is None:
                logger.warning(f"Unknown event code: {event_code}")
                continue

            for dt in sorted(dates):
                i += 1
                if verbose and i % 50 == 0:
                    print(f"  [{i}/{total}] {event_code} {dt}")

                result = self.extract_event(dt, event_def)

                all_stats.append(result["event_stats"])

                if not result["event_orderflow_pre"].empty:
                    all_orderflow[(dt, event_code, "pre")] = result["event_orderflow_pre"]
                if not result["event_orderflow_post"].empty:
                    all_orderflow[(dt, event_code, "post")] = result["event_orderflow_post"]

        # Build stats DataFrame
        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df["date"] = pd.to_datetime(stats_df["date"])
            stats_df = stats_df.set_index(["date", "event_code"]).sort_index()
        else:
            stats_df = pd.DataFrame()

        # Build surprise Series
        surprise_series = None
        if stats_df is not None and "surprise" in stats_df.columns:
            s = stats_df["surprise"].dropna()
            if len(s) > 0:
                surprise_series = s

        if verbose:
            print(f"Extracted {len(all_stats)} event occurrences, "
                  f"{len(all_orderflow)} orderflow windows")

        return {
            "event_stats": stats_df,
            "event_orderflow": all_orderflow,
            "event_surprise": surprise_series,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        prefix: str = "event",
    ) -> List[str]:
        """Save extraction results to disk.

        Parameters
        ----------
        results : dict
            Output from ``extract_all()``.
        output_dir : str
            Directory to write files into (created if needed).
        prefix : str
            Filename prefix, e.g. ``"ng_release"`` produces
            ``ng_release_stats.csv``, ``ng_release_orderflow_pre.parquet``, etc.

        Returns
        -------
        list of str — paths of files written.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        written: List[str] = []

        # 1. Event stats
        stats_df = results.get("event_stats")
        if stats_df is not None and not stats_df.empty:
            fp = out_path / f"{prefix}_stats.csv"
            stats_df.to_csv(fp)
            written.append(str(fp))
            logger.info(f"Saved stats → {fp}")

        # 2. Orderflow — single parquet with date/event_code/side columns
        orderflow = results.get("event_orderflow", {})
        if orderflow:
            frames = []
            for (dt, code, side), df in orderflow.items():
                chunk = df.copy()
                chunk["date"] = pd.Timestamp(dt)
                chunk["event_code"] = code
                chunk["side"] = side
                frames.append(chunk)
            merged = pd.concat(frames, ignore_index=True)
            fp = out_path / f"{prefix}_orderflow.parquet"
            merged.to_parquet(fp, index=False)
            written.append(str(fp))
            logger.info(f"Saved {len(orderflow)} orderflow windows → {fp}")

        # 3. Surprise series
        surprise = results.get("event_surprise")
        if surprise is not None and len(surprise) > 0:
            fp = out_path / f"{prefix}_surprise.csv"
            surprise.to_csv(fp, header=True)
            written.append(str(fp))
            logger.info(f"Saved surprise → {fp}")

        return written
