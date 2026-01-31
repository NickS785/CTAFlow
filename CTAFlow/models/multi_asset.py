
"""
multi_asset_momentum.py
======================

MultiAssetMomentum: DeepIDMomentum-style preprocessing + DataLoader creation
for *multiple tickers* stored in a directory tree.

This module delegates dataset/loader construction to DeepIDMomentum.get_loaders()
for each ticker, then optionally concatenates datasets across tickers.

Per-ticker artifacts (expected inside each <root_dir>/<TICKER>/)
---------------------------------------------------------------
- features.csv     (summary features)
- profiles.npz     (profile arrays)
- vpin.parquet     (sequential data)
- rasterized.npz   (rasterized data)
- target.csv       (targets)
- intraday.csv     (optional intraday bars; used if present, otherwise synthesized)

Summary schema alignment
------------------------
When training multi-asset models, tickers often have different engineered summary
feature sets. But the model needs a consistent summary feature dimension across
samples.

This module supports three strategies (see SummarySelectionConfig):

1) "exact":
   Use the strict column intersection across tickers (fast, but often small)

2) "signature":
   Intersect on *feature signatures* rather than exact names. A signature is
   (window, remainder) extracted from columns such as:
       0930_60min_deseasonalized_volume -> window=60min, remainder=deseasonalized_volume
   This lets you match the "same type and duration" features even if anchors differ.

   You can also ask it to pick the "best" common window by preference (e.g. try 240min
   across all tickers; if unavailable fall back to 120min then 60min).

3) "recompute":
   Ignore features.csv and recompute a universal summary from intraday bars using
   rolling session return/volatility features. (This uses the same concept as
   CTAFlow's session_features utilities.)

Filename resolution (prefix rules)
----------------------------------
Generic file names are used by default. If `prefix` is provided:

- ticker_in_prefix=False:
    {prefix}_{generic_name}.{ext}
    e.g. v2_features.csv

- ticker_in_prefix=True:
    {ticker}_{prefix_if_any}_{generic_name}.{ext}
    e.g. ES_v2_features.csv, ES_features.csv if prefix=None

Dataset caching (save/load once created)
----------------------------------------
Creating windowed datasets can be expensive. This module provides a dataset cache
to persist created datasets via torch.save and reload them later.

- save_dataset_cache(path, payload)
- load_dataset_cache(path, validate=True)
- get_loaders(..., use_cache=True, cache_path=..., save_cache=True)

"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import warnings

import numpy as np
import pandas as pd

import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader

# ---- import DeepIDMomentum from your project ----
try:
    from CTAFlow.models.intraday_momentum import DeepIDMomentum
except Exception:
    try:
        from intraday_momentum import DeepIDMomentum  # type: ignore
    except Exception as e:  # pragma: no cover
        DeepIDMomentum = None  # type: ignore
        _IMPORT_ERR = e

# ---- intraday CSV reader ----
try:
    from CTAFlow.data.raw_formatting.intraday_manager import read_exported_df
except Exception:
    read_exported_df = None  # type: ignore

# ---- session feature recompute utilities ----
try:
    from CTAFlow.features.session.session_features import (
        session_returns,
        session_realized_volatility,
        cumulative_session_returns,
        cumulative_session_volatility,
    )
except Exception:
    # lightweight fallback (matches the logic in CTAFlow/features/session/session_features.py)
    from datetime import time as _time
    import numpy as _np

    def _ensure_datetime_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts" not in df.columns:
                raise KeyError("DataFrame must have a DatetimeIndex or a 'ts' column")
            df = df.copy()
            df.index = pd.DatetimeIndex(df["ts"])
        if df.index.tz is None:
            df = df.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        else:
            df = df.tz_convert(tz)
        return df

    def _group_by_session(intraday_df: pd.DataFrame, session_start: _time, session_end: _time, tz: str):
        localized = _ensure_datetime_index(intraday_df, tz)
        session_df = localized.between_time(session_start, session_end, inclusive="both")
        return session_df.groupby(session_df.index.normalize())

    def session_returns(intraday_df: pd.DataFrame, price_col: str = "Close",
                        session_start: _time = _time(9, 30), session_end: _time = _time(16, 0),
                        tz: str = "America/New_York") -> pd.Series:
        grouped = _group_by_session(intraday_df, session_start, session_end, tz)
        return grouped.apply(lambda df: df[price_col].iloc[-1] / df[price_col].iloc[0] - 1.0)

    def session_realized_volatility(intraday_df: pd.DataFrame, price_col: str = "Close",
                                    session_start: _time = _time(9, 30), session_end: _time = _time(16, 0),
                                    tz: str = "America/New_York") -> pd.Series:
        grouped = _group_by_session(intraday_df, session_start, session_end, tz)
        return grouped.apply(lambda df: _np.sqrt(_np.square(df[price_col].pct_change().dropna()).sum()))

    def cumulative_session_returns(intraday_df: Optional[pd.DataFrame] = None, returns: Optional[pd.Series] = None,
                                   n_periods: Sequence[int] = (1, 5, 10), price_col: str = "Close",
                                   session_start: _time = _time(9, 30), session_end: _time = _time(16, 0),
                                   tz: str = "America/New_York") -> pd.DataFrame:
        if returns is None:
            if intraday_df is None:
                raise ValueError("Provide intraday_df or precomputed session returns")
            returns = session_returns(intraday_df, price_col, session_start, session_end, tz)
        log_returns = _np.log1p(returns)
        features = {}
        for n in n_periods:
            agg = log_returns.rolling(n).sum().shift(1)
            features[f"session_return_{n}"] = _np.expm1(agg)
        return pd.DataFrame(features, index=returns.index)

    def cumulative_session_volatility(intraday_df: Optional[pd.DataFrame] = None, volatility: Optional[pd.Series] = None,
                                      n_periods: Sequence[int] = (1, 5, 10), price_col: str = "Close",
                                      session_start: _time = _time(9, 30), session_end: _time = _time(16, 0),
                                      tz: str = "America/New_York") -> pd.DataFrame:
        if volatility is None:
            if intraday_df is None:
                raise ValueError("Provide intraday_df or precomputed session volatility")
            volatility = session_realized_volatility(intraday_df, price_col, session_start, session_end, tz)
        features = {}
        for n in n_periods:
            features[f"session_volatility_{n}"] = volatility.rolling(n).mean().shift(1)
        return pd.DataFrame(features, index=volatility.index)


# -----------------------------
# Specs / helpers
# -----------------------------

@dataclass(frozen=True)
class GenericFiles:
    """Default filenames expected inside each ticker folder."""
    summary: str = "features.csv"
    profiles: str = "profiles.npz"
    vpin: str = "vpin.parquet"
    rasterized: str = "rasterized.npz"
    target: str = "target.csv"
    # Optional intraday bars filename. Default requested: intraday.csv
    intraday: Optional[str] = "intraday.csv"


@dataclass(frozen=True)
class SummarySelectionConfig:
    """
    Configure how we make multi-ticker summary features schema-consistent.

    strategy:
      - "exact": strict intersection of columns across tickers
      - "signature": intersect on (window, remainder) signatures
      - "recompute": ignore features.csv and recompute universal session features

    prefer_windows:
      Ordered list of windows to try for signature matching. The first window that
      yields >= min_common features is selected. Example: ("240min","120min","60min")
    require_substrings:
      Keep only columns whose remainder contains all substrings (case-insensitive).
      Example: ("deseasonalized",)
    min_common:
      Minimum number of shared features required; otherwise we fall back to the
      best available window or raise if strict=True.
    strict:
      If True, raise when we cannot produce a non-empty aligned schema.
    drop_datetime_cols:
      Drop date/time columns like "Datetime" from the summary features.
    always_include:
      Columns that are kept if present per ticker (added after alignment), useful
      for ubiquitous scalars like "rv_open" or "rsv_pos_open". These still must exist
      in *all* tickers if you want a consistent schema.
    """
    strategy: str = "signature"
    prefer_windows: Tuple[str, ...] = ("240min", "120min", "60min", "1d", "5d", "10d", "20d")
    require_substrings: Tuple[str, ...] = ("deseasonalized",)
    min_common: int = 8
    strict: bool = False
    drop_datetime_cols: bool = True
    always_include: Tuple[str, ...] = ("rv_open", "rsv_pos_open", "rsv_neg_open")

def _split_name_ext(filename: str) -> Tuple[str, str]:
    p = Path(filename)
    ext = p.suffix[1:] if p.suffix.startswith(".") else p.suffix
    return p.stem, ext


def _normalize_prefix(prefix: Optional[str]) -> Optional[str]:
    if prefix is None:
        return None
    prefix = str(prefix).strip()
    return prefix if prefix else None


def _auto_detect_prefix_has_ticker(prefix: str, ticker: str) -> bool:
    pr = prefix.lower()
    tk = ticker.lower()
    return pr == tk or pr.startswith(tk + "_") or pr.startswith(tk + "-")


def _read_tabular(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, parse_dates=True, index_col=0)
    # Strip whitespace from column names (handles "Date, Time, Last" style CSVs)
    df.columns = df.columns.str.strip()
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to recover a datetime index from a "Datetime" col
        for c in ("Datetime", "datetime", "DateTime", "ts", "date", "Date"):
            if c in df.columns:
                df = df.copy()
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df = df.set_index(c)
                break
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
    return df


def _read_target(path: Union[str, Path], target_col: str = "target") -> pd.Series:
    df = _read_tabular(path)
    if target_col in df.columns:
        s = df[target_col]
    else:
        for c in ("y", "label", "labels", "ret", "return", "target"):
            if c in df.columns:
                s = df[c]
                break
        else:
            s = df.iloc[:, 0]

    # If target is a single-column DataFrame that got read, ensure Series
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def _synthesize_intraday_from_sequential(sequential_df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize a minimal intraday OHLCV DataFrame from sequential data.
    IntradayMomentum mainly needs a DatetimeIndex and a price column.
    """
    if not isinstance(sequential_df.index, pd.DatetimeIndex):
        sdf = sequential_df.copy()
        sdf.index = pd.to_datetime(sdf.index)
    else:
        sdf = sequential_df

    price_col = None
    for c in ("Close", "close", "price", "last", "Last", "px", "mid"):
        if c in sdf.columns:
            price_col = c
            break

    if price_col is None:
        out = pd.DataFrame({"Close": np.ones(len(sdf), dtype=np.float32)}, index=sdf.index)
    else:
        out = pd.DataFrame({"Close": pd.to_numeric(sdf[price_col], errors="coerce")}, index=sdf.index)
        out["Close"] = out["Close"].ffill().bfill().astype(np.float32)

    return out


class _WithMeta(Dataset):
    """Append a constant meta LongTensor to each dataset item."""
    def __init__(self, base: Dataset, meta: torch.Tensor):
        self.base = base
        self.meta = meta

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        return (*item, self.meta)


def _file_signature(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {"path": str(p), "exists": False, "mtime": None, "size": 0}
    st = p.stat()
    return {"path": str(p), "exists": True, "mtime": float(st.st_mtime), "size": int(st.st_size)}


# ---- summary signature parsing ----

_SIG_ANCHOR = re.compile(r"^(?P<anchor>\d{4})_(?P<rest>.+)$")
_SIG_WINDOW = re.compile(r"(?P<pre>.*?)(?P<window>\d+)(?P<unit>min|m|hr|h|d)(?P<post>.*)", re.IGNORECASE)

def _normalize_window(n: int, unit: str) -> str:
    u = unit.lower()
    if u in ("m", "min"):
        return f"{n}min"
    if u in ("h", "hr"):
        # convert hours to minutes? keep hour token to avoid mismatch; prefer explicit if present
        return f"{n}h"
    if u == "d":
        return f"{n}d"
    return f"{n}{u}"


def feature_signature(col: str) -> Optional[Tuple[str, str]]:
    """
    Extract a (window, remainder) signature from a feature column name.

    Examples:
      0930_60min_deseasonalized_volume -> ("60min", "deseasonalized_volume")
      dist_5d_high -> ("5d", "dist_high")
      rsv_pos_open -> None (no window)
    """
    if col is None:
        return None
    c = str(col)
    if c.lower() in ("datetime", "date", "ts"):
        return None

    # strip leading anchor
    m = _SIG_ANCHOR.match(c)
    rest = m.group("rest") if m else c

    # detect window token anywhere in rest
    m2 = re.search(r"_(\d+)\s*(min|m|hr|h|d)(?:_|$)", rest, flags=re.IGNORECASE)
    if m2:
        n = int(m2.group(1))
        unit = m2.group(2)
        window = _normalize_window(n, unit)
        # remove the _<window>_ token and normalize underscores
        remainder = re.sub(rf"_{n}\s*{unit}(_|$)", "_", rest, flags=re.IGNORECASE)
        remainder = remainder.strip("_")
        return window, remainder

    # alternative patterns like "dist_5d_high" (window in middle)
    m3 = re.search(r"_(\d+)d(?:_|$)", rest, flags=re.IGNORECASE)
    if m3:
        n = int(m3.group(1))
        window = f"{n}d"
        remainder = re.sub(rf"_{n}d(_|$)", "_", rest, flags=re.IGNORECASE).strip("_")
        return window, remainder

    return None


def _filter_by_substrings(remainder: str, required: Sequence[str]) -> bool:
    r = remainder.lower()
    return all(s.lower() in r for s in required)


# -----------------------------
# Multi-asset wrapper
# -----------------------------

class MultiAssetMomentum(DeepIDMomentum):  # type: ignore[misc]
    """
    Multi-ticker wrapper around DeepIDMomentum.get_loaders().

    This class inherits DeepIDMomentum to match your requirement, but functions as
    a multi-ticker container. It holds a DeepIDMomentum instance per ticker and
    delegates loader creation.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        target_dir: Union[str, Path],
        tickers: Optional[Sequence[str]] = None,
        *,
        generic_files: GenericFiles = GenericFiles(),
        prefix: Optional[str] = None,
        ticker_in_prefix: bool = False,
        avoid_double_ticker: bool = True,
        strict: bool = True,
        deep_kwargs: Optional[Dict[str, Any]] = None,
        # summary alignment configuration
        summary_config: SummarySelectionConfig = SummarySelectionConfig(),
    ) -> None:
        if DeepIDMomentum is None:  # pragma: no cover
            raise ImportError(
                "Could not import DeepIDMomentum. Ensure CTAFlow is on PYTHONPATH or "
                "place this file next to intraday_momentum.py."
            ) from _IMPORT_ERR

        self.root_dir = Path(root_dir).expanduser().resolve()
        self.target_dir = Path(target_dir).expanduser().resolve()

        self.generic_files = generic_files
        self.prefix = _normalize_prefix(prefix)
        self.ticker_in_prefix = bool(ticker_in_prefix)
        self.avoid_double_ticker = bool(avoid_double_ticker)
        self.strict = bool(strict)

        self._tickers = list(tickers) if tickers is not None else self.discover_tickers()
        self.deep_kwargs = dict(deep_kwargs or {})
        self.summary_config = summary_config

        self._models: Dict[str, DeepIDMomentum] = {}

        # cache for aligned summary schema
        self._aligned_summary_cols: Optional[List[str]] = None
        self._per_ticker_summary_cols: Optional[Dict[str, List[str]]] = None

    # ---------- naming / discovery ----------

    def discover_tickers(self) -> List[str]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir does not exist: {self.root_dir}")
        out: List[str] = []
        for p in sorted(self.root_dir.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                out.append(p.name)
        return out

    @property
    def tickers(self) -> List[str]:
        return list(self._tickers)

    def ticker_dir(self, ticker: str) -> Path:
        return (self.root_dir / ticker).resolve()

    def build_filename(
        self,
        ticker: str,
        generic_filename: str,
        *,
        prefix: Optional[str] = None,
        ticker_in_prefix: Optional[bool] = None,
    ) -> str:
        prefix = _normalize_prefix(prefix if prefix is not None else self.prefix)
        tip = self.ticker_in_prefix if ticker_in_prefix is None else bool(ticker_in_prefix)

        stem, ext = _split_name_ext(generic_filename)

        if prefix and "{ticker}" in prefix:
            prefix = prefix.format(ticker=ticker)

        if not prefix and not tip:
            return f"{stem}.{ext}"

        if tip:
            parts: List[str] = [ticker]
            if prefix:
                if self.avoid_double_ticker and _auto_detect_prefix_has_ticker(prefix, ticker):
                    parts = [prefix]
                else:
                    parts.append(prefix)
            parts.append(stem)
            return "_".join(parts) + f".{ext}"

        if prefix:
            return f"{prefix}_{stem}.{ext}"

        return f"{stem}.{ext}"

    def path_for(self, ticker: str, generic_filename: str, *, must_exist: bool = True) -> Path:
        tdir = self.ticker_dir(ticker)
        fn = self.build_filename(ticker, generic_filename)
        p = (tdir / fn).resolve()
        if must_exist and not p.exists():
            msg = f"Missing file for {ticker}: expected {p}"
            if self.strict:
                raise FileNotFoundError(msg)
            warnings.warn(msg)
        return p

    def paths_for_ticker(self, ticker: str, *, must_exist: bool = True) -> Dict[str, Path]:
        gf = self.generic_files
        out = {
            "summary": self.path_for(ticker, gf.summary, must_exist=must_exist),
            "profiles": self.path_for(ticker, gf.profiles, must_exist=must_exist),
            "vpin": self.path_for(ticker, gf.vpin, must_exist=must_exist),
            "rasterized": self.path_for(ticker, gf.rasterized, must_exist=must_exist),
            # Target is optional - can be computed from intraday if missing
            "target": self.path_for(ticker, gf.target, must_exist=False),
        }
        if gf.intraday:
            out["intraday"] = self.path_for(ticker, gf.intraday, must_exist=False)
        return out

    # ---------- summary alignment ----------

    def _load_raw_summaries(self, tickers: Sequence[str], max_workers: int = 8) -> Dict[str, pd.DataFrame]:
        """Load summary DataFrames for all tickers in parallel."""
        drop_cols = self.summary_config.drop_datetime_cols

        def _load_one(t: str) -> Tuple[str, pd.DataFrame]:
            p = self.path_for(t, self.generic_files.summary, must_exist=True)
            df = _read_tabular(p)
            if drop_cols:
                for c in ("Datetime", "datetime", "DateTime", "ts", "date", "Date"):
                    if c in df.columns:
                        df = df.drop(columns=[c])
            return t, df

        out: Dict[str, pd.DataFrame] = {}
        if len(tickers) <= 2:
            # No benefit from threading for small ticker counts
            for t in tickers:
                _, df = _load_one(t)
                out[t] = df
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(tickers))) as executor:
                futures = {executor.submit(_load_one, t): t for t in tickers}
                for future in as_completed(futures):
                    t, df = future.result()
                    out[t] = df
        return out

    def _align_summary_exact(self, summaries: Dict[str, pd.DataFrame]) -> Tuple[List[str], Dict[str, List[str]]]:
        cols_sets = [set(df.columns) for df in summaries.values()]
        common = set.intersection(*cols_sets) if cols_sets else set()
        common = [c for c in sorted(common) if c not in ("Datetime", "datetime", "DateTime", "ts", "date", "Date")]
        per = {t: common for t in summaries.keys()}
        return common, per

    def _align_summary_signature(self, summaries: Dict[str, pd.DataFrame]) -> Tuple[List[str], Dict[str, List[str]]]:
        cfg = self.summary_config

        # Build per ticker: signature -> list of columns that match
        sig_map: Dict[str, Dict[Tuple[str, str], List[str]]] = {}
        for t, df in summaries.items():
            m: Dict[Tuple[str, str], List[str]] = {}
            for c in df.columns:
                sig = feature_signature(c)
                if sig is None:
                    continue
                window, remainder = sig
                if cfg.require_substrings and not _filter_by_substrings(remainder, cfg.require_substrings):
                    continue
                m.setdefault((window, remainder), []).append(c)
            sig_map[t] = m

        # Candidate windows
        # For each preferred window, compute intersection size
        best_window = None
        best_sigs: List[Tuple[str, str]] = []
        best_count = -1

        for w in cfg.prefer_windows:
            # gather sigs of window w for each ticker
            per_sets = []
            for t in summaries.keys():
                keys = [k for k in sig_map[t].keys() if k[0] == w]
                per_sets.append(set(keys))
            if not per_sets:
                continue
            inter = set.intersection(*per_sets) if per_sets else set()
            cnt = len(inter)
            if cnt >= cfg.min_common and best_window is None:
                best_window = w
                best_sigs = sorted(inter)
                break
            if cnt > best_count:
                best_count = cnt
                best_window = w if cnt > 0 else best_window
                best_sigs = sorted(inter) if cnt > 0 else best_sigs

        if not best_sigs:
            msg = "Could not find any shared summary feature signatures across tickers."
            if cfg.strict or self.strict:
                raise ValueError(msg)
            warnings.warn(msg)
            return [], {t: [] for t in summaries.keys()}

        # Add always_include (must exist in all tickers to remain schema-consistent)
        always = []
        if cfg.always_include:
            for c in cfg.always_include:
                if all(c in df.columns for df in summaries.values()):
                    always.append(c)

        # Choose one actual column per signature per ticker
        per_cols: Dict[str, List[str]] = {}
        # Build global aligned names as canonical remainder names + window to avoid anchor differences
        aligned_names: List[str] = []
        for (window, remainder) in best_sigs:
            aligned_names.append(f"{window}__{remainder}")

        for t, df in summaries.items():
            cols: List[str] = []
            for (window, remainder), aligned in zip(best_sigs, aligned_names):
                candidates = sig_map[t].get((window, remainder), [])
                if not candidates:
                    # should not happen given intersection, but keep safe
                    continue
                # deterministic pick: smallest column name
                chosen = sorted(candidates)[0]
                cols.append(chosen)
            # prepend always include (exact names)
            cols = always + cols
            per_cols[t] = cols

        # The "common schema" is represented as the aligned_names, but the actual
        # per-ticker columns may differ (anchors). We return aligned_names for reference
        # and per-ticker columns for selection.
        # For DeepIDMomentum, we must supply actual columns; we keep aligned_names in cache.
        return aligned_names, per_cols

    def _recompute_summary_universal(
        self,
        ticker: str,
        intraday_df: pd.DataFrame,
        *,
        tz: str = "America/New_York",
        session_start: Optional[Tuple[int, int]] = None,
        session_end: Optional[Tuple[int, int]] = None,
        n_periods: Sequence[int] = (1, 5, 10),
    ) -> pd.DataFrame:
        """
        Recompute a universal summary feature table from intraday bars.
        Produces rolling session return and session volatility features (shifted 1).
        """
        from datetime import time
        ss = time(*(session_start or (9, 30)))
        se = time(*(session_end or (16, 0)))

        rets = session_returns(intraday_df, price_col="Close", session_start=ss, session_end=se, tz=tz)
        vol = session_realized_volatility(intraday_df, price_col="Close", session_start=ss, session_end=se, tz=tz)

        Xr = cumulative_session_returns(returns=rets, n_periods=n_periods)
        Xv = cumulative_session_volatility(volatility=vol, n_periods=n_periods)

        X = pd.concat([Xr, Xv], axis=1)
        X.index = pd.to_datetime(X.index)
        X = X.sort_index()
        return X

    def prepare_summary_schema(self, tickers: Optional[Sequence[str]] = None) -> None:
        """
        Compute and cache the aligned summary schema for the selected tickers.
        Call this once before building models if you want stable alignment.
        """
        use_tickers = list(tickers) if tickers is not None else self._tickers
        cfg = self.summary_config

        if cfg.strategy == "recompute":
            # schema is determined by recompute method; fixed names
            self._aligned_summary_cols = ["session_return_1", "session_return_5", "session_return_10",
                                         "session_volatility_1", "session_volatility_5", "session_volatility_10"]
            self._per_ticker_summary_cols = None
            return

        summaries = self._load_raw_summaries(use_tickers)
        if cfg.strategy == "exact":
            aligned, per = self._align_summary_exact(summaries)
        elif cfg.strategy == "signature":
            aligned, per = self._align_summary_signature(summaries)
        else:
            raise ValueError(f"Unknown summary_config.strategy={cfg.strategy!r}")

        self._aligned_summary_cols = aligned
        self._per_ticker_summary_cols = per

    # ---------- per-ticker model build ----------

    def get_model(
        self,
        ticker: str,
        *,
        target_col: str = "target",
        force_reload: bool = False,
        use_profile_scaler: bool = True,
    ) -> DeepIDMomentum:
        """
        Load (or return cached) DeepIDMomentum instance for ticker, from folder artifacts.
        """
        if (not force_reload) and ticker in self._models:
            return self._models[ticker]

        # Ensure summary schema is prepared
        if self._aligned_summary_cols is None and self.summary_config.strategy in ("exact", "signature", "recompute"):
            self.prepare_summary_schema(self._tickers)

        paths = self.paths_for_ticker(ticker, must_exist=True)

        sequential_df = _read_tabular(paths["vpin"])

        # Target: load from file if exists, otherwise create placeholder (to be computed later)
        target_path = paths.get("target")
        if target_path is not None and target_path.exists():
            target_series = _read_target(target_path, target_col=target_col)
        else:
            # No target file - create placeholder Series aligned with sequential data dates
            # User should call _calculate_target_returns() to compute actual targets
            warnings.warn(
                f"[{ticker}] target.csv not found. Creating placeholder targets. "
                f"Call model._calculate_target_returns() to compute targets from intraday data."
            )
            # Use sequential data dates to create placeholder
            if isinstance(sequential_df.index, pd.DatetimeIndex):
                dates = sequential_df.index.normalize().unique()
            else:
                dates = pd.to_datetime(sequential_df.index).normalize().unique()
            target_series = pd.Series(np.nan, index=dates, name="target")

        # intraday: prefer file if present, else synthesize
        intraday_df: pd.DataFrame
        ip = paths.get("intraday")
        if ip is not None and ip.exists():
            if read_exported_df is not None:
                # Use the standard intraday reader (handles column names, datetime index, Last->Close)
                intraday_df = read_exported_df(str(ip))
            else:
                # Fallback if import failed
                intraday_df = _read_tabular(ip)
                if "Close" not in intraday_df.columns:
                    for c in ("close", "Last", "last", "price"):
                        if c in intraday_df.columns:
                            intraday_df = intraday_df.rename(columns={c: "Close"})
                            break
        else:
            intraday_df = _synthesize_intraday_from_sequential(sequential_df)

        # summary features selection / recompute
        if self.summary_config.strategy == "recompute":
            features_df = self._recompute_summary_universal(ticker, intraday_df)
        else:
            raw = _read_tabular(paths["summary"])
            if self.summary_config.drop_datetime_cols:
                for c in ("Datetime", "datetime", "DateTime", "ts", "date", "Date"):
                    if c in raw.columns:
                        raw = raw.drop(columns=[c])
            # select per ticker columns
            per = self._per_ticker_summary_cols or {}
            sel = per.get(ticker)
            if sel is None:
                # fall back: if exact, aligned are real columns; if signature, choose intersection of actual cols
                sel = [c for c in raw.columns if c in (self._aligned_summary_cols or [])]
            if sel:
                features_df = raw[sel].copy()
            else:
                features_df = raw.copy()

        # Profile arrays
        try:
            profile_array, profile_dates = DeepIDMomentum._load_profile(  # type: ignore[attr-defined]
                str(paths["profiles"]), use_scaler=use_profile_scaler
            )
        except Exception as e:
            warnings.warn(f"[{ticker}] profile load via _load_profile failed ({e}); trying raw np.load fallback.")
            npz = np.load(paths["profiles"], allow_pickle=True)
            key = None
            for k in ("tensor", "profiles", "data", "arr_0"):
                if k in npz.files:
                    key = k
                    break
            if key is None:
                raise ValueError(f"Could not infer profile array key in {paths['profiles']}. Keys={npz.files}")
            profile_array = np.asarray(npz[key], dtype=np.float32)
            dkey = None
            for k in ("dates", "date", "dates_str", "arr_1"):
                if k in npz.files:
                    dkey = k
                    break
            profile_dates = np.asarray(npz[dkey]) if dkey else None

        # Rasterized
        try:
            rasterized_data = DeepIDMomentum._load_rasterized(str(paths["rasterized"]))  # type: ignore[attr-defined]
        except Exception as e:
            warnings.warn(f"[{ticker}] rasterized load via _load_rasterized failed ({e}); passing path string instead.")
            rasterized_data = str(paths["rasterized"])

        kwargs_copy = dict(self.deep_kwargs)
        for k in ("intraday_data", "sequential_data", "profile_array", "profile_dates", "rasterized_data"):
            kwargs_copy.pop(k, None)

        model = DeepIDMomentum(  # type: ignore[call-arg]
            intraday_data=intraday_df,
            sequential_data=sequential_df,
            profile_array=profile_array,
            profile_dates=profile_dates,
            rasterized_data=rasterized_data,
            **kwargs_copy,
        )

        # override summary and target
        model.training_data["summary"] = features_df.copy()
        model.feature_names = list(features_df.columns)
        model.target_data = target_series

        self._models[ticker] = model
        return model

    # ---------- dataset caching ----------

    def build_cache_signature(self, tickers: Sequence[str]) -> Dict[str, Any]:
        sig: Dict[str, Any] = {"tickers": list(tickers), "files": {}}
        for t in tickers:
            paths = self.paths_for_ticker(t, must_exist=True)
            fs = {k: _file_signature(p) for k, p in paths.items()}
            sig["files"][t] = fs
        return sig

    def save_dataset_cache(self, cache_path: Union[str, Path], payload: Dict[str, Any]) -> Path:
        cp = Path(cache_path).expanduser().resolve()
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cp)
        return cp

    def load_dataset_cache(
        self,
        cache_path: Union[str, Path],
        *,
        validate: bool = True,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        cp = Path(cache_path).expanduser().resolve()
        if not cp.exists():
            raise FileNotFoundError(f"Cache file not found: {cp}")

        # Load with timeout to prevent hanging on corrupted files
        def _load():
            return torch.load(cp, map_location="cpu", weights_only=False)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_load)
            try:
                payload = future.result(timeout=timeout)
            except TimeoutError:
                raise TimeoutError(f"torch.load timed out after {timeout}s loading {cp}. File may be corrupted.")
        if validate and isinstance(payload, dict) and "signature" in payload:
            sig_old = payload["signature"]
            try:
                sig_new = self.build_cache_signature(payload.get("tickers", self._tickers))
                mismatches = []
                for t in sig_old.get("tickers", []):
                    old_files = sig_old.get("files", {}).get(t, {})
                    new_files = sig_new.get("files", {}).get(t, {})
                    for k, old in old_files.items():
                        new = new_files.get(k, {})
                        if old.get("exists") != new.get("exists"):
                            mismatches.append((t, k, "exists", old.get("exists"), new.get("exists")))
                            continue
                        if old.get("exists"):
                            if old.get("mtime") != new.get("mtime") or old.get("size") != new.get("size"):
                                mismatches.append((t, k, "mtime/size", old.get("mtime"), new.get("mtime")))
                if mismatches:
                    warnings.warn(
                        f"Dataset cache appears stale: {len(mismatches)} source file changes detected. "
                        f"Consider rebuilding. First few: {mismatches[:5]}"
                    )
            except Exception as e:
                warnings.warn(f"Could not validate cache signature: {e}")
        return payload

    # ---------- multi-ticker loaders ----------

    def _compute_common_dates(
        self,
        tickers: Sequence[str],
        target_col: str = "target",
    ) -> Tuple[pd.DatetimeIndex, Dict[str, pd.DatetimeIndex]]:
        """
        Compute the intersection of available dates across all tickers.

        Returns
        -------
        common_dates : pd.DatetimeIndex
            Sorted intersection of dates available across all tickers
        per_ticker_dates : Dict[str, pd.DatetimeIndex]
            Available dates for each ticker
        """
        per_ticker_dates: Dict[str, pd.DatetimeIndex] = {}

        for t in tickers:
            paths = self.paths_for_ticker(t, must_exist=True)
            target_series = _read_target(paths["target"], target_col=target_col)
            if isinstance(target_series.index, pd.DatetimeIndex):
                dates = target_series.index.normalize().unique()
            else:
                dates = pd.to_datetime(target_series.index).normalize().unique()
            per_ticker_dates[t] = pd.DatetimeIndex(sorted(dates))

        # Compute intersection
        if not per_ticker_dates:
            return pd.DatetimeIndex([]), {}

        common = set(per_ticker_dates[tickers[0]])
        for t in tickers[1:]:
            common &= set(per_ticker_dates[t])

        common_dates = pd.DatetimeIndex(sorted(common))
        return common_dates, per_ticker_dates

    def compute_val_cutoff_date(
        self,
        tickers: Optional[Sequence[str]] = None,
        val_split_size: float = 0.2,
        target_col: str = "target",
    ) -> pd.Timestamp:
        """
        Compute a consistent validation cutoff date across all tickers.

        This ensures no lookahead bias when using ConcatDataset - all tickers
        will use the same date to split train/val, based on the common date
        intersection.

        Parameters
        ----------
        tickers : Sequence[str], optional
            Tickers to consider (defaults to all)
        val_split_size : float
            Fraction of data for validation (default 0.2 = 20%)
        target_col : str
            Target column name for reading dates

        Returns
        -------
        pd.Timestamp
            The cutoff date - dates before this are training, dates >= are validation
        """
        use_tickers = list(tickers) if tickers is not None else self._tickers
        common_dates, _ = self._compute_common_dates(use_tickers, target_col)

        if len(common_dates) == 0:
            raise ValueError("No common dates found across tickers")

        split_idx = int(len(common_dates) * (1 - val_split_size))
        if split_idx >= len(common_dates):
            split_idx = len(common_dates) - 1

        return common_dates[split_idx]

    def get_loaders(
        self,
        *,
        tickers: Optional[Sequence[str]] = None,
        mode: str = "concat",
        add_meta: bool = False,
        ticker_id_map: Optional[Mapping[str, int]] = None,
        asset_class_id_map: Optional[Mapping[str, int]] = None,
        asset_subclass_id_map: Optional[Mapping[str, int]] = None,
        meta_default: Tuple[int, int, int] = (0, 0, 0),
        # WSPR mode for MultiAssetWSPR models
        use_wspr: bool = False,
        # caching controls
        cache_path: Optional[Union[str, Path]] = None,
        use_cache: bool = False,
        save_cache: bool = False,
        validate_cache: bool = True,
        # Date-based split for lookahead prevention
        val_cutoff_date: Optional[Union[str, pd.Timestamp]] = None,
        auto_align_dates: bool = True,
        # DeepIDMomentum.get_loaders passthrough:
        target_col: str = "target",
        **loader_kwargs: Any,
    ):
        """
        Build DataLoaders for multiple tickers with consistent date-based splits.

        IMPORTANT: When using val_split=True with ConcatDataset (mode='concat'),
        this method enforces a consistent validation cutoff date across all tickers
        to prevent lookahead bias. Without this, each ticker's percentage-based split
        could result in different cutoff dates, causing validation data from one ticker
        to overlap temporally with training data from another.

        Parameters
        ----------
        tickers : Sequence[str], optional
            Tickers to include (defaults to all discovered tickers)
        mode : str, default 'concat'
            'concat': Concatenate all ticker datasets into single train/val loaders
            'dict': Return dict mapping ticker -> (train_loader, val_loader)
        add_meta : bool, default False
            If True, append (ticker_id, asset_class_id, subclass_id) tensor to each sample.
            Ignored when use_wspr=True (metadata is embedded in WSPRWindowDataset).
        use_wspr : bool, default False
            If True, use WSPRWindowDataset which provides:
            - Windowed summary/profile for LSTM processing
            - Only recent day's raster/sequential (not windowed)
            - Full meta dict with ticker IDs + calendar features (month, dow, doy)
            Required for MultiAssetWSPR / RecurrentWSPR models.
            When True, ticker_id_map/asset_class_id_map/asset_subclass_id_map are
            injected into each ticker's dataset for wspr_collate_fn to use.
        val_cutoff_date : str or pd.Timestamp, optional
            Explicit cutoff date for train/val split. Dates < cutoff are training,
            dates >= cutoff are validation. If None and auto_align_dates=True,
            a consistent cutoff is computed automatically.
        auto_align_dates : bool, default True
            If True and val_split=True, automatically compute a consistent cutoff
            date from the intersection of dates across all tickers. This prevents
            lookahead bias when using ConcatDataset.
        **loader_kwargs
            Passed to DeepIDMomentum.get_loaders(). Common options:
            - val_split: bool (default False) - whether to split train/val
            - val_split_size: float (default 0.2) - fraction for validation
            - batch_size: int (default 32)
            - include_spatial: bool - include profile/rasterized data

        Returns
        -------
        DataLoader or Tuple[DataLoader, DataLoader] or Dict
            Depending on mode and val_split settings.

        Examples
        --------
        >>> # Safe multi-asset training with automatic date alignment
        >>> train_loader, val_loader = mam.get_loaders(
        ...     val_split=True,
        ...     auto_align_dates=True,  # default, ensures no lookahead
        ...     batch_size=64
        ... )
        >>>
        >>> # Explicit cutoff date
        >>> train_loader, val_loader = mam.get_loaders(
        ...     val_split=True,
        ...     val_cutoff_date='2024-01-01',
        ...     batch_size=64
        ... )
        >>>
        >>> # WSPR mode for MultiAssetWSPR models (includes calendar meta)
        >>> train_loader, val_loader = mam.get_loaders(
        ...     val_split=True,
        ...     use_wspr=True,
        ...     ticker_id_map={'HE': 0, 'LE': 1},
        ...     asset_class_id_map={'HE': 0, 'LE': 0},  # Both livestock
        ...     asset_subclass_id_map={'HE': 0, 'LE': 1},
        ...     use_rasterized=True,
        ...     include_spatial=True,
        ...     windowed=True,
        ...     window_days=10,
        ...     batch_size=16
        ... )
        """
        use_tickers = list(tickers) if tickers is not None else self._tickers
        if not use_tickers:
            raise ValueError("No tickers selected.")
        if mode not in ("concat", "dict"):
            raise ValueError("mode must be one of: 'concat', 'dict'")

        # ---- cache load path ----
        if use_cache:
            if cache_path is None:
                raise ValueError("use_cache=True requires cache_path")
            payload = self.load_dataset_cache(cache_path, validate=validate_cache)
            if payload.get("mode") != mode:
                warnings.warn(f"Cache mode mismatch: cache has {payload.get('mode')}, requested {mode}. Rebuilding.")
            else:
                return self._loaders_from_cache(payload, mode=mode, **loader_kwargs)

        # ---- Date alignment for lookahead prevention ----
        # If val_split is requested, ensure consistent date-based split across all tickers
        wants_val_split = loader_kwargs.get("val_split", False)
        val_split_size = loader_kwargs.get("val_split_size", 0.2)

        effective_cutoff: Optional[pd.Timestamp] = None
        if val_cutoff_date is not None:
            effective_cutoff = pd.Timestamp(val_cutoff_date)
        elif auto_align_dates and wants_val_split:
            # Auto-compute a consistent cutoff date across all tickers
            effective_cutoff = self.compute_val_cutoff_date(
                tickers=use_tickers,
                val_split_size=val_split_size,
                target_col=target_col,
            )
            warnings.warn(
                f"Auto-aligned validation cutoff date: {effective_cutoff.strftime('%Y-%m-%d')}. "
                f"All tickers will use this date to prevent lookahead bias."
            )

        # Ensure summary schema computed for selected tickers
        if self._aligned_summary_cols is None:
            self.prepare_summary_schema(use_tickers)

        per: Dict[str, Any] = {}

        def _build_ticker_loaders(t: str, cutoff_date: Optional[pd.Timestamp] = None) -> Tuple[str, Any]:
            """Build loaders for a single ticker (thread-safe: no shared state mutation)."""
            m = self.get_model(t, target_col=target_col)

            # Build kwargs for this ticker
            ticker_kwargs = {**loader_kwargs}

            # If WSPR mode, inject ticker metadata into the loader kwargs
            if use_wspr:
                tid = (ticker_id_map or {}).get(t, meta_default[0])
                cid = (asset_class_id_map or {}).get(t, meta_default[1])
                sid = (asset_subclass_id_map or {}).get(t, meta_default[2])
                ticker_kwargs.update({
                    "use_wspr": True,
                    "ticker_id": tid,
                    "asset_class_id": cid,
                    "asset_subclass_id": sid,
                })

            if cutoff_date is not None and wants_val_split:
                # Date-based split: build train and val loaders separately
                # This prevents lookahead bias when concatenating across tickers
                # Note: DeepIDMomentum uses inclusive end_date, so train ends 1 day before cutoff
                train_end = cutoff_date - pd.Timedelta(days=1)
                train_kwargs = {**ticker_kwargs, "val_split": False, "end_date": train_end}
                val_kwargs = {**ticker_kwargs, "val_split": False, "start_date": cutoff_date}

                train_loader = m.get_loaders(**train_kwargs)
                val_loader = m.get_loaders(**val_kwargs)
                out = (train_loader, val_loader)
            else:
                out = m.get_loaders(**ticker_kwargs)

            # add_meta is ignored when use_wspr=True (metadata already in dataset)
            if add_meta and not use_wspr:
                tid = (ticker_id_map or {}).get(t, meta_default[0])
                cid = (asset_class_id_map or {}).get(t, meta_default[1])
                sid = (asset_subclass_id_map or {}).get(t, meta_default[2])
                meta = torch.tensor([tid, cid, sid], dtype=torch.long)

                if isinstance(out, tuple) and len(out) == 2:
                    tr, va = out
                    out = (
                        self._rebuild_loader(_WithMeta(tr.dataset, meta), template=tr, shuffle=True),
                        self._rebuild_loader(_WithMeta(va.dataset, meta), template=va, shuffle=False),
                    )
                else:
                    ld = out
                    out = self._rebuild_loader(_WithMeta(ld.dataset, meta), template=ld, shuffle=True)
            return t, out

        # Load tickers - parallel when >2 tickers and parallel_load enabled
        parallel_load = loader_kwargs.pop("parallel_load", len(use_tickers) > 2)
        max_workers = loader_kwargs.pop("max_workers", 4)

        if parallel_load and len(use_tickers) > 2:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(use_tickers))) as executor:
                futures = {executor.submit(_build_ticker_loaders, t, effective_cutoff): t for t in use_tickers}
                for future in as_completed(futures):
                    try:
                        t, out = future.result(timeout=300)  # 5 min timeout per ticker
                        per[t] = out
                    except Exception as e:
                        ticker = futures[future]
                        warnings.warn(f"Failed to load ticker {ticker}: {e}")
                        if self.strict:
                            raise
        else:
            for t in use_tickers:
                _, out = _build_ticker_loaders(t, effective_cutoff)
                per[t] = out

        if save_cache and cache_path is not None:
            payload = self._build_cache_payload(per, tickers=use_tickers, mode=mode)
            self.save_dataset_cache(cache_path, payload)

        if mode == "dict":
            return per

        first = per[use_tickers[0]]
        if isinstance(first, tuple) and len(first) == 2:
            train_datasets = [per[t][0].dataset for t in use_tickers]
            val_datasets = [per[t][1].dataset for t in use_tickers]
            train_ds = ConcatDataset(train_datasets)
            val_ds = ConcatDataset(val_datasets)

            train_loader0, val_loader0 = first
            train_loader = self._rebuild_loader(
                train_ds,
                template=train_loader0,
                shuffle=loader_kwargs.get("shuffle_train", True),
            )
            val_loader = self._rebuild_loader(val_ds, template=val_loader0, shuffle=False)
            return train_loader, val_loader

        datasets = [per[t].dataset for t in use_tickers]
        ds = ConcatDataset(datasets)
        loader0 = first
        loader = self._rebuild_loader(
            ds,
            template=loader0,
            shuffle=loader_kwargs.get("shuffle_train", True),
        )
        return loader

    # ---------- internal helpers ----------

    @staticmethod
    def _rebuild_loader(ds: Dataset, *, template: DataLoader, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=template.batch_size,
            shuffle=shuffle,
            num_workers=template.num_workers,
            collate_fn=template.collate_fn,
            drop_last=template.drop_last,
            pin_memory=template.pin_memory,
            persistent_workers=getattr(template, "persistent_workers", False),
        )

    def _build_cache_payload(self, per: Dict[str, Any], *, tickers: Sequence[str], mode: str) -> Dict[str, Any]:
        first = per[list(tickers)[0]]
        val_split = bool(isinstance(first, tuple) and len(first) == 2)

        payload: Dict[str, Any] = {
            "version": 2,
            "mode": mode,
            "tickers": list(tickers),
            "val_split": val_split,
            "signature": self.build_cache_signature(tickers),
        }

        if mode == "dict":
            if val_split:
                payload["datasets"] = {t: (per[t][0].dataset, per[t][1].dataset) for t in tickers}
            else:
                payload["datasets"] = {t: per[t].dataset for t in tickers}
        else:
            if val_split:
                payload["datasets"] = {
                    "train": [per[t][0].dataset for t in tickers],
                    "val": [per[t][1].dataset for t in tickers],
                }
            else:
                payload["datasets"] = {"train": [per[t].dataset for t in tickers]}

        def _loader_cfg(ld: DataLoader) -> Dict[str, Any]:
            return {
                "batch_size": ld.batch_size,
                "num_workers": ld.num_workers,
                "drop_last": ld.drop_last,
                "pin_memory": ld.pin_memory,
                "persistent_workers": getattr(ld, "persistent_workers", False),
                "collate_fn": ld.collate_fn,
            }

        if val_split:
            payload["train_loader_cfg"] = _loader_cfg(first[0])
            payload["val_loader_cfg"] = _loader_cfg(first[1])
        else:
            payload["train_loader_cfg"] = _loader_cfg(first)

        return payload

    def _loaders_from_cache(self, payload: Dict[str, Any], *, mode: str, **loader_kwargs: Any):
        if payload.get("mode") != mode:
            raise ValueError(f"Cache mode mismatch: cache={payload.get('mode')} requested={mode}")

        val_split = bool(payload.get("val_split", False))

        def _make_loader(ds: Dataset, cfg: Dict[str, Any], *, shuffle: bool) -> DataLoader:
            return DataLoader(
                ds,
                batch_size=cfg["batch_size"],
                shuffle=shuffle,
                num_workers=cfg["num_workers"],
                collate_fn=cfg.get("collate_fn"),
                drop_last=cfg.get("drop_last", False),
                pin_memory=cfg.get("pin_memory", False),
                persistent_workers=cfg.get("persistent_workers", False),
            )

        if mode == "dict":
            out: Dict[str, Any] = {}
            dsets = payload["datasets"]
            if val_split:
                for t, (tr_ds, va_ds) in dsets.items():
                    out[t] = (
                        _make_loader(tr_ds, payload["train_loader_cfg"], shuffle=loader_kwargs.get("shuffle_train", True)),
                        _make_loader(va_ds, payload["val_loader_cfg"], shuffle=False),
                    )
            else:
                for t, tr_ds in dsets.items():
                    out[t] = _make_loader(tr_ds, payload["train_loader_cfg"], shuffle=loader_kwargs.get("shuffle_train", True))
            return out

        dsets = payload["datasets"]
        if val_split:
            train_ds = ConcatDataset(dsets["train"])
            val_ds = ConcatDataset(dsets["val"])
            return (
                _make_loader(train_ds, payload["train_loader_cfg"], shuffle=loader_kwargs.get("shuffle_train", True)),
                _make_loader(val_ds, payload["val_loader_cfg"], shuffle=False),
            )

        train_ds = ConcatDataset(dsets["train"])
        return _make_loader(train_ds, payload["train_loader_cfg"], shuffle=loader_kwargs.get("shuffle_train", True))


__all__ = ["GenericFiles", "SummarySelectionConfig", "MultiAssetMomentum", "feature_signature"]
