"""Crack-spread continuous prep built on top of ContinuousIntradayPrep.

This module uses a synthetic 3-2-1 crack spread as the master clock for
time-based spread features and targets. The orderflow branch is intentionally
kept separate so the dataset layer can mask and collate CL / HO / RB volume
bucket sequences without flattening them into ``df_out``.
"""
from __future__ import annotations

import logging
import importlib.util
import sys
from datetime import date as date_type, time as time_type
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from CTAFlow.data.raw_formatting.intraday_manager import read_exported_df, read_synthetic_csv

from .intraday_continuous import ContinuousIntradayPrep, SessionSpec

logger = logging.getLogger(__name__)

DEFAULT_CRACK_TICKERS: Tuple[str, str, str] = ("CL", "HO", "RB")
DEFAULT_TARGET_TICKER = "CRACK"
_SPREAD_PATH_CANDIDATES: Tuple[str, ...] = (
    "synthetic/crack_spread.csv",
    "synthetics/crack_spread.csv",
    "crack_spread.csv",
)
_ORDERFLOW_BASE_COLS: Tuple[str, ...] = (
    "vpin",
    "bucket_return",
    "log_duration",
    "signed_imbalance",
    "imb_frac",
    "buy_dom",
    "sell_dom",
    "vol_ratio",
    "vol",
    "buy",
    "sell",
    "imbalance",
    "close",
    "ps_poc",
    "ps_val",
    "ps_vah",
    "pd_poc",
    "pd_val",
    "pd_vah",
)
_VPIN_SPATIAL_READ_COLS: Tuple[str, ...] = ("close", "vol", "signed_imbalance", "vpin")
_DEFAULT_VPIN_READ_COLS: Tuple[str, ...] = tuple(
    dict.fromkeys((*_ORDERFLOW_BASE_COLS, *_VPIN_SPATIAL_READ_COLS))
)
_VPIN_FOLD_SUM_COLS: Tuple[str, ...] = ("vol", "buy", "sell", "imbalance")
_VPIN_FOLD_LAST_COLS: Tuple[str, ...] = (
    "close",
    "ps_poc",
    "ps_val",
    "ps_vah",
    "pd_poc",
    "pd_val",
    "pd_vah",
)
_LONDON_OPEN = time_type(2, 0)
_LONDON_CLOSE = time_type(11, 0)
_USA_OPEN = time_type(8, 30)
_USA_CLOSE = time_type(16, 0)


def rolling_zscore(
    s: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
    clip: Optional[float] = 5.0,
    eps: float = 1e-8,
) -> pd.Series:
    """Causal rolling z-score with optional clipping."""
    minp = min_periods or max(8, window // 4)
    mean = s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std().clip(lower=eps)
    z = (s - mean) / std
    if clip is not None:
        z = z.clip(-clip, clip)
    return z


def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, tz-naive DataFrame with a strict DatetimeIndex."""
    out = df.copy()
    if out.empty:
        return out
    return _normalize_datetime_index_inplace(out)


def _normalize_datetime_index_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame index without making an up-front full copy."""
    out = df
    if not isinstance(out.index, pd.DatetimeIndex):
        if "ts_end" in out.columns:
            out.index = pd.to_datetime(out["ts_end"], errors="coerce")
        elif "Datetime" in out.columns:
            out.index = pd.to_datetime(out["Datetime"], errors="coerce")
        elif "timestamp" in out.columns:
            out.index = pd.to_datetime(out["timestamp"], errors="coerce")
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    if out.index.hasnans:
        out = out[~out.index.isna()]
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def _fold_vpin_rows(
    df: pd.DataFrame,
    n_folds: int,
    normalize: bool = True,
) -> pd.DataFrame:
    """Fold consecutive VPIN rows into coarser buckets."""
    if n_folds <= 1 or df.empty:
        return df

    out = _normalize_datetime_index(df) if normalize else df
    numeric_cols = list(out.select_dtypes(include="number").columns)
    if not numeric_cols:
        return out

    n_rows = len(out)
    group_ids = np.arange(n_rows, dtype=np.int32) // int(n_folds)
    agg_map = {}
    for col in numeric_cols:
        if col in _VPIN_FOLD_SUM_COLS:
            agg_map[col] = "sum"
        elif col in _VPIN_FOLD_LAST_COLS:
            agg_map[col] = "last"
        else:
            agg_map[col] = "mean"

    folded = out.loc[:, numeric_cols].groupby(group_ids, sort=False).agg(agg_map)
    last_positions = np.minimum(
        (np.arange(len(folded), dtype=np.int32) + 1) * int(n_folds),
        n_rows,
    ) - 1
    folded.index = out.index.values[last_positions]
    folded.index.name = out.index.name
    return folded


def _load_event_presets_module():
    """Load event presets directly to avoid importing the full screeners package."""
    module_name = "_ctaflow_event_presets_direct"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = Path(__file__).resolve().parents[2] / "screeners" / "event_presets.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load event presets module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class CrackSpreadContinuousPrep(ContinuousIntradayPrep):
    """Continuous prep for a synthetic 3-2-1 crack spread anchor series."""

    @staticmethod
    def normalize_target_ticker(
        target_ticker: Optional[str],
    ) -> str:
        """Normalize target source names to CRACK or an uppercase ticker."""
        if target_ticker is None:
            return DEFAULT_TARGET_TICKER
        normalized = str(target_ticker).strip().upper()
        if normalized in {"", "CRACK", "SPREAD", "CRACK_SPREAD"}:
            return DEFAULT_TARGET_TICKER
        return normalized

    @staticmethod
    def get_known_temporal_cols(
        tickers: Tuple[str, ...] = DEFAULT_CRACK_TICKERS,
    ) -> List[str]:
        """Known-future temporal features emitted by the crack prep."""
        cols = [
            "kt_day_of_week",
            "kt_hour",
            "kt_session_label",
        ]
        for code in CrackSpreadContinuousPrep.get_crack_event_codes(tickers=tickers):
            cols.extend(
                [
                    f"kt_evt_{code}_is_release_day",
                    f"kt_evt_{code}_bars_to_release",
                    f"kt_evt_{code}_is_pre_release",
                ]
            )
        return cols

    @staticmethod
    def get_crack_event_codes(
        tickers: Tuple[str, ...] = DEFAULT_CRACK_TICKERS,
    ) -> List[str]:
        """Union of event codes for the crack constituents."""
        event_presets = _load_event_presets_module()

        codes = set()
        for ticker in tickers:
            codes.update(event_presets.TICKER_EVENT_MAP.get(ticker.upper(), []))
        return sorted(codes)

    @staticmethod
    def get_feature_cols(
        steps_60m: int = 12,
        bar_minutes: int = 5,
        momentum_lookbacks: Tuple[int, ...] = (5, 10, 20),
        sma_windows: Tuple[int, ...] = (50, 200),
        rv_lookbacks: Tuple[int, ...] = (1, 5, 20),
        add_daily: bool = True,
        add_overnight: bool = True,
        add_deseas: bool = True,
        add_time_features: bool = True,
        add_resample_precalc: bool = True,
        resample_rules: Tuple[str, ...] = ("15min", "30min", "60min"),
        add_bid_ask: bool = True,
        add_event_markers: bool = False,
        tickers: Tuple[str, ...] = DEFAULT_CRACK_TICKERS,
    ) -> List[str]:
        cols = ContinuousIntradayPrep.get_feature_cols(
            steps_60m=steps_60m,
            bar_minutes=bar_minutes,
            momentum_lookbacks=momentum_lookbacks,
            sma_windows=sma_windows,
            rv_lookbacks=rv_lookbacks,
            add_daily=add_daily,
            add_overnight=add_overnight,
            add_deseas=add_deseas,
            add_time_features=add_time_features,
            add_resample_precalc=add_resample_precalc,
            resample_rules=resample_rules,
            add_bid_ask=add_bid_ask,
            add_event_markers=add_event_markers,
        )
        cols.extend(
            [
                "spread_log_ret",
                "spread_ret_1",
                "spread_ret_3",
                "spread_ret_6",
                "spread_ret_12",
                "spread_range",
                "spread_hl_range",
                "spread_body_ratio",
                "spread_close_to_open",
                "spread_close_pos_in_bar",
                "spread_roll_vol_3",
                "spread_roll_vol_12",
                "spread_roll_vol_24",
                "spread_cum_ret_3",
                "spread_cum_ret_6",
                "spread_cum_ret_12",
                "spread_log_volume",
                "spread_volume_z36",
                f"spread_vwap_{steps_60m * bar_minutes}m_dist",
                "spread_z_1d",
                "spread_z_5d",
                "spread_z_20d",
                "spread_london_open_rel_close",
                "spread_london_close_rel_close",
                "spread_usa_open_rel_close",
                "spread_usa_close_rel_close",
            ]
        )
        cols.extend(CrackSpreadContinuousPrep.get_known_temporal_cols(tickers=tickers))
        return cols

    @staticmethod
    def get_orderflow_base_cols() -> List[str]:
        """Canonical raw orderflow columns used for the separate VPIN branch."""
        return list(_ORDERFLOW_BASE_COLS)

    def __init__(
        self,
        root_features_dir: Union[str, Path],
        tickers: Tuple[str, str, str] = DEFAULT_CRACK_TICKERS,
        sessions: Optional[Sequence[SessionSpec]] = None,
        bar_minutes: int = 5,
        eps: float = 1e-8,
        target_mode: str = "delta",
        fold: bool = False,
        n_folds: int = 3,
    ):
        super().__init__(sessions=sessions, bar_minutes=bar_minutes, eps=eps)
        self.root_features_dir = Path(root_features_dir)
        self.tickers = tuple(t.upper() for t in tickers)
        self.target_mode = target_mode
        self.fold = bool(fold)
        self.n_folds = max(1, int(n_folds))

    def _resample_ohlcv_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample a 5-minute OHLCV frame onto the configured master clock."""
        out = self._standardize_columns(df)
        if self.bar_minutes <= 5:
            return out

        rule = f"{self.bar_minutes}min"
        agg = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
        if "BidVolume" in out.columns:
            agg["BidVolume"] = "sum"
        if "AskVolume" in out.columns:
            agg["AskVolume"] = "sum"

        out = out.resample(rule, label="right", closed="right").agg(agg)
        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        return out

    def _resolve_spread_path(self) -> Path:
        for rel in _SPREAD_PATH_CANDIDATES:
            candidate = self.root_features_dir / rel
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find crack spread CSV under {self.root_features_dir}. "
            f"Tried: {_SPREAD_PATH_CANDIDATES}"
        )

    def _resolve_intraday_path(self, ticker: str) -> Path:
        path = self.root_features_dir / ticker / "intraday.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing intraday.csv for {ticker}: {path}")
        return path

    def _resolve_vpin_path(self, ticker: str) -> Path:
        candidates = [
            self.root_features_dir / ticker / "crack_vpin.parquet",
            self.root_features_dir / ticker / "vpin.parquet",
            self.root_features_dir / f"crack_{ticker.lower()}_vpin.parquet",
            self.root_features_dir / f"crack_{ticker}_vpin.parquet",
            self.root_features_dir / f"{ticker}_vpin.parquet",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Missing VPIN parquet for {ticker}. Tried: {candidates}")

    def _load_spread_df(self) -> pd.DataFrame:
        path = self._resolve_spread_path()
        try:
            spread = read_synthetic_csv(path)
        except Exception:
            spread = read_exported_df(path)
        return self._standardize_columns(spread)

    def _load_intraday_df(self, ticker: str) -> pd.DataFrame:
        return self._standardize_columns(read_exported_df(self._resolve_intraday_path(ticker)))

    def _build_target_frame(
        self,
        target_ticker: Optional[str],
        spread_df: pd.DataFrame,
        asset_intraday: Dict[str, pd.DataFrame],
        anchor_idx: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Build the OHLCV frame used to calculate targets on the master clock."""
        resolved_target = self.normalize_target_ticker(target_ticker)
        if resolved_target == DEFAULT_TARGET_TICKER:
            target_df = self._resample_ohlcv_frame(spread_df)
        else:
            if resolved_target not in asset_intraday:
                raise ValueError(
                    f"Unknown target_ticker={target_ticker!r}. "
                    f"Expected one of {self.tickers} or 'CRACK'."
                )
            target_df = self._resample_ohlcv_frame(asset_intraday[resolved_target])

        if not target_df.index.equals(anchor_idx):
            target_df = target_df.reindex(anchor_idx)
        return target_df

    def _apply_target_frame(
        self,
        df: pd.DataFrame,
        target_df: pd.DataFrame,
        steps: int,
        prefix: str = "y_fwd",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Overwrite target columns using an alternate OHLCV target source."""
        out = df.copy()
        close = target_df["Close"].astype(float)
        logp = np.log(close.clip(lower=self.eps))
        target_cols: List[str] = []

        rv = None
        if self.target_mode == "vol_norm_delta":
            logret = logp.diff()
            daily_rv = (logret ** 2).groupby(target_df.index.normalize()).sum().pipe(np.sqrt)
            rv = daily_rv.shift(1).reindex(target_df.index.normalize()).ffill().values.astype(np.float32)

        for h in range(1, steps + 1):
            col = f"{prefix}_{h}"
            if self.target_mode == "logret":
                out[col] = logp.shift(-h) - logp
            elif self.target_mode == "vol_norm_delta":
                raw_delta = close.shift(-h) - close
                if rv is not None:
                    out[col] = raw_delta / (rv + self.eps)
                else:
                    out[col] = raw_delta
            else:
                out[col] = close.shift(-h) - close
            target_cols.append(col)

        return out, target_cols

    def _load_vpin_df(
        self,
        ticker: str,
        columns: Optional[Sequence[str]] = None,
        fold: Optional[bool] = None,
        n_folds: Optional[int] = None,
        start_ts: Optional[Union[str, pd.Timestamp]] = None,
        end_ts: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        selected_cols = _DEFAULT_VPIN_READ_COLS if columns is None else tuple(columns)
        read_cols = list(dict.fromkeys(["ts_end", *selected_cols]))
        filters = []
        if start_ts is not None:
            filters.append(("ts_end", ">=", pd.Timestamp(start_ts)))
        if end_ts is not None:
            filters.append(("ts_end", "<=", pd.Timestamp(end_ts)))
        df = pd.read_parquet(
            self._resolve_vpin_path(ticker),
            columns=read_cols,
            filters=filters or None,
        )

        float64_cols = list(df.select_dtypes(include="float64").columns)
        for col in float64_cols:
            df[col] = df[col].astype(np.float32, copy=False)

        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts_end" in df.columns:
                df.index = pd.to_datetime(df["ts_end"], errors="coerce")
            elif "Datetime" in df.columns:
                df.index = pd.to_datetime(df["Datetime"], errors="coerce")
            elif "timestamp" in df.columns:
                df.index = pd.to_datetime(df["timestamp"], errors="coerce")
            else:
                df.index = pd.to_datetime(df.index, errors="coerce")

        # Drop helper columns that should not become model inputs.
        for col in ("date", "ticker", "ts_end"):
            if col in df.columns:
                df = df.drop(columns=col)

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if df.index.hasnans:
            df = df[~df.index.isna()]
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        do_fold = self.fold if fold is None else bool(fold)
        fold_size = self.n_folds if n_folds is None else max(1, int(n_folds))
        if do_fold and fold_size > 1:
            before_rows = len(df)
            df = _fold_vpin_rows(df, fold_size, normalize=False)
            logger.info("[%s] folded VPIN rows %s -> %s using n_folds=%s", ticker, before_rows, len(df), fold_size)
        df = _normalize_datetime_index_inplace(df)
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                df[col] = df[col].astype(np.float32, copy=False)
        return df

    def load_raw_inputs(
        self,
        load_orderflow: bool = False,
        orderflow_columns: Optional[Sequence[str]] = None,
        fold: Optional[bool] = None,
        n_folds: Optional[int] = None,
        vpin_start_ts: Optional[Union[str, pd.Timestamp]] = None,
        vpin_end_ts: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Dict[str, object]:
        asset_intraday = {ticker: self._load_intraday_df(ticker) for ticker in self.tickers}
        out = {
            "spread": self._load_spread_df(),
            "asset_intraday": asset_intraday,
        }
        if load_orderflow:
            out["asset_vpin"] = {
                ticker: self._load_vpin_df(
                    ticker,
                    columns=orderflow_columns,
                    fold=fold,
                    n_folds=n_folds,
                    start_ts=vpin_start_ts,
                    end_ts=vpin_end_ts,
                )
                for ticker in self.tickers
            }
        return out

    def build_spread_anchor_df(
        self,
        spread_df: pd.DataFrame,
        asset_intraday: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        df = self._resample_ohlcv_frame(spread_df)
        anchor_idx = df.index

        total_volume = pd.Series(0.0, index=anchor_idx)
        total_bid = pd.Series(0.0, index=anchor_idx)
        total_ask = pd.Series(0.0, index=anchor_idx)
        have_bid_ask = True

        for ticker, asset_df in asset_intraday.items():
            aligned = self._resample_ohlcv_frame(asset_df).reindex(anchor_idx)
            total_volume = total_volume.add(aligned["Volume"].fillna(0.0), fill_value=0.0)
            if "BidVolume" in aligned.columns and "AskVolume" in aligned.columns:
                total_bid = total_bid.add(aligned["BidVolume"].fillna(0.0), fill_value=0.0)
                total_ask = total_ask.add(aligned["AskVolume"].fillna(0.0), fill_value=0.0)
            else:
                have_bid_ask = False
                logger.debug("[%s] missing bid/ask volume; spread bid/ask features disabled", ticker)

        df["Volume"] = total_volume.astype(np.float32)
        if have_bid_ask:
            df["BidVolume"] = total_bid.astype(np.float32)
            df["AskVolume"] = total_ask.astype(np.float32)

        return df

    def add_spread_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        bars_per_day = max(1, int(round(390 / max(self.bar_minutes, 1))))
        close = df["Close"].astype(float)
        open_ = df["Open"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float).clip(lower=0.0)

        log_close = np.log(close.clip(lower=self.eps))
        df["spread_log_ret"] = log_close.diff()
        for bars in (1, 3, 6, 12):
            df[f"spread_ret_{bars}"] = close.pct_change(bars)
            df[f"spread_cum_ret_{bars}"] = df["spread_log_ret"].rolling(bars, min_periods=1).sum()
        df["spread_range"] = (high - low) / close.clip(lower=self.eps)
        df["spread_hl_range"] = high - low
        df["spread_close_to_open"] = (close - open_) / open_.clip(lower=self.eps)
        df["spread_body_ratio"] = (close - open_) / (high - low).replace(0.0, np.nan)
        df["spread_close_pos_in_bar"] = ((close - low) / (high - low).replace(0.0, np.nan) - 0.5).clip(-0.5, 0.5)
        df["spread_roll_vol_3"] = df["spread_log_ret"].rolling(3, min_periods=1).std()
        df["spread_roll_vol_12"] = df["spread_log_ret"].rolling(12, min_periods=1).std()
        df["spread_roll_vol_24"] = df["spread_log_ret"].rolling(24, min_periods=1).std()
        df["spread_log_volume"] = np.log1p(volume)
        df["spread_volume_z36"] = rolling_zscore(df["spread_log_volume"], 36, eps=self.eps)
        for col in df.columns:
            if col.startswith("vwap_") and col.endswith("_dist"):
                df[f"spread_{col}"] = df[col]
        df["spread_z_1d"] = rolling_zscore(close, max(1, bars_per_day), eps=self.eps)
        df["spread_z_5d"] = rolling_zscore(close, max(1, bars_per_day * 5), eps=self.eps)
        df["spread_z_20d"] = rolling_zscore(close, max(1, bars_per_day * 20), eps=self.eps)
        df = self.add_session_reference_price_features(df)
        df = self.add_known_temporal_features(df)
        return df

    def _causal_session_reference_feature(
        self,
        df: pd.DataFrame,
        event_time: time_type,
        source_col: str,
        out_col: str,
    ) -> pd.DataFrame:
        """Causally forward-fill the most recent session marker price."""
        out = df.copy()
        close = out["Close"].astype(float).clip(lower=self.eps)
        source = out[source_col].astype(float)
        hit_mask = (
            (out.index.hour == event_time.hour) &
            (out.index.minute == event_time.minute)
        )
        ref = pd.Series(np.where(hit_mask, source, np.nan), index=out.index, dtype=float).ffill()
        out[out_col] = (ref - close) / close
        return out

    def add_session_reference_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add causal London / US open-close reference price features."""
        out = df.copy()
        out = self._causal_session_reference_feature(
            out,
            event_time=_LONDON_OPEN,
            source_col="Open",
            out_col="spread_london_open_rel_close",
        )
        out = self._causal_session_reference_feature(
            out,
            event_time=_LONDON_CLOSE,
            source_col="Close",
            out_col="spread_london_close_rel_close",
        )
        out = self._causal_session_reference_feature(
            out,
            event_time=_USA_OPEN,
            source_col="Open",
            out_col="spread_usa_open_rel_close",
        )
        out = self._causal_session_reference_feature(
            out,
            event_time=_USA_CLOSE,
            source_col="Close",
            out_col="spread_usa_close_rel_close",
        )
        return out

    def _session_label_from_index(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Map timestamps to London=0, USA=1, Overnight=2."""
        london_mask = self._time_mask(idx, _LONDON_OPEN.strftime("%H:%M"), _LONDON_CLOSE.strftime("%H:%M"))
        usa_mask = self._time_mask(idx, _USA_OPEN.strftime("%H:%M"), _USA_CLOSE.strftime("%H:%M"))
        return np.where(usa_mask, 1, np.where(london_mask, 0, 2)).astype(np.float32)

    def add_known_temporal_features(
        self,
        df: pd.DataFrame,
        pre_release_bars: int = 12,
    ) -> pd.DataFrame:
        """Add code-specific known-future event, calendar, and session features."""
        event_presets = _load_event_presets_module()

        out = df.copy()
        idx = out.index
        dates = idx.date
        unique_dates = np.unique(dates)
        bar_minutes_total = (idx.hour * 60 + idx.minute).astype(np.int32)

        out["kt_day_of_week"] = idx.dayofweek.astype(np.float32)
        out["kt_hour"] = idx.hour.astype(np.float32)
        out["kt_session_label"] = self._session_label_from_index(idx)

        crack_event_codes = self.get_crack_event_codes(self.tickers)
        for code in crack_event_codes:
            event_def = event_presets.ALL_EVENT_DEFS.get(code)
            col_prefix = f"kt_evt_{code}"
            if event_def is None:
                out[f"{col_prefix}_is_release_day"] = np.float32(0.0)
                out[f"{col_prefix}_bars_to_release"] = np.float32(-1.0)
                out[f"{col_prefix}_is_pre_release"] = np.float32(0.0)
                continue

            release_info: Dict[date_type, int] = {}
            for current_date in unique_dates:
                if not event_presets.is_matching_event_slot(current_date, event_def):
                    continue
                release_dt = event_presets.event_release_dt_for_date(
                    event_def,
                    current_date,
                    instrument_tz="America/Chicago",
                )
                release_info[current_date] = release_dt.hour * 60 + release_dt.minute

            is_release_day = np.zeros(len(out), dtype=np.float32)
            bars_to_release = np.full(len(out), -1.0, dtype=np.float32)
            is_pre_release = np.zeros(len(out), dtype=np.float32)

            for current_date, rel_min in release_info.items():
                day_mask = dates == current_date
                if not np.any(day_mask):
                    continue
                is_release_day[day_mask] = 1.0
                day_bar_mins = bar_minutes_total[day_mask]
                diff = (rel_min - day_bar_mins).astype(np.float32)
                session_len = max(int(day_bar_mins.max() - day_bar_mins.min()), 1)
                bars_to_release[day_mask] = np.clip(diff / session_len, -1.0, 1.0)

                pre_window_mins = pre_release_bars * self.bar_minutes
                pre_mask = day_mask & (bar_minutes_total >= rel_min - pre_window_mins) & (bar_minutes_total < rel_min)
                is_pre_release[pre_mask] = 1.0

            out[f"{col_prefix}_is_release_day"] = is_release_day
            out[f"{col_prefix}_bars_to_release"] = bars_to_release
            out[f"{col_prefix}_is_pre_release"] = is_pre_release

        return out

    def load_orderflow_frames(
        self,
        orderflow_columns: Optional[Sequence[str]] = None,
        fold: Optional[bool] = None,
        n_folds: Optional[int] = None,
        start_ts: Optional[Union[str, pd.Timestamp]] = None,
        end_ts: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Load scaled per-asset VPIN streams without merging onto ``df_out``."""
        from CTAFlow.data.datasets.v3_continuous import scale_vpin_features

        frames: Dict[str, pd.DataFrame] = {}
        common_cols: Optional[set[str]] = None

        for ticker in self.tickers:
            raw = self._load_vpin_df(
                ticker,
                columns=orderflow_columns,
                fold=fold,
                n_folds=n_folds,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            numeric = raw.select_dtypes(include="number")
            if numeric.empty:
                frames[ticker] = pd.DataFrame(index=raw.index)
                continue
            del raw
            scaled = scale_vpin_features(numeric).astype(np.float32)
            del numeric
            frames[ticker] = _normalize_datetime_index_inplace(scaled)
            del scaled
            cols = set(frames[ticker].columns)
            common_cols = cols if common_cols is None else common_cols & cols

        if not frames:
            return {}, []

        first_ticker = next((ticker for ticker in self.tickers if ticker in frames), self.tickers[0])
        ordered_cols = [
            col for col in frames.get(first_ticker, pd.DataFrame()).columns
            if common_cols is None or col in common_cols
        ]
        for ticker in frames:
            frames[ticker] = frames[ticker].loc[:, ordered_cols]
        return frames, ordered_cols

    def add_targets_multistep(
        self,
        df: pd.DataFrame,
        steps: int = 12,
        prefix: str = "y_fwd",
    ) -> Tuple[pd.DataFrame, List[str]]:
        out = df.copy()
        target_cols: List[str] = []
        close = out["Close"].astype(float)
        logp = np.log(close.clip(lower=self.eps))
        rv = out["rv_1d"].astype(float).clip(lower=self.eps) if "rv_1d" in out.columns else None

        for h in range(1, steps + 1):
            col = f"{prefix}_{h}"
            if self.target_mode == "logret":
                out[col] = logp.shift(-h) - logp
            elif self.target_mode == "vol_norm_delta":
                raw_delta = close.shift(-h) - close
                out[col] = raw_delta / (rv + self.eps) if rv is not None else raw_delta
            else:
                out[col] = close.shift(-h) - close
            target_cols.append(col)

        return out, target_cols

    def prepare_from_root(
        self,
        steps_60m: int = 12,
        target_ticker: Optional[str] = None,
        keep_only_active: bool = False,
        add_daily: bool = True,
        add_overnight: bool = True,
        add_deseas: bool = True,
        add_time_features: bool = True,
        add_resample_precalc: bool = True,
        resample_rules: Tuple[str, ...] = ("15min", "30min", "60min"),
        rolling_days_deseas: int = 252,
        refit_interval: int = 10,
        use_legacy_deseas: bool = False,
        apply_scaling: bool = False,
        scale_to_basis_points: bool = True,
        add_bid_ask: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        raw = self.load_raw_inputs(load_orderflow=False)
        anchor = self.build_spread_anchor_df(raw["spread"], raw["asset_intraday"])
        target_df = self._build_target_frame(
            target_ticker=target_ticker,
            spread_df=raw["spread"],
            asset_intraday=raw["asset_intraday"],
            anchor_idx=anchor.index,
        )

        df_out, train_mask, target_cols = super().prepare(
            anchor,
            steps_60m=steps_60m,
            keep_only_active=keep_only_active,
            add_daily=add_daily,
            add_overnight=add_overnight,
            add_deseas=add_deseas,
            add_time_features=add_time_features,
            add_resample_precalc=add_resample_precalc,
            resample_rules=resample_rules,
            rolling_days_deseas=rolling_days_deseas,
            refit_interval=refit_interval,
            use_legacy_deseas=use_legacy_deseas,
            apply_scaling=False,
            scale_to_basis_points=scale_to_basis_points,
            add_bid_ask=add_bid_ask,
            ticker=None,
            add_event_markers=False,
        )

        df_out = self.add_spread_specific_features(df_out)
        resolved_target = self.normalize_target_ticker(target_ticker)
        if resolved_target != DEFAULT_TARGET_TICKER:
            df_out, target_cols = self._apply_target_frame(
                df_out,
                target_df=target_df,
                steps=steps_60m,
            )

        # Refresh mask after added feature/target transforms.
        train_mask = df_out["is_active"].astype(bool)
        for col in target_cols:
            train_mask &= df_out[col].notna()

        if keep_only_active:
            df_out = df_out.loc[train_mask].copy()

        if apply_scaling:
            df_out = self.scale_features(
                df_out,
                scale_to_basis_points=scale_to_basis_points,
                clip_outliers=True,
                inplace=False,
            )

        return df_out, train_mask, target_cols

    def build_indexed_dataset_from_root(
        self,
        spread_lookback: int = 64,
        orderflow_lookback: int = 128,
        orderflow_columns: Optional[Sequence[str]] = None,
        fold: Optional[bool] = None,
        n_folds: Optional[int] = None,
        vpin_start_ts: Optional[Union[str, pd.Timestamp]] = None,
        vpin_end_ts: Optional[Union[str, pd.Timestamp]] = None,
        target_ticker: Optional[str] = None,
        spread_feature_cols: Optional[Sequence[str]] = None,
        known_temporal_cols: Optional[Sequence[str]] = None,
        session_only: bool = True,
        sample_session: Optional[str] = None,
        sample_session_start: Optional[str] = None,
        sample_session_end: Optional[str] = None,
        stride: int = 1,
        require_all_assets: bool = True,
        include_known_temporal_features: bool = True,
        **prepare_kwargs,
    ):
        """Prepare root inputs and return a lazy indexed crack-spread dataset."""
        from CTAFlow.data.datasets.crack_spread_continuous import (
            build_crack_spread_indexed_dataset,
        )

        df_out, _, target_cols = self.prepare_from_root(
            target_ticker=target_ticker,
            **prepare_kwargs,
        )
        orderflow_frames, _ = self.load_orderflow_frames(
            orderflow_columns=orderflow_columns,
            fold=fold,
            n_folds=n_folds,
            start_ts=vpin_start_ts,
            end_ts=vpin_end_ts,
        )
        target_col = target_cols[-1]
        return build_crack_spread_indexed_dataset(
            df_out=df_out,
            orderflow_frames=orderflow_frames,
            target_col=target_col,
            spread_feature_cols=spread_feature_cols,
            known_temporal_cols=known_temporal_cols,
            tickers=self.tickers,
            spread_lookback=spread_lookback,
            orderflow_lookback=orderflow_lookback,
            session_only=session_only,
            sample_session=sample_session,
            sample_session_start=sample_session_start,
            sample_session_end=sample_session_end,
            stride=stride,
            require_all_assets=require_all_assets,
            include_known_temporal_features=include_known_temporal_features,
        )
