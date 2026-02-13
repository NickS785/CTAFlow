from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from CTAFlow.data.classifications_reference import COMMODITY_TICKERS, FINANCIAL_TICKERS


DEFAULT_MARKET_TICKERS: Dict[str, Dict[str, Any]] = {
    "^GSPC": {"name": "sp500", "category": "stock_index", "use_level": False},
    "^IXIC": {"name": "nasdaq", "category": "stock_index", "use_level": False},
    "^DJI": {"name": "dow_jones", "category": "stock_index", "use_level": False},
    "^RUT": {"name": "russell_2000", "category": "stock_index", "use_level": False},
    "^STOXX50E": {"name": "euro_stoxx_50", "category": "stock_index", "use_level": False},
    "^N225": {"name": "nikkei_225", "category": "stock_index", "use_level": False},
    "CL=F": {"name": "crude_oil", "category": "commodity", "use_level": False},
    "GC=F": {"name": "gold", "category": "commodity", "use_level": False},
    "SI=F": {"name": "silver", "category": "commodity", "use_level": False},
    "^VIX": {"name": "vix", "category": "volatility", "use_level": True},
    "DX-Y.NYB": {"name": "us_dollar_index", "category": "currency_index", "use_level": True},
    "AGG": {"name": "corp_bond_index", "category": "bond", "use_level": False},
    "^TNX": {"name": "10y_treasury", "category": "treasury", "use_level": True},
    "^IRX": {"name": "3m_treasury", "category": "treasury", "use_level": True},
}


def _safe_cache_name(symbol: str) -> str:
    return (
        symbol.replace("^", "_hat_")
        .replace("=", "_eq_")
        .replace(".", "_dot_")
        .replace("/", "_slash_")
    )


def _normalize_col_label(label: Any) -> str:
    return str(label).strip().lower().replace("_", " ")


def _select_ohlcv_columns_from_multiindex(
    frame: pd.DataFrame,
    preferred_symbol: Optional[str] = None,
) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    # 1) If symbol level exists, try selecting that symbol first.
    if preferred_symbol is not None:
        for lvl in range(frame.columns.nlevels):
            vals = [str(v) for v in frame.columns.get_level_values(lvl)]
            if preferred_symbol in vals:
                try:
                    picked = frame.xs(preferred_symbol, axis=1, level=lvl, drop_level=True)
                    if isinstance(picked, pd.Series):
                        picked = picked.to_frame()
                    if not isinstance(picked.columns, pd.MultiIndex):
                        return picked
                    frame = picked
                except Exception:
                    pass

    # 2) Identify which level contains OHLCV labels and use that level as columns.
    ohlcv_keys = {"open", "high", "low", "close", "adj close", "volume"}
    best_level = None
    best_hits = -1
    for lvl in range(frame.columns.nlevels):
        vals = [_normalize_col_label(v) for v in frame.columns.get_level_values(lvl)]
        hits = sum(v in ohlcv_keys for v in vals)
        if hits > best_hits:
            best_hits = hits
            best_level = lvl

    if best_level is not None and best_hits > 0:
        out = frame.copy()
        out.columns = out.columns.get_level_values(best_level)
        return out

    # 3) Fallback: drop the last level.
    return frame.droplevel(-1, axis=1)


def _standardize_ohlcv(df: pd.DataFrame, preferred_symbol: Optional[str] = None) -> pd.DataFrame:
    frame = df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame = _select_ohlcv_columns_from_multiindex(frame, preferred_symbol=preferred_symbol)
    frame.columns = [str(c).strip() for c in frame.columns]

    col_map: Dict[str, str] = {}
    for col in frame.columns:
        lc = col.lower()
        if lc == "open":
            col_map[col] = "open"
        elif lc == "high":
            col_map[col] = "high"
        elif lc == "low":
            col_map[col] = "low"
        elif lc == "close":
            col_map[col] = "close"
        elif "adj" in lc and "close" in lc:
            col_map[col] = "adj_close"
        elif lc == "volume":
            col_map[col] = "volume"
    frame = frame.rename(columns=col_map)

    if "close" not in frame.columns:
        raise ValueError("missing required price column 'close'")
    if "open" not in frame.columns:
        frame["open"] = frame["close"]
    if "high" not in frame.columns:
        frame["high"] = frame["close"]
    if "low" not in frame.columns:
        frame["low"] = frame["close"]
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    frame = frame.sort_index()
    frame.index = pd.to_datetime(frame.index)
    return frame[["open", "high", "low", "close", "adj_close", "volume"]]


def _resample_to_daily(df: pd.DataFrame, rule: str = "1D") -> pd.DataFrame:
    frame = _standardize_ohlcv(df)
    if frame.index.tz is not None:
        frame = frame.tz_convert("UTC").tz_localize(None)
    else:
        frame.index = frame.index.tz_localize(None)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "adj_close": "last",
        "volume": "sum",
    }
    out = frame.resample(rule).agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.sort_index()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=max(window // 4, 2)).mean()
    sigma = series.rolling(window, min_periods=max(window // 4, 2)).std() + 1e-10
    return (series - mu) / sigma


def _default_asset_metadata() -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    for ticker, info in COMMODITY_TICKERS.items():
        meta[ticker] = {
            "name": info.get("name", ticker),
            "category": info.get("category", "Commodity"),
            "classification": "Commodity",
        }
    for ticker, info in FINANCIAL_TICKERS.items():
        meta[ticker] = {
            "name": info.get("name", ticker),
            "category": info.get("category", "Financial"),
            "classification": "Financial",
        }
    return meta


def default_asset_tickers() -> List[str]:
    return sorted(_default_asset_metadata().keys())


def _default_yahoo_symbol(asset_ticker: str) -> str:
    if asset_ticker.endswith("_F"):
        return f"{asset_ticker[:-2]}=F"
    return asset_ticker


def _to_classification_ticker(symbol: str) -> str:
    if symbol.endswith("=F"):
        return f"{symbol[:-2]}_F"
    return symbol


def download_tickers(
    keys: Sequence[str],
    key_to_symbol: Optional[Mapping[str, str]] = None,
    start: str = "2000-01-01",
    end: str = "2024-01-01",
    cache_dir: str = "./data_cache",
    interval: str = "1d",
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV frames via yfinance and cache each symbol independently.

    `keys` are logical names returned in the output dictionary.
    `key_to_symbol` maps each logical key to the download symbol.
    """
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("yfinance is required for download_tickers()") from exc

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    output: Dict[str, pd.DataFrame] = {}
    symbol_map = {k: key_to_symbol[k] if key_to_symbol and k in key_to_symbol else k for k in keys}

    for key in keys:
        symbol = symbol_map[key]
        cache_file = cache_path / f"{_safe_cache_name(symbol)}_{interval}.pkl"

        if cache_file.exists() and not force_refresh:
            frame = pd.read_pickle(cache_file)
        else:
            frame = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if frame is None or frame.empty:
                continue
            frame = _standardize_ohlcv(frame, preferred_symbol=symbol)
            frame.to_pickle(cache_file)

        if frame is None or frame.empty:
            continue
        output[key] = frame

    return output


@dataclass
class AssetUniverse:
    tickers: List[str]
    class_map: Dict[str, int]
    class_names: Dict[int, str]
    ticker_to_download: Dict[str, str]
    ticker_meta: Dict[str, Dict[str, str]]


def build_asset_universe(
    asset_tickers: Optional[Sequence[str]] = None,
    ticker_download_map: Optional[Mapping[str, str]] = None,
    classification_group: str = "category",
) -> AssetUniverse:
    metadata = _default_asset_metadata()
    selected_in = list(asset_tickers) if asset_tickers is not None else default_asset_tickers()
    selected_in = [str(t).strip() for t in selected_in if str(t).strip()]

    # Canonicalize yfinance symbols (e.g., CL=F) to classification keys (e.g., CL_F)
    selected: List[str] = []
    original_by_canonical: Dict[str, str] = {}
    for ticker in selected_in:
        canonical = ticker
        if ticker not in metadata:
            maybe_cls = _to_classification_ticker(ticker)
            if maybe_cls in metadata:
                canonical = maybe_cls
        if canonical not in original_by_canonical:
            original_by_canonical[canonical] = ticker
            selected.append(canonical)

    labels: Dict[str, str] = {}
    for ticker in selected:
        info = metadata.get(ticker)
        if info is None:
            labels[ticker] = "Unknown"
            metadata[ticker] = {"name": ticker, "category": "Unknown", "classification": "Unknown"}
            continue
        if classification_group == "classification":
            labels[ticker] = info["classification"]
        elif classification_group == "category":
            labels[ticker] = info["category"]
        else:
            labels[ticker] = f"{info['classification']}::{info['category']}"

    class_names_sorted = sorted(set(labels.values()))
    class_name_to_id = {name: idx for idx, name in enumerate(class_names_sorted)}
    class_map = {ticker: class_name_to_id[labels[ticker]] for ticker in selected}
    class_names = {idx: name for name, idx in class_name_to_id.items()}

    ticker_to_download: Dict[str, str] = {}
    for ticker in selected:
        src = original_by_canonical.get(ticker, ticker)
        if ticker_download_map and src in ticker_download_map:
            ticker_to_download[ticker] = str(ticker_download_map[src])
        elif ticker_download_map and ticker in ticker_download_map:
            ticker_to_download[ticker] = str(ticker_download_map[ticker])
        elif src.endswith("=F"):
            ticker_to_download[ticker] = src
        else:
            ticker_to_download[ticker] = _default_yahoo_symbol(ticker)

    return AssetUniverse(
        tickers=selected,
        class_map=class_map,
        class_names=class_names,
        ticker_to_download=ticker_to_download,
        ticker_meta=metadata,
    )


class MarketFeatureEngine:
    """
    Daily market feature builder used by multi-asset qLSTM pipelines.

    Supports either:
    1) downloading selected market tickers, or
    2) using externally supplied raw OHLCV frames (resampled to daily).
    """

    def __init__(
        self,
        start: str = "2000-01-01",
        end: str = "2024-01-01",
        zscore_window: int = 219,
        cache_dir: str = "./data_cache",
        market_tickers: Optional[Sequence[str]] = None,
        market_meta: Optional[Mapping[str, Dict[str, Any]]] = None,
        raw_data: Optional[Mapping[str, pd.DataFrame]] = None,
        resample_rule: str = "1D",
        download_all: bool = True,
    ):
        self.start = start
        self.end = end
        self.zscore_window = int(zscore_window)
        self.cache_dir = cache_dir
        self.resample_rule = resample_rule

        self.market_meta = dict(market_meta or DEFAULT_MARKET_TICKERS)
        self.download_all = bool(download_all)
        if self.download_all:
            self.selected_tickers = list(self.market_meta.keys())
        else:
            self.selected_tickers = (
                list(market_tickers) if market_tickers is not None else list(self.market_meta.keys())
            )
        self.input_raw_data = dict(raw_data) if raw_data is not None else None

        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.feature_names: List[str] = []
        self.market_df: Optional[pd.DataFrame] = None

    def download(self) -> "MarketFeatureEngine":
        if self.download_all:
            self.selected_tickers = list(self.market_meta.keys())

        # Always pull the full market set when download_all=True.
        # If raw_data is supplied, treat it as a per-ticker override layer.
        if self.download_all:
            self.raw_data = download_tickers(
                keys=self.selected_tickers,
                start=self.start,
                end=self.end,
                cache_dir=self.cache_dir,
                interval="1d",
            )
            if self.input_raw_data is not None:
                for ticker in self.selected_tickers:
                    if ticker not in self.input_raw_data:
                        continue
                    self.raw_data[ticker] = _resample_to_daily(
                        self.input_raw_data[ticker], self.resample_rule
                    )
            return self

        if self.input_raw_data is not None:
            for ticker in self.selected_tickers:
                if ticker not in self.input_raw_data:
                    continue
                self.raw_data[ticker] = _resample_to_daily(self.input_raw_data[ticker], self.resample_rule)
        else:
            self.raw_data = download_tickers(
                keys=self.selected_tickers,
                start=self.start,
                end=self.end,
                cache_dir=self.cache_dir,
                interval="1d",
            )
        return self

    def build_features(self) -> pd.DataFrame:
        if not self.raw_data:
            raise RuntimeError("No market data loaded. Call download() or provide raw_data.")

        feature_series: Dict[str, pd.Series] = {}
        for ticker in self.selected_tickers:
            if ticker not in self.raw_data:
                continue
            meta = self.market_meta.get(
                ticker,
                {
                    "name": ticker.lower().replace("^", "").replace("=", "_"),
                    "category": "custom",
                    "use_level": False,
                },
            )

            frame = self.raw_data[ticker]
            price_col = "adj_close" if "adj_close" in frame.columns else "close"
            prices = frame[price_col].dropna().astype(np.float64)
            if len(prices) < 50:
                continue

            name = str(meta.get("name", ticker)).lower().replace(" ", "_")
            if bool(meta.get("use_level", False)):
                feature_series[name] = prices
            else:
                feature_series[name] = np.log(prices / prices.shift(1))

        if "^TNX" in self.raw_data and "^IRX" in self.raw_data:
            tnx = self.raw_data["^TNX"]["adj_close"].astype(np.float64)
            irx = self.raw_data["^IRX"]["adj_close"].astype(np.float64)
            idx = tnx.index.intersection(irx.index)
            if len(idx) > 50:
                feature_series["yield_curve_slope"] = tnx.reindex(idx) - irx.reindex(idx)

        if not feature_series:
            raise RuntimeError("No market features could be computed from current inputs.")

        combined = pd.DataFrame(feature_series).sort_index()
        combined = combined.ffill().bfill().fillna(0.0)
        normalized = combined.apply(lambda col: _rolling_zscore(col, self.zscore_window))
        normalized = normalized.clip(-5.0, 5.0).ffill().bfill().fillna(0.0)

        self.feature_names = list(normalized.columns)
        self.market_df = normalized
        return normalized

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


class AssetFeatureEngine:
    """
    Daily per-asset feature builder for multi-asset qLSTM training.

    Asset ticker selection is configurable. By default it uses the repository's
    classification reference rather than a static hardcoded paper list.
    """

    RETURN_WINDOWS = [2, 5, 22]
    VOL_WINDOWS = [2, 5, 22]
    MA_WINDOWS = [2, 5, 22]
    SHARPE_WINDOWS = [2, 5, 22]

    def __init__(
        self,
        start: str = "2000-01-01",
        end: str = "2024-01-01",
        zscore_window: int = 219,
        ewma_lambda: float = 0.94,
        cache_dir: str = "./data_cache",
        asset_tickers: Optional[Sequence[str]] = None,
        ticker_download_map: Optional[Mapping[str, str]] = None,
        classification_group: str = "category",
        raw_data: Optional[Mapping[str, pd.DataFrame]] = None,
        resample_rule: str = "1D",
    ):
        self.start = start
        self.end = end
        self.zscore_window = int(zscore_window)
        self.ewma_lambda = float(ewma_lambda)
        self.cache_dir = cache_dir
        self.resample_rule = resample_rule

        self.universe = build_asset_universe(
            asset_tickers=asset_tickers,
            ticker_download_map=ticker_download_map,
            classification_group=classification_group,
        )
        self.input_raw_data = dict(raw_data) if raw_data is not None else None

        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.asset_features: Dict[str, pd.DataFrame] = {}
        self.asset_log_returns: Dict[str, pd.Series] = {}
        self.feature_names: List[str] = []

    @property
    def asset_tickers(self) -> List[str]:
        return self.universe.tickers

    @property
    def asset_class_map(self) -> Dict[str, int]:
        return dict(self.universe.class_map)

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self.universe.class_names)

    def download(self) -> "AssetFeatureEngine":
        if self.input_raw_data is not None:
            for ticker in self.asset_tickers:
                symbol = self.universe.ticker_to_download[ticker]
                if ticker in self.input_raw_data:
                    frame = self.input_raw_data[ticker]
                elif symbol in self.input_raw_data:
                    frame = self.input_raw_data[symbol]
                else:
                    continue
                self.raw_data[ticker] = _resample_to_daily(frame, self.resample_rule)
            return self

        downloaded = download_tickers(
            keys=self.asset_tickers,
            key_to_symbol=self.universe.ticker_to_download,
            start=self.start,
            end=self.end,
            cache_dir=self.cache_dir,
            interval="1d",
        )
        self.raw_data = downloaded
        return self

    @staticmethod
    def _log_returns(prices: pd.Series) -> pd.Series:
        return np.log(prices / prices.shift(1))

    @staticmethod
    def _multi_period_returns(log_ret: pd.Series, windows: Sequence[int]) -> Dict[str, pd.Series]:
        return {f"ret_{w}d": log_ret.rolling(int(w)).sum() for w in windows}

    @staticmethod
    def _cumulative_return(prices: pd.Series) -> pd.Series:
        first = float(prices.iloc[0]) if len(prices) else 1.0
        return prices / max(first, 1e-10) - 1.0

    @staticmethod
    def _realized_vol(log_ret: pd.Series, windows: Sequence[int]) -> Dict[str, pd.Series]:
        return {f"vol_{w}d": log_ret.rolling(int(w)).std() for w in windows}

    @staticmethod
    def _sma_ratio(prices: pd.Series, windows: Sequence[int]) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for w in windows:
            sma = prices.rolling(int(w)).mean()
            out[f"sma_{w}d_ratio"] = prices / (sma + 1e-10) - 1.0
        return out

    @staticmethod
    def _ema_ratio(prices: pd.Series, windows: Sequence[int]) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for w in windows:
            ema = prices.ewm(span=int(w), adjust=False).mean()
            out[f"ema_{w}d_ratio"] = prices / (ema + 1e-10) - 1.0
        return out

    @staticmethod
    def _rsi(log_ret: pd.Series, window: int = 14) -> pd.Series:
        gains = log_ret.clip(lower=0.0)
        losses = (-log_ret).clip(lower=0.0)
        avg_gain = gains.rolling(window).mean()
        avg_loss = losses.rolling(window).mean() + 1e-10
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = (ema12 - ema26) / (prices + 1e-10)
        signal = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line, signal

    @staticmethod
    def _bollinger_pctb(prices: pd.Series, window: int = 20) -> pd.Series:
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std() + 1e-10
        upper = sma + 2.0 * std
        lower = sma - 2.0 * std
        return (prices - lower) / (upper - lower + 1e-10)

    @staticmethod
    def _stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        lowest = low.rolling(window).min()
        highest = high.rolling(window).max()
        return (close - lowest) / (highest - lowest + 1e-10) * 100.0

    @staticmethod
    def _sharpe_ratios(log_ret: pd.Series, windows: Sequence[int]) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for w in windows:
            mu = log_ret.rolling(int(w)).mean()
            sigma = log_ret.rolling(int(w)).std() + 1e-10
            out[f"sharpe_{w}d"] = mu / sigma
        return out

    @staticmethod
    def _high_low_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        return (high - low) / (close + 1e-10)

    def _build_single_asset(self, ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
        price_col = "adj_close" if "adj_close" in frame.columns else "close"
        prices = frame[price_col].dropna().astype(np.float64)
        if len(prices) < 100:
            raise ValueError(f"{ticker} has insufficient observations ({len(prices)})")

        has_hl = ("high" in frame.columns) and ("low" in frame.columns)
        log_ret = self._log_returns(prices)
        self.asset_log_returns[ticker] = log_ret

        feats: Dict[str, pd.Series] = {}
        feats["log_ret"] = log_ret
        feats.update(self._multi_period_returns(log_ret, self.RETURN_WINDOWS))
        feats["cum_ret"] = self._cumulative_return(prices)
        feats.update(self._realized_vol(log_ret, self.VOL_WINDOWS))
        feats["skew_22d"] = log_ret.rolling(22).skew()
        feats["kurt_22d"] = log_ret.rolling(22).kurt()
        feats.update(self._sma_ratio(prices, self.MA_WINDOWS))
        feats.update(self._ema_ratio(prices, self.MA_WINDOWS))

        feats["rsi_14"] = self._rsi(log_ret, 14)
        macd_line, macd_signal = self._macd(prices)
        feats["macd_line"] = macd_line
        feats["macd_signal"] = macd_signal
        feats["macd_hist"] = macd_line - macd_signal
        feats["boll_pctb"] = self._bollinger_pctb(prices, 20)

        if has_hl:
            high = frame["high"].reindex(prices.index).astype(np.float64)
            low = frame["low"].reindex(prices.index).astype(np.float64)
            feats["stoch_k"] = self._stochastic_k(high, low, prices, 14)
            feats["hl_range"] = self._high_low_range(high, low, prices)
        else:
            feats["stoch_k"] = pd.Series(50.0, index=prices.index)
            feats["hl_range"] = pd.Series(0.0, index=prices.index)

        feats.update(self._sharpe_ratios(log_ret, self.SHARPE_WINDOWS))
        if "volume" in frame.columns:
            vol = frame["volume"].reindex(prices.index).astype(np.float64)
            vol = vol.replace(0.0, np.nan)
            if vol.notna().sum() > 50:
                feats["volume_zscore"] = _rolling_zscore(np.log1p(vol.ffill()), self.zscore_window)
            else:
                feats["volume_zscore"] = pd.Series(0.0, index=prices.index)
        else:
            feats["volume_zscore"] = pd.Series(0.0, index=prices.index)

        feat_df = pd.DataFrame(feats).ffill().bfill().fillna(0.0)
        normalized = feat_df.apply(lambda col: _rolling_zscore(col, self.zscore_window))
        normalized = normalized.clip(-5.0, 5.0).ffill().bfill().fillna(0.0)
        return normalized

    def build_features(self) -> Dict[str, pd.DataFrame]:
        if not self.raw_data:
            raise RuntimeError("No asset data loaded. Call download() or provide raw_data.")

        out: Dict[str, pd.DataFrame] = {}
        for ticker, frame in self.raw_data.items():
            try:
                feat_df = self._build_single_asset(ticker, frame)
            except Exception:
                continue
            out[ticker] = feat_df
            if not self.feature_names:
                self.feature_names = list(feat_df.columns)

        self.asset_features = out
        return out

    @property
    def num_features(self) -> int:
        return len(self.feature_names)
