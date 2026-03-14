"""
Data preparation for the Heterogeneous Macro GAT.

Produces two types of node feature tensors:
  - Asset nodes (Gold, DXY, US_2Y, US_10Y, TIPS): 10-feature daily vectors
    derived from price/yield series (return, rolling vols, vol-weighted
    returns, MA distances).
  - Economic nodes (Inflation, Labor, Growth): grouped YoY indicators,
    forward-filled to the daily grid.

Leverages MacroClient for FRED data and yfinance for Gold/DXY.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FRED series for yields + TIPS
# ---------------------------------------------------------------------------
GAT_FRED_SERIES: Dict[str, str] = {
    "YIELD_10Y": "DGS10",
    "YIELD_2Y": "DGS2",
    "TIPS_10Y": "DFII10",  # 10Y TIPS real yield → inflation expectations
}

# Yahoo tickers for price-based asset nodes
GAT_YAHOO_ASSETS: Dict[str, str] = {
    "Gold": "GC=F",
    "DXY": "DX-Y.NYB",
}

# Economic node groupings — each maps to columns in MacroClient.fetch_econ_data()
ECON_NODE_MAP: Dict[str, List[str]] = {
    "Inflation": ["CPI_YOY", "CORE_PCE_YOY"],
    "Labor": ["UNRATE", "PAYEMS_YOY"],
    "Growth": ["NGDP_YOY", "UMCSENT"],
}

# Asset node ordering (must match GAT model's node_order)
ASSET_NODE_ORDER = ["Gold", "DXY", "US_2Y", "US_10Y", "TIPS"]
ECON_NODE_ORDER = ["Inflation", "Labor", "Growth"]
NODE_ORDER = ASSET_NODE_ORDER + ECON_NODE_ORDER

# Asset feature engineering windows
VOL_WINDOWS = [5, 22, 63]
MA_WINDOWS = [20, 50, 200]


def _build_asset_features(
    series: pd.Series,
    is_yield: bool = False,
) -> pd.DataFrame:
    """
    Build 10-feature vector for a single asset node.

    For price-based assets: log returns, rolling vols, vol-weighted returns,
    MA distances.
    For yield-based assets: diff changes instead of log returns, but same
    structure.

    Returns DataFrame with 10 columns, same index as input.
    """
    series = series.dropna().astype(np.float64)

    if is_yield:
        # Yield series: use first differences (basis point changes)
        daily_change = series.diff(1)
    else:
        # Price series: log returns
        daily_change = np.log(series / series.shift(1))

    feats: Dict[str, pd.Series] = {}

    # 1) Daily return/change
    feats["ret_1d"] = daily_change

    # 2-4) Rolling volatilities
    for w in VOL_WINDOWS:
        feats[f"vol_{w}d"] = daily_change.rolling(w, min_periods=max(w // 2, 2)).std()

    # 5-7) Volatility-weighted returns (Sharpe-like)
    for w in VOL_WINDOWS:
        vol = feats[f"vol_{w}d"]
        cum_ret = daily_change.rolling(w, min_periods=max(w // 2, 2)).sum()
        feats[f"vwr_{w}d"] = cum_ret / (vol + 1e-10)

    # 8-10) Moving average distances
    for w in MA_WINDOWS:
        ma = series.rolling(w, min_periods=max(w // 2, 2)).mean()
        if is_yield:
            # For yields, distance in absolute terms (bps)
            feats[f"mad_{w}d"] = series - ma
        else:
            # For prices, relative distance
            feats[f"mad_{w}d"] = series / (ma + 1e-10) - 1.0

    out = pd.DataFrame(feats, index=series.index)
    return out.ffill().bfill().fillna(0.0)


class MacroGATPrep:
    """
    End-to-end data preparation for the Heterogeneous Macro GAT.

    Usage::

        prep = MacroGATPrep()
        node_dict = prep.fetch_and_build(
            start_date=datetime(2010, 1, 1),
            end_date=datetime(2024, 1, 1),
        )
        # node_dict keys: Gold, DXY, US_2Y, US_10Y, TIPS, Inflation, Labor, Growth
        # Asset nodes: 10 features each. Econ nodes: Inflation=2, Labor=2, Growth=2.
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key
        self._fred = None

    def _get_fred(self):
        if self._fred is not None:
            return self._fred
        try:
            from fredapi import Fred
        except ImportError:
            logger.error("fredapi not installed — pip install fredapi")
            return None
        import os
        key = self.fred_api_key or os.getenv("FRED_API_KEY")
        if not key:
            logger.error("FRED_API_KEY not set")
            return None
        self._fred = Fred(api_key=key)
        return self._fred

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_fred_yields(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch DGS10, DGS2, DFII10 (TIPS) from FRED."""
        fred = self._get_fred()
        if fred is None:
            return pd.DataFrame()

        end = end_date or datetime.utcnow()
        frames = {}
        for alias, sid in GAT_FRED_SERIES.items():
            try:
                s = fred.get_series(sid, observation_start=start_date, observation_end=end)
                if s is not None and not s.empty:
                    frames[alias] = s
            except Exception as e:
                logger.warning("FRED %s (%s) failed: %s", alias, sid, e)

        if not frames:
            return pd.DataFrame()
        out = pd.DataFrame(frames)
        out.index = pd.to_datetime(out.index)
        return out.sort_index().ffill()

    def fetch_yahoo_assets(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.Series]:
        """Fetch Gold and DXY daily close from yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return {}

        result = {}
        for name, ticker in GAT_YAHOO_ASSETS.items():
            try:
                raw = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                )
                if raw is not None and not raw.empty:
                    close = raw["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    close.index = pd.to_datetime(close.index)
                    result[name] = close.sort_index().dropna()
            except Exception as e:
                logger.warning("yfinance %s (%s) failed: %s", name, ticker, e)

        return result

    def fetch_econ_data(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch economic indicators via MacroClient and return YoY values."""
        from CTAFlow.data.ext.macro_client import MacroClient

        client = MacroClient(fred_api_key=self.fred_api_key)
        econ = client.fetch_econ_data(start_date=start_date, end_date=end_date)
        return econ

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------

    def build_asset_nodes(
        self,
        yahoo_assets: Dict[str, pd.Series],
        fred_yields: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Build 10-feature DataFrames for each asset node."""
        nodes: Dict[str, pd.DataFrame] = {}

        # Price-based nodes (Gold, DXY)
        for name in ["Gold", "DXY"]:
            if name in yahoo_assets:
                nodes[name] = _build_asset_features(yahoo_assets[name], is_yield=False)

        # Yield-based nodes (US_2Y, US_10Y, TIPS)
        yield_map = {
            "US_2Y": "YIELD_2Y",
            "US_10Y": "YIELD_10Y",
            "TIPS": "TIPS_10Y",
        }
        for node_name, col in yield_map.items():
            if col in fred_yields.columns:
                nodes[node_name] = _build_asset_features(
                    fred_yields[col], is_yield=True
                )

        return nodes

    def build_econ_nodes(
        self,
        econ_df: pd.DataFrame,
        daily_index: pd.DatetimeIndex,
    ) -> Dict[str, pd.DataFrame]:
        """
        Build economic node DataFrames by grouping indicators.

        Each node's features are the YoY values (or levels for UNRATE/UMCSENT),
        forward-filled to the daily grid.
        """
        # Forward-fill econ data to daily
        econ_daily = econ_df.reindex(daily_index).ffill()

        nodes: Dict[str, pd.DataFrame] = {}
        for node_name, columns in ECON_NODE_MAP.items():
            available = [c for c in columns if c in econ_daily.columns]
            if not available:
                logger.warning("No columns found for econ node %s", node_name)
                continue
            node_df = econ_daily[available].fillna(0.0)
            nodes[node_name] = node_df

        return nodes

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fetch_and_build(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all data and build node feature DataFrames.

        Returns a dict mapping node names to DataFrames, all aligned to
        the same daily business-day index. Asset nodes have 10 features,
        economic nodes have variable features (Inflation=3, Labor=2, Growth=2).
        """
        # Fetch raw data
        fred_yields = self.fetch_fred_yields(start_date, end_date)
        yahoo_assets = self.fetch_yahoo_assets(start_date, end_date)
        econ_df = self.fetch_econ_data(start_date, end_date)

        # Build asset nodes
        asset_nodes = self.build_asset_nodes(yahoo_assets, fred_yields)

        # Determine common daily index from asset nodes
        all_indices = [df.index for df in asset_nodes.values()]
        if not all_indices:
            raise ValueError("No asset data could be fetched")
        daily_index = all_indices[0]
        for idx in all_indices[1:]:
            daily_index = daily_index.union(idx)
        daily_index = daily_index.sort_values()

        # Build econ nodes
        econ_nodes = self.build_econ_nodes(econ_df, daily_index)

        # Align all nodes to common index
        result: Dict[str, pd.DataFrame] = {}
        for name, df in {**asset_nodes, **econ_nodes}.items():
            aligned = df.reindex(daily_index).ffill().bfill().fillna(0.0)
            result[name] = aligned

        # Summary
        logger.info(
            "MacroGATPrep: %d nodes, %d daily obs, date range %s → %s",
            len(result),
            len(daily_index),
            daily_index[0].strftime("%Y-%m-%d"),
            daily_index[-1].strftime("%Y-%m-%d"),
        )
        for name, df in result.items():
            logger.info("  %s: %d features %s", name, df.shape[1], list(df.columns))

        return result

    @staticmethod
    def to_tensors(
        node_dict: Dict[str, pd.DataFrame],
        seq_len: int = 20,
    ) -> Tuple[Dict[str, np.ndarray], pd.DatetimeIndex]:
        """
        Convert node DataFrames to rolling-window numpy arrays.

        Returns:
            tensors: Dict mapping node names to arrays of shape
                     [num_samples, seq_len, num_features]
            dates: DatetimeIndex of valid prediction dates (aligned to
                   last day of each window)
        """
        # Get common index
        ref_name = list(node_dict.keys())[0]
        ref_index = node_dict[ref_name].index
        n = len(ref_index)

        if n <= seq_len:
            raise ValueError(
                f"Not enough data ({n} days) for seq_len={seq_len}"
            )

        tensors: Dict[str, np.ndarray] = {}
        for name, df in node_dict.items():
            vals = df.values.astype(np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(
                vals, window_shape=seq_len, axis=0
            )
            # sliding_window_view gives [n - seq_len + 1, features, seq_len]
            # transpose to [n - seq_len + 1, seq_len, features]
            tensors[name] = windows.transpose(0, 2, 1).copy()

        dates = ref_index[seq_len - 1:]
        return tensors, dates

    @staticmethod
    def node_feature_dims(node_dict: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Return the number of features per node."""
        return {name: df.shape[1] for name, df in node_dict.items()}
