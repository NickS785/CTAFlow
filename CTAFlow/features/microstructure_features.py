# trade_only_microfeatures.py
# Research-grade, trade-only microstructure features.
# Now supports BOTH per-trade (with 'side') AND Sierra aggregated CSVs (AskTradeVolume/BidTradeVolume).
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict

# -------------------- Config --------------------
@dataclass
class Params:
    # Common
    resample: str = "1s"
    delta: str = "5s"
    base: str  = "120s"
    lag_autocorr: int = 1
    price_col: str = "price"
    size_col: str  = "size"    # per-trade mode only
    side_col: str  = "side"    # per-trade mode only ('B','S')
    normalize_qcum: bool = True
    large_trade_thresh: float = 0.0
    rv_method: str = "std"
    assume_tz: str = "America/Chicago"   # used if timestamps are naive

    # Sierra column hints (auto-detect if None)
    ts_hint: Optional[str]   = None
    last_hint: Optional[str] = None
    ask_trd_hint: Optional[str] = None
    bid_trd_hint: Optional[str] = None
    ntrades_hint: Optional[str] = None

# -------------------- Utilities --------------------
def _ensure_sorted(df: pd.DataFrame, ts: str = "timestamp") -> pd.DataFrame:
    if df.index.name != ts:
        d = df.copy()
        if ts in d.columns:
            d = d.sort_values(ts).set_index(ts)
        else:
            d = d.sort_index()
        return d
    if not df.index.is_monotonic_increasing:
        return df.sort_index()
    return df

def _has_per_trade_side(d: pd.DataFrame, params: Params) -> bool:
    return params.side_col in d.columns

def _detect_sierra_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    def norm(s): return s.lower().replace(" ", "").replace("_","")
    cols = {norm(c): c for c in df.columns}
    def pick(*cands):
        for k in cands:
            kk = norm(k)
            if kk in cols: return cols[kk]
        return None
    return dict(
        ts  = pick("DateTime","datetime","timestamp","SCDateTime","StartDateTime","Date Time"),
        px  = pick("Last","Close","Price","LastPrice","Settlement"),
        ask = pick("AskTradeVolume","Ask Trade Volume","AskTradeVol","UpVolume","UpTickVolume"),
        bid = pick("BidTradeVolume","Bid Trade Volume","BidTradeVol","DownVolume","DownTickVolume"),
        ntr = pick("NumberOfTrades","Trades","TradeCount","#Trades"),
    )

# -------------------- Per-trade path --------------------
def _signed(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    d = _ensure_sorted(df)
    if params.side_col not in d.columns:
        raise ValueError("Missing 'side' column with values {'B','S'} for per-trade path.")
    d["sign"] = np.where(d[params.side_col].values == "B", 1, -1).astype(np.int8)
    d["flow"] = d["sign"] * d[params.size_col].astype(float)
    d["ret"]  = d[params.price_col].astype(float).diff()
    return d

def _to_grid_per_trade(d: pd.DataFrame, params: Params) -> pd.DataFrame:
    g = pd.DataFrame({
        "price": d[params.price_col].astype(float).resample(params.resample).last().ffill(),
        "n_trades": d["sign"].resample(params.resample).size(),
        "buy_count": (d["sign"]>0).resample(params.resample).sum().astype("float"),
        "sell_count": (d["sign"]<0).resample(params.resample).sum().astype("float"),
        "buy_vol": (d["flow"].where(d["sign"]>0, 0.0)).resample(params.resample).sum(),
        "sell_vol": (-d["flow"].where(d["sign"]<0, 0.0)).resample(params.resample).sum(),
        "flow_sum": d["flow"].resample(params.resample).sum(),
        "sign_rate": d["sign"].resample(params.resample).mean(),
    })
    g.index.name = "timestamp"
    return g

# -------------------- Sierra path --------------------
def _to_grid_sierra(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    # Auto-detect columns unless hints provided
    hints = _detect_sierra_cols(df)
    ts_col   = params.ts_hint   or hints["ts"]
    last_col = params.last_hint or hints["px"]
    ask_col  = params.ask_trd_hint or hints["ask"]
    bid_col  = params.bid_trd_hint or hints["bid"]
    ntr_col  = params.ntrades_hint or hints["ntr"]

    if ts_col is None or last_col is None:
        raise ValueError(f"Could not detect timestamp/price in Sierra data. Columns: {list(df.columns)}")

    # Parse timestamp; localize if naive
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(params.assume_tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    d = pd.DataFrame(index=ts)
    d.index.name = "timestamp"
    d["price"] = pd.to_numeric(df[last_col], errors="coerce")
    d["buy_vol"]  = pd.to_numeric(df[ask_col], errors="coerce") if ask_col else 0.0
    d["sell_vol"] = pd.to_numeric(df[bid_col], errors="coerce") if bid_col else 0.0
    d["n_trades"] = pd.to_numeric(df[ntr_col], errors="coerce") if ntr_col else np.nan
    if d["n_trades"].isna().all():
        d["n_trades"] = np.where((d["buy_vol"].fillna(0)+d["sell_vol"].fillna(0))>0, 1.0, 0.0)

    # Sierra has no per-trade sign; build proxies
    d = d.fillna({"buy_vol":0.0,"sell_vol":0.0})
    d["flow_sum"] = d["buy_vol"] - d["sell_vol"]
    d["buy_count"]  = (d["buy_vol"] > 0).astype(float)
    d["sell_count"] = (d["sell_vol"] > 0).astype(float)
    # sign_rate proxy on the 1s grid (computed after resample)
    r = params.resample
    g = pd.DataFrame({
        "price": d["price"].resample(r).last().ffill(),
        "n_trades": d["n_trades"].resample(r).sum(),
        "buy_count": d["buy_count"].resample(r).sum(),
        "sell_count": d["sell_count"].resample(r).sum(),
        "buy_vol": d["buy_vol"].resample(r).sum(),
        "sell_vol": d["sell_vol"].resample(r).sum(),
        "flow_sum": d["flow_sum"].resample(r).sum(),
    })
    g["sign_rate"] = np.sign(g["flow_sum"].replace(0, np.nan)).fillna(0.0)
    g.index.name = "timestamp"
    return g

# -------------------- Core features (grid-based) --------------------
def arrival_rates(grid: pd.DataFrame, params: Params) -> pd.DataFrame:
    Δs = max(pd.to_timedelta(params.delta).total_seconds(), 1e-9)
    lam_buy  = grid["buy_count"].rolling(params.delta, min_periods=1).sum() / Δs
    lam_sell = grid["sell_count"].rolling(params.delta, min_periods=1).sum() / Δs
    return pd.DataFrame({"lam_buy": lam_buy, "lam_sell": lam_sell})

def trade_imbalance(grid: pd.DataFrame, params: Params) -> pd.Series:
    volp = grid["buy_vol"].rolling(params.delta, min_periods=1).sum()
    volm = grid["sell_vol"].rolling(params.delta, min_periods=1).sum()
    return ((volp - volm) / (volp + volm).replace(0, np.nan)).rename("ti")

def q_cumulative(grid: pd.DataFrame, params: Params) -> pd.Series:
    q = grid["flow_sum"].cumsum()
    if params.normalize_qcum:
        norm = grid["flow_sum"].abs().rolling(params.base, min_periods=10).sum().replace(0, np.nan)
        q = q / norm
    return q.rename("q_cum")

def burstiness_counts(grid: pd.DataFrame, params: Params) -> pd.DataFrame:
    counts = grid["n_trades"]
    mean_c = counts.rolling(params.delta, min_periods=1).mean()
    var_c  = counts.rolling(params.delta, min_periods=1).var(ddof=0)
    fano   = (var_c / mean_c.replace(0,np.nan)).rename("fano")
    cv_iat = (counts.rolling(params.delta, min_periods=2).std(ddof=0) /
              counts.rolling(params.delta, min_periods=2).mean().replace(0,np.nan)).rename("cv_iat")
    return pd.DataFrame({"fano": fano, "cv_iat": cv_iat})

def sign_autocorr(grid: pd.DataFrame, params: Params) -> pd.Series:
    sr = grid["sign_rate"].fillna(0.0)
    small = "10s" if pd.to_timedelta(params.delta) >= pd.to_timedelta("10s") else params.delta
    ac1 = sr.rolling(small, min_periods=3).apply(lambda x: x.autocorr(lag=params.lag_autocorr) or 0.0, raw=False)
    return ac1.rolling(params.delta, min_periods=1).mean().rename("sign_ac1")

def realized_vol(grid: pd.DataFrame, params: Params) -> pd.Series:
    ret1 = grid["price"].ffill().diff()
    return ret1.rolling(params.delta, min_periods=2).std(ddof=0).rename("rv")

def impact_elasticity(df_or_grid: pd.DataFrame, params: Params) -> pd.Series:
    """
    Per-trade path: uses per-trade signed size.
    Sierra path   : uses per-second signed volume (buy_vol - sell_vol).
    """
    if _has_per_trade_side(df_or_grid, params):
        d = _ensure_sorted(df_or_grid)
        if "sign" not in d.columns:
            d = _signed(d, params)
        dp = d[params.price_col].astype(float).diff()
        signed_size = (d["sign"] * d[params.size_col].astype(float)).replace(0, np.nan)
        impact = (dp / signed_size).replace([np.inf, -np.inf], np.nan)
        imp_g = impact.resample(params.resample).mean()
        out = imp_g.rolling(params.delta, min_periods=3).median().rename("impact_e")
        out.index.name = "timestamp"
        return out
    else:
        # Sierra grid expected
        g = df_or_grid if "flow_sum" in df_or_grid.columns else _to_grid_sierra(df_or_grid, params)
        ret1 = g["price"].ffill().diff()
        signed_vol = g["flow_sum"].replace(0, np.nan)
        return (ret1 / signed_vol).replace([np.inf,-np.inf], np.nan).rolling(params.delta, min_periods=3)\
               .median().rename("impact_e")

def flow_move(grid: pd.DataFrame, params: Params) -> pd.Series:
    ret1 = grid["price"].ffill().diff().fillna(0.0)
    return (np.sign(grid["flow_sum"].fillna(0.0)) * ret1).rolling(params.delta, min_periods=1).sum().rename("flow_move")

def cluster_intensity(df_or_grid: pd.DataFrame, params: Params) -> pd.Series:
    """
    Per-trade: large-trade arrivals per second.
    Sierra   : large *per-second* total volume vs rolling 95th percentile.
    """
    if _has_per_trade_side(df_or_grid, params):
        d = _ensure_sorted(df_or_grid)
        if params.large_trade_thresh and params.large_trade_thresh > 0:
            mask = d[params.size_col] >= params.large_trade_thresh
        else:
            z = d.copy()
            z["day"] = z.index.tz_convert(None).date
            thresh = z.groupby("day")[params.size_col].transform(lambda s: s.quantile(0.95))
            mask = z[params.size_col] >= thresh
        large_per_s = mask.resample(params.resample).sum().astype(float)
        Δs = max(pd.to_timedelta(params.delta).total_seconds(), 1e-9)
        out = (large_per_s.rolling(params.delta, min_periods=1).sum() / Δs).rename("cluster_intensity")
        out.index.name = "timestamp"
        return out
    else:
        g = df_or_grid if "buy_vol" in df_or_grid.columns else _to_grid_sierra(df_or_grid, params)
        total_v = (g["buy_vol"] + g["sell_vol"]).fillna(0.0)
        thresh = total_v.rolling("1D", min_periods=10).quantile(0.95)
        large = (total_v >= thresh).astype(float)
        Δs = max(pd.to_timedelta(params.delta).total_seconds(), 1e-9)
        return (large.rolling(params.delta, min_periods=1).sum() / Δs).rename("cluster_intensity")

def phi_interaction(features: pd.DataFrame) -> pd.Series:
    if not {"lam_buy","lam_sell","rv"}.issubset(features.columns):
        return pd.Series(index=features.index, dtype=float, name="phi")
    return ((features["lam_buy"] - features["lam_sell"]) * features["rv"]).rename("phi")

# -------------------- Public API --------------------
def build_features(trades: pd.DataFrame, params: Optional[Params] = None) -> pd.DataFrame:
    """
    Accepts either:
      (A) Per-trade prints with ['timestamp','price','size','side']  (tz-aware or naive with assume_tz)
      (B) Sierra aggregated rows with ['DateTime','Last/Close','AskTradeVolume','BidTradeVolume', ...]
    Returns a 1-second indexed DataFrame of features.
    """
    params = params or Params()

    # Ensure time index is tz-aware UTC for both paths
    use_sierra = True
    if "timestamp" in trades.columns:
        # If already on a timestamp column, try to detect if side exists (per-trade path)
        use_sierra = not _has_per_trade_side(trades, params)

    if not use_sierra and params.side_col in trades.columns:
        # -------- Per-trade path --------
        if "timestamp" in trades.columns:
            t = trades.copy()
            t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
            if getattr(t["timestamp"].dt, "tz", None) is None:
                t["timestamp"] = t["timestamp"].dt.tz_localize(params.assume_tz, nonexistent="shift_forward",
                                                               ambiguous="NaT").dt.tz_convert("UTC")
            else:
                t["timestamp"] = t["timestamp"].dt.tz_convert("UTC")
            t = t.sort_values("timestamp").set_index("timestamp")
        else:
            t = _ensure_sorted(trades, "timestamp")
        t = _signed(t, params)
        g = _to_grid_per_trade(t, params)

        arr = arrival_rates(g, params)
        ti  = trade_imbalance(g, params)
        q   = q_cumulative(g, params)
        bur = burstiness_counts(g, params)
        ac1 = sign_autocorr(g, params)
        rv  = realized_vol(g, params)
        fm  = flow_move(g, params)
        imp = impact_elasticity(t, params)
        ci  = cluster_intensity(t, params)
    else:
        # -------- Sierra path --------
        g = _to_grid_sierra(trades, params)

        arr = arrival_rates(g, params)
        ti  = trade_imbalance(g, params)
        q   = q_cumulative(g, params)
        bur = burstiness_counts(g, params)
        ac1 = sign_autocorr(g, params)
        rv  = realized_vol(g, params)
        fm  = flow_move(g, params)
        # impact/cluster computed from per-second aggregates
        imp = impact_elasticity(g, params)
        ci  = cluster_intensity(g, params)

    feats = g[[]]
    for comp in (arr, ti, q, bur, ac1, rv, imp, fm, ci):
        feats = feats.join(comp, how="outer") if isinstance(comp, pd.Series) else feats.join(comp, how="outer")
    feats["phi"] = phi_interaction(feats)
    return feats.sort_index().ffill(limit=5)
