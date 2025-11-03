# orderflow_research.py
# Purpose: Python research pipeline for session profiles, informed-flow detection, and a single predictor.
# Fits Sierra-style CSV/DF (DateTime, Last/Close, AskTradeVolume, BidTradeVolume, NumberOfTrades).
# Research prototype only; not trading advice.

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Union
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# ============================== Config ==============================

@dataclass
class SessionCfg:
    rth_start: str = "08:30:00"   # CME metals in CT; set to 09:30:00 for equities
    rth_end: str   = "15:00:00"   # adjust per product
    tz: str        = "America/Chicago"
    ib_minutes: int = 60
    tick: float     = 0.1         # set per symbol (e.g., GC 0.1, SI 0.005)
    profile_tick: float = 0.1     # bin for volume profile

@dataclass
class FlowCfg:
    band_ticks: int = 2            # price proximity to level
    arr_window: str = "5s"         # Δ window for arrival rate
    base_window: str = "120s"      # W baseline (for z-scores)
    z_thresh: float = 2.0          # trigger threshold
    resample: str = "1s"           # grid for flow/px

@dataclass
class LabelCfg:
    tgt_ticks: int = 8
    stp_ticks: int = 6
    horizon: str   = "180s"

@dataclass
class FeatureCfg:
    # Default feature flags if features=None in run_pipeline()
    f_zmax: bool = True
    f_imb: bool = True
    f_dist: bool = True
    f_is_rth: bool = True
    f_lvl_bins: bool = True
    f_dir: bool = True
    # Optional adds (come from flow table)
    f_lam: bool = False
    f_z_buy_sell: bool = False

# ============================== Single class ==============================

class OrderflowPipeline:
    """
    Ingest Sierra-style aggregated data → sessionize → compute session levels →
    build 1s flow/z-scores from Ask/Bid trade volumes → trigger near levels →
    label hypothetical trades → fit ridge predictor with selectable features.
    """

    # ---------- I/O & normalization ----------
    @staticmethod
    def _norm(s: str) -> str:
        return s.lower().replace(" ", "").replace("_","")

    def _detect_cols(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        cols = {self._norm(c): c for c in df.columns}
        def pick(*cands):
            for k in cands:
                kk = self._norm(k)
                if kk in cols:
                    return cols[kk]
            return None
        return dict(
            ts  = pick("DateTime","datetime","timestamp","SCDateTime","StartDateTime","Datetime","ts","time"),
            px  = pick("Last","Close","Price","LastPrice","Settlement","Px","PX","px"),
            buy = pick("AskVolume","Ask Trade Volume","AskTradeVol","UpVolume","UpTickVolume","BuyVolume","buy"),
            sell= pick("BidVolume","Bid Trade Volume","BidTradeVol","DownVolume","DownTickVolume","SellVolume","sell"),
            ntr = pick("NumberOfTrades","Trades","NumTrades", "TradeCount","#Trades","ntr")
        )

    def load(self, data: Union[str, pd.DataFrame], tz: str) -> pd.DataFrame:
        """
        Accept a CSV path or a preloaded DataFrame.
        Normalize to columns: timestamp(tz-aware), price, buy_vol, sell_vol, n_trades.
        """
        if isinstance(data, str):
            raw = pd.read_csv(data)
        else:
            raw = data.copy()

        cols = self._detect_cols(raw)
        if cols["ts"] is None or cols["px"] is None:
            raise ValueError(f"Could not detect timestamp/price columns; got {list(raw.columns)}")

        ts = pd.to_datetime(raw[cols["ts"]], errors="coerce", utc=True).dt.tz_convert(tz)
        df = pd.DataFrame({
            "timestamp": ts,
            "price": pd.to_numeric(raw[cols["px"]], errors="coerce"),
            "buy_vol": pd.to_numeric(raw[cols["buy"]], errors="coerce") if cols["buy"] else 0.0,
            "sell_vol": pd.to_numeric(raw[cols["sell"]], errors="coerce") if cols["sell"] else 0.0,
            "n_trades": pd.to_numeric(raw[cols["ntr"]], errors="coerce") if cols["ntr"] else np.nan,
        }).dropna(subset=["timestamp"])

        # If n_trades missing, use 1 when any volume that row; else 0.
        if df["n_trades"].isna().all():
            df["n_trades"] = ((df["buy_vol"].fillna(0) + df["sell_vol"].fillna(0)) > 0).astype(float)

        # Synthetic 'size' for session VWAP/profile: total traded volume this row
        df["size"] = (df["buy_vol"].fillna(0) + df["sell_vol"].fillna(0)).astype(float)

        return df.sort_values("timestamp").reset_index(drop=True)

    # ---------- Sessionization & prior summary ----------
    def mark_sessions(self, t: pd.DataFrame, cfg: SessionCfg) -> pd.DataFrame:
        ts = t["timestamp"]
        t = t.copy()
        t["date"] = ts.dt.date
        is_rth = (ts.dt.time >= pd.to_datetime(cfg.rth_start).time()) & \
                 (ts.dt.time <  pd.to_datetime(cfg.rth_end).time())
        t["is_rth"] = is_rth.astype(int)
        # increments at RTH/ON boundary
        prior = t["is_rth"].shift().fillna(t["is_rth"].iloc[0])
        rth_block = t["is_rth"].ne(prior).cumsum()
        t["session_id"] = (t["date"].astype(str) + "_" + rth_block.astype(str)).astype("category").cat.codes
        return t

    def prior_day_summary(self, trades: pd.DataFrame, day: pd.Timestamp) -> Dict[str, float]:
        prev = trades[trades["timestamp"].dt.normalize() == (pd.Timestamp(day).tz_localize(None) - pd.Timedelta(days=1))]
        if prev.empty: return {}
        return {"High": prev["price"].max(),
                "Low": prev["price"].min(),
                "Close": prev["price"].iloc[-1]}

    # ---------- Session levels ----------
    def compute_session_levels(self, session_df: pd.DataFrame, cfg: SessionCfg,
                               prior_summary: Dict[str, float]) -> Dict[str, float]:
        d = {}
        first_ts = session_df["timestamp"].iloc[0]
        ib_cut = first_ts + pd.Timedelta(minutes=cfg.ib_minutes)
        ib = session_df[session_df["timestamp"] <= ib_cut]
        d["IBH"] = ib["price"].max()
        d["IBL"] = ib["price"].min()

        # Session VWAP using aggregated trade volume
        v = session_df["size"].astype(float)
        p = session_df["price"].astype(float)
        d["VWAP"] = (p * v).sum() / max(v.sum(), 1e-12)

        d["PriorH"] = prior_summary.get("High", np.nan)
        d["PriorL"] = prior_summary.get("Low", np.nan)
        d["PriorC"] = prior_summary.get("Close", np.nan)

        # Naive session volume profile (weights = size)
        price_min, price_max = session_df["price"].min(), session_df["price"].max()
        if not np.isfinite(price_min) or not np.isfinite(price_max):
            d["POC"]=d["VAL"]=d["VAH"]=np.nan
            return d
        bins = np.arange(np.floor(price_min / cfg.profile_tick) * cfg.profile_tick,
                         np.ceil(price_max / cfg.profile_tick) * cfg.profile_tick + cfg.profile_tick/2,
                         cfg.profile_tick)
        vp, edges = np.histogram(session_df["price"], bins=bins, weights=session_df["size"])
        centers = (edges[:-1] + edges[1:]) / 2.0
        if vp.sum() > 0:
            poc_idx = int(np.argmax(vp)); d["POC"] = centers[poc_idx]
            target = 0.7 * vp.sum(); lo = hi = poc_idx; cum = vp[poc_idx]
            while cum < target and (lo > 0 or hi < len(vp) - 1):
                left = vp[lo - 1] if lo > 0 else -1
                right = vp[hi + 1] if hi < len(vp) - 1 else -1
                if right > left: hi += 1; cum += vp[hi]
                else: lo -= 1; cum += vp[lo]
            d["VAL"], d["VAH"] = centers[lo], centers[hi]
        else:
            d["POC"]=d["VAL"]=d["VAH"]=np.nan
        return d

    # ---------- Flow from aggregated Ask/Bid trade volumes ----------
    def arrival_rates(self, trades: pd.DataFrame, flow_cfg: FlowCfg) -> pd.DataFrame:
        """
        Build 1s grid of 'counts' & z-scores from Sierra aggregated volumes:
        - Count a 'buy event' for a second if AskTradeVolume>0; likewise sell.
        - lam_buy/lam_sell = rolling event-rate (per sec) over Δ
        - z_buy/z_sell = Poisson z-score of event-rate vs baseline window W
        - imb = rate difference normalized by sum
        """
        df = trades.set_index("timestamp")
        # collapse to 1s
        buy_ev = (df["buy_vol"].fillna(0) > 0).astype(int).resample(flow_cfg.resample).sum().rename("n_buy")
        sell_ev= (df["sell_vol"].fillna(0) > 0).astype(int).resample(flow_cfg.resample).sum().rename("n_sell")

        Δ = pd.to_timedelta(flow_cfg.arr_window).total_seconds()
        W = pd.to_timedelta(flow_cfg.base_window).total_seconds()

        rate_buy  = buy_ev.rolling(flow_cfg.arr_window, min_periods=1).sum() / max(Δ,1e-9)
        rate_sell = sell_ev.rolling(flow_cfg.arr_window, min_periods=1).sum() / max(Δ,1e-9)
        mu_buy  = buy_ev.rolling(flow_cfg.base_window, min_periods=1).sum() / max(W,1e-9)
        mu_sell = sell_ev.rolling(flow_cfg.base_window, min_periods=1).sum() / max(W,1e-9)

        sig_buy  = np.sqrt(mu_buy.clip(lower=1e-6))
        sig_sell = np.sqrt(mu_sell.clip(lower=1e-6))
        z_buy  = (rate_buy  - mu_buy ) / sig_buy
        z_sell = (rate_sell - mu_sell) / sig_sell
        imb = (rate_buy - rate_sell) / (rate_buy + rate_sell).replace(0, np.nan)

        out = pd.concat([rate_buy.rename("lam_buy"),
                         rate_sell.rename("lam_sell"),
                         z_buy.rename("z_buy"),
                         z_sell.rename("z_sell"),
                         imb.rename("imb")], axis=1).ffill()
        return out.reset_index().rename(columns={"index":"timestamp"})

    # ---------- Level handling ----------
    @staticmethod
    def build_level_table(levels: Dict[str, float]) -> pd.DataFrame:
        rows = []
        TYPE = {"IBH":1,"IBL":2,"VWAP":3,"PriorH":4,"PriorL":5,"PriorC":6,"POC":7,"VAH":8,"VAL":9}
        for k,v in levels.items():
            if np.isfinite(v): rows.append({"level_name": k, "level_px": float(v), "level_type": TYPE.get(k, 0)})
        return pd.DataFrame(rows)

    @staticmethod
    def nearest_level(price: float, level_df: pd.DataFrame, band: float) -> Tuple[int, float, int, str]:
        if level_df.empty: return (-1, np.nan, 0, "")
        diffs = np.abs(level_df["level_px"] - price)
        j = int(diffs.idxmin())
        if diffs.iloc[j] <= band:
            r = level_df.iloc[j]
            return (j, float(r["level_px"]), int(r["level_type"]), str(r["level_name"]))
        return (-1, np.nan, 0, "")

    def generate_signals(self, session_df: pd.DataFrame, flow: pd.DataFrame,
                         level_df: pd.DataFrame, sess_cfg: SessionCfg, flow_cfg: FlowCfg) -> pd.DataFrame:
        # 1s price & RTH on same grid as flow
        px1s = session_df.set_index("timestamp")["price"].resample(flow_cfg.resample).last().ffill().rename("price")
        on_rth = session_df.set_index("timestamp")["is_rth"].resample(flow_cfg.resample).last().ffill().rename("is_rth")
        grid = pd.concat([px1s, on_rth], axis=1).reset_index()
        merged = pd.merge_asof(grid.sort_values("timestamp"),
                               flow.sort_values("timestamp"),
                               on="timestamp", direction="backward")

        band = flow_cfg.band_ticks * sess_cfg.tick
        rows = []
        for r in merged.itertuples(index=False):
            idx, lvl_px, lvl_type, lvl_name = self.nearest_level(r.price, level_df, band)
            if idx < 0: continue
            zmax = np.nanmax([getattr(r, "z_buy"), getattr(r, "z_sell")])
            if np.isfinite(zmax) and zmax >= flow_cfg.z_thresh:
                direction = 1 if getattr(r, "z_buy") >= getattr(r, "z_sell") else -1
                rows.append({
                    "timestamp": r.timestamp,
                    "price": r.price,
                    "level_px": lvl_px,
                    "level_type": lvl_type,
                    "level_name": lvl_name,
                    "dist_ticks": (r.price - lvl_px)/sess_cfg.tick,
                    "lam_buy": getattr(r, "lam_buy"),
                    "lam_sell": getattr(r, "lam_sell"),
                    "z_buy": getattr(r, "z_buy"),
                    "z_sell": getattr(r, "z_sell"),
                    "imb": getattr(r, "imb"),
                    "is_rth": int(r.is_rth),
                    "direction": direction
                })
        return pd.DataFrame(rows)

    # ---------- Labeling ----------
    def label_entries(self, entries: pd.DataFrame, trades: pd.DataFrame,
                      sess_cfg: SessionCfg, lab_cfg: LabelCfg) -> pd.DataFrame:
        if entries.empty: return entries.assign(y=np.nan)
        t = trades.set_index("timestamp")[["price"]].sort_index()
        tgt_px = lab_cfg.tgt_ticks * sess_cfg.tick
        stp_px = lab_cfg.stp_ticks * sess_cfg.tick

        ys, exits = [], []
        for r in entries.itertuples(index=False):
            t0 = r.timestamp
            t1 = t0 + pd.Timedelta(lab_cfg.horizon)
            path = t.loc[t0:t1, "price"]
            if path.empty:
                ys.append(np.nan); exits.append(pd.NaT); continue
            entry_px = r.price
            if r.direction > 0:
                hit_tgt = path[path >= entry_px + tgt_px]
                hit_stp = path[path <= entry_px - stp_px]
            else:
                hit_tgt = path[path <= entry_px - tgt_px]
                hit_stp = path[path >= entry_px + stp_px]
            t_hit = pd.NaT; ret_ticks = np.nan
            if not hit_tgt.empty and not hit_stp.empty:
                if hit_tgt.index[0] <= hit_stp.index[0]:
                    ret_ticks = lab_cfg.tgt_ticks; t_hit = hit_tgt.index[0]
                else:
                    ret_ticks = -lab_cfg.stp_ticks; t_hit = hit_stp.index[0]
            elif not hit_tgt.empty:
                ret_ticks = lab_cfg.tgt_ticks; t_hit = hit_tgt.index[0]
            elif not hit_stp.empty:
                ret_ticks = -lab_cfg.stp_ticks; t_hit = hit_stp.index[0]
            else:
                exit_px = path.iloc[-1]
                ret_ticks = (exit_px - entry_px)/sess_cfg.tick if r.direction>0 else (entry_px - exit_px)/sess_cfg.tick
                t_hit = path.index[-1]
            ys.append(float(ret_ticks)); exits.append(t_hit)
        return entries.assign(y=ys, exit_ts=exits)

    # ---------- Feature map (selectable) ----------
    def feature_map(self, df: pd.DataFrame, features: Optional[Iterable[str]], feat_cfg: FeatureCfg) -> pd.DataFrame:
        # base primitives always available in entries
        cols = {}
        if features is None:
            # Use flags
            if feat_cfg.f_zmax: cols["zmax"] = df[["z_buy","z_sell"]].max(axis=1)
            if feat_cfg.f_imb:  cols["imb"]  = df["imb"].fillna(0.0)
            if feat_cfg.f_dist: cols["dist"] = df["dist_ticks"]
            if feat_cfg.f_is_rth: cols["is_rth"] = df["is_rth"].astype(int)
            if feat_cfg.f_lvl_bins:
                cols["lvl_is_vwap"]   = (df["level_type"]==3).astype(int)
                cols["lvl_is_ib"]     = ((df["level_type"]==1)|(df["level_type"]==2)).astype(int)
                cols["lvl_is_prior"]  = ((df["level_type"]>=4)&(df["level_type"]<=6)).astype(int)
                cols["lvl_is_profile"]= ((df["level_type"]>=7)).astype(int)
            if feat_cfg.f_dir: cols["dir"] = df["direction"].astype(int)
            if feat_cfg.f_lam:
                cols["lam_buy"] = df.get("lam_buy", pd.Series(index=df.index, dtype=float))
                cols["lam_sell"]= df.get("lam_sell", pd.Series(index=df.index, dtype=float))
            if feat_cfg.f_z_buy_sell:
                cols["z_buy"]   = df.get("z_buy", pd.Series(index=df.index, dtype=float))
                cols["z_sell"]  = df.get("z_sell", pd.Series(index=df.index, dtype=float))
        else:
            # explicit list
            for k in features:
                k = k.lower()
                if k == "zmax": cols["zmax"] = df[["z_buy","z_sell"]].max(axis=1)
                elif k in {"imb","dist","is_rth","dir","lam_buy","lam_sell","z_buy","z_sell"}:
                    cols[k] = df[k] if k in df.columns else pd.Series(index=df.index, dtype=float)
                elif k == "lvl_is_vwap":
                    cols["lvl_is_vwap"] = (df["level_type"]==3).astype(int)
                elif k == "lvl_is_ib":
                    cols["lvl_is_ib"] = ((df["level_type"]==1)|(df["level_type"]==2)).astype(int)
                elif k == "lvl_is_prior":
                    cols["lvl_is_prior"] = ((df["level_type"]>=4)&(df["level_type"]<=6)).astype(int)
                elif k == "lvl_is_profile":
                    cols["lvl_is_profile"] = ((df["level_type"]>=7)).astype(int)
                else:
                    # ignore unknown silently to keep research flow fast
                    pass

        X = pd.DataFrame(cols).copy()
        if "intercept" not in X.columns:
            X.insert(0, "intercept", 1.0)
        return X

    # ---------- Fit predictor ----------
    @staticmethod
    def fit_predictor(entries_labeled: pd.DataFrame, X: pd.DataFrame) -> Tuple[Ridge, float]:
        clean = entries_labeled.dropna(subset=["y"]).copy()
        if clean.empty:
            raise ValueError("No labeled trades.")
        y = clean["y"].values
        model = Ridge(alpha=1.0, fit_intercept=False)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X.loc[clean.index].values, y, cv=tscv, scoring="neg_mean_squared_error")
        model.fit(X.loc[clean.index].values, y)
        rmse = float(np.sqrt(-scores.mean()))
        return model, rmse

    # ---------- Orchestration ----------
    def run_pipeline(self,
                     data: Union[str, pd.DataFrame],
                     sess_cfg: SessionCfg,
                     flow_cfg: FlowCfg,
                     lab_cfg: LabelCfg,
                     feat_cfg: FeatureCfg = FeatureCfg(),
                     features: Optional[Iterable[str]] = None
                     ) -> Dict[str, pd.DataFrame]:
        """
        End-to-end run. 'features' is an optional list of feature names for the predictor; if None, uses feat_cfg flags.
        Returns: dict with 'entries', 'labeled', 'X', 'coef'
        """
        t = self.load(data, tz=sess_cfg.tz)
        t = self.mark_sessions(t, sess_cfg)

        # Build flow table on whole tape
        flow = self.arrival_rates(t[["timestamp","buy_vol","sell_vol"]], flow_cfg)

        # Per-session processing
        collected = []
        for sid, sess_df in t.groupby("session_id"):
            if len(sess_df) < 100:  # skip tiny sessions
                continue
            day = pd.Timestamp(sess_df["timestamp"].iloc[0].date())
            prior = self.prior_day_summary(t, day)
            levels = self.compute_session_levels(sess_df, sess_cfg, prior)
            lvl_tbl = self.build_level_table(levels)
            if lvl_tbl.empty:
                continue
            entries = self.generate_signals(sess_df, flow, lvl_tbl, sess_cfg, flow_cfg)
            if entries.empty:
                continue
            labeled = self.label_entries(entries, sess_df[["timestamp","price"]], sess_cfg, lab_cfg)
            collected.append(labeled)

        if not collected:
            return {"entries": pd.DataFrame(), "labeled": pd.DataFrame(), "X": pd.DataFrame(), "coef": pd.DataFrame()}

        entries_all = pd.concat(collected, ignore_index=True)
        # Build feature matrix with selection
        X = self.feature_map(entries_all, features, feat_cfg)
        # Fit predictor
        model, rmse = self.fit_predictor(entries_all, X)
        coef = pd.DataFrame({"feature": X.columns, "coef": model.coef_}).assign(metric=f"RMSE={rmse:.3f}" )

        return {"entries": entries_all.drop(columns=["exit_ts"], errors="ignore"),
                "labeled": entries_all,
                "X": X,
                "coef": coef}
