import numpy as np
import pandas as pd
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Iterable, Dict, Optional, Tuple, List

# ---------- Config ----------
@dataclass
class SessCfg:
    rth_start: str = "08:30:00"
    rth_end: str   = "15:00:00"
    tz: str        = "America/Chicago"
    ib_minutes: int = 60
    tick: float     = 0.1
    profile_tick: float = 0.1

@dataclass
class FlowCfg:
    delta_s: int = 5     # Δ in seconds
    base_s: int  = 120   # W in seconds
    z_thresh: float = 2.0
    band_ticks: int = 2

@dataclass
class LabCfg:
    tgt_ticks: int = 8
    stp_ticks: int = 6
    horizon_s: int = 180

# ---------- Utils: online rolling sums via ring buffers ----------
class RollingSums:
    """O(1) update rolling sums for up to two windows (Δ and W) without pandas. Zero-copy-ish."""
    def __init__(self, win_delta: int, win_base: int):
        self.wd = win_delta
        self.wb = win_base
        self.buf_buy_d = deque(maxlen=win_delta)
        self.buf_sell_d = deque(maxlen=win_delta)
        self.buf_buy_b = deque(maxlen=win_base)
        self.buf_sell_b = deque(maxlen=win_base)
        self.sum_buy_d = 0.0
        self.sum_sell_d = 0.0
        self.sum_buy_b = 0.0
        self.sum_sell_b = 0.0

    def push(self, n_buy: float, n_sell: float):
        # Δ window
        self.buf_buy_d.append(n_buy); self.sum_buy_d += n_buy
        self.buf_sell_d.append(n_sell); self.sum_sell_d += n_sell
        if len(self.buf_buy_d) > self.wd: self.sum_buy_d -= self.buf_buy_d[0]
        if len(self.buf_sell_d) > self.wd: self.sum_sell_d -= self.buf_sell_d[0]
        # W window
        self.buf_buy_b.append(n_buy); self.sum_buy_b += n_buy
        self.buf_sell_b.append(n_sell); self.sum_sell_b += n_sell
        if len(self.buf_buy_b) > self.wb: self.sum_buy_b -= self.buf_buy_b[0]
        if len(self.buf_sell_b) > self.wb: self.sum_sell_b -= self.buf_sell_b[0]

    def lambda_buy(self):  return self.sum_buy_d / max(len(self.buf_buy_d), 1)
    def lambda_sell(self): return self.sum_sell_d / max(len(self.buf_sell_d), 1)
    def mu_buy(self):      return self.sum_buy_b / max(len(self.buf_buy_b), 1)
    def mu_sell(self):     return self.sum_sell_b / max(len(self.buf_sell_b), 1)

# ---------- Session state (streaming) ----------
class SessionState:
    """Holds per-session arrays / aggregates at 1-second resolution without keeping ticks."""
    __slots__ = ("epoch0","px_last","buy_sec","sell_sec","ntr_sec","sec_index",
                 "ib_cut_epoch","vwap_num","vwap_den","profile","first_epoch",
                 "cfg","ring")

    def __init__(self, first_epoch: int, cfg: SessCfg, flow: FlowCfg):
        self.epoch0 = first_epoch
        self.first_epoch = first_epoch
        cap = 24*60*60 + 10  # max seconds per day
        # Preallocate dynamic lists; we’ll append (avoids reindexing DataFrames)
        self.px_last: List[float] = []
        self.buy_sec: List[float] = []
        self.sell_sec: List[float] = []
        self.ntr_sec: List[float] = []
        self.sec_index: List[int] = []
        # IB / VWAP / Profile
        self.cfg = cfg
        self.ib_cut_epoch = first_epoch + cfg.ib_minutes*60
        self.vwap_num = 0.0
        self.vwap_den = 0.0
        self.profile = defaultdict(float)  # price-bin -> volume
        # rolling flow rings
        self.ring = RollingSums(flow.delta_s, flow.base_s)

    def _bin_price(self, px: float) -> float:
        t = self.cfg.profile_tick
        return np.round(px / t) * t

    def push_second(self, epoch_s: int, px_last: float, buy: float, sell: float, ntr: float):
        """Append pre-aggregated per-second data"""
        self.sec_index.append(epoch_s)
        self.px_last.append(px_last)
        self.buy_sec.append(buy)
        self.sell_sec.append(sell)
        self.ntr_sec.append(ntr)

        # IB/VWAP/Profile (streaming)
        if epoch_s <= self.ib_cut_epoch:
            # IBH/IBL will be derived from px_last array slice later (no extra memory)
            pass
        vol = buy + sell
        self.vwap_num += px_last * vol
        self.vwap_den += vol
        self.profile[self._bin_price(px_last)] += vol

        # update ring with "events" (presence per second)
        n_buy_ev  = 1.0 if buy  > 0 else 0.0
        n_sell_ev = 1.0 if sell > 0 else 0.0
        self.ring.push(n_buy_ev, n_sell_ev)

    def finalize_levels(self) -> Dict[str, float]:
        px = np.asarray(self.px_last, dtype=np.float32)
        sec = np.asarray(self.sec_index, dtype=np.int64)

        # IB slice
        ib_mask = sec <= self.ib_cut_epoch
        IBH = np.nanmax(px[ib_mask]) if ib_mask.any() else np.nan
        IBL = np.nanmin(px[ib_mask]) if ib_mask.any() else np.nan

        VWAP = self.vwap_num / self.vwap_den if self.vwap_den > 0 else np.nan

        # 70% value area around POC from streaming histogram
        if self.profile:
            keys = np.array(list(self.profile.keys()))
            vals = np.array([self.profile[k] for k in keys])
            if vals.sum() > 0:
                poc_idx = int(np.argmax(vals))
                POC = float(keys[poc_idx])
                target = 0.7 * vals.sum()
                lo = hi = poc_idx; cum = vals[poc_idx]
                # expand heuristically
                while cum < target and (lo > 0 or hi < len(vals)-1):
                    left  = vals[lo-1] if lo > 0 else -1
                    right = vals[hi+1] if hi < len(vals)-1 else -1
                    if right > left: hi += 1; cum += vals[hi]
                    else: lo -= 1; cum += vals[lo]
                VAL, VAH = float(keys[lo]), float(keys[hi])
            else:
                POC = VAL = VAH = np.nan
        else:
            POC = VAL = VAH = np.nan

        return {"IBH": IBH, "IBL": IBL, "VWAP": VWAP, "POC": POC, "VAL": VAL, "VAH": VAH}

def rolling_mean_same(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Same-length rolling mean with simple edge padding.
    - O(n) using cumulative sums
    - No pandas.rolling (saves memory)
    - Returns float64 (cast as needed)
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.size
    if n == 0:
        return arr.copy()
    if window <= 1:
        return arr.copy()
    if window > n:
        # Not enough points: just return a flat mean
        return np.full(n, arr.mean())

    # core rolling mean (valid region)
    c = np.cumsum(np.insert(arr, 0, 0.0))
    core = (c[window:] - c[:-window]) / window              # length n - window + 1

    # pad to "same" length by extending edge values
    left_pad = window // 2
    right_pad = n - left_pad - core.size
    left_vals = np.full(left_pad, core[0])
    right_vals = np.full(right_pad, core[-1])
    out = np.concatenate([left_vals, core, right_vals])
    return out


# ---------- Streaming 1s aggregator over tick-like rows ----------
def seconds_floor_ns(ns: np.ndarray) -> np.ndarray:
    # ns = int64 nanoseconds; return floor to seconds in int64 seconds
    return ns // 1_000_000_000

def stream_seconds(chunks: Iterable[pd.DataFrame],
                   tz: str,
                   col_ts: str, col_px: str, col_buy: str, col_sell: str, col_ntr: Optional[str]):
    """
    Yield (epoch_second, last_px, sum_buy, sum_sell, sum_ntr) per second in time order.
    Never stores more than one chunk and small second-level accumulators.
    """
    # For zero-copy, we operate on numpy arrays as much as possible
    last_sec = None
    acc_px = np.nan
    acc_buy = 0.0
    acc_sell = 0.0
    acc_ntr = 0.0

    for df in chunks:
        # minimal selection to avoid materializing extra columns
        # If you want epoch ints (fast for per-second bucketing):
        arr_ts = pd.to_datetime(df[col_ts], errors="coerce", utc=True)
        arr_ns = arr_ts.view("int64")  # ns since epoch
        sec = (arr_ns // 1_000_000_000)  # integer seconds
        px      = pd.to_numeric(df[col_px], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        buy     = pd.to_numeric(df[col_buy], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        sell    = pd.to_numeric(df[col_sell], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        ntr     = pd.to_numeric(df[col_ntr], errors="coerce").to_numpy(dtype=np.float32, copy=False) if col_ntr else np.zeros_like(px)

        # aggregate within chunk by epoch second using np.add.at (in-place)
        # we need last price per second → we’ll scan forward and emit when sec changes
        for s, p, b, se, nt in zip(sec, px, buy, sell, ntr):
            if last_sec is None:
                last_sec, acc_px, acc_buy, acc_sell, acc_ntr = s, p, b, se, nt
                continue
            if s == last_sec:
                acc_px = p   # last
                acc_buy += b
                acc_sell += se
                acc_ntr += nt
            else:
                yield (int(last_sec), float(acc_px), float(acc_buy), float(acc_sell), float(acc_ntr))
                last_sec, acc_px, acc_buy, acc_sell, acc_ntr = s, p, b, se, nt

    if last_sec is not None:
        yield (int(last_sec), float(acc_px), float(acc_buy), float(acc_sell), float(acc_ntr))

# ---------- The memory-efficient pipeline (sketch) ----------
class OrderflowPipelineLite:
    """
    Streaming, low-memory pipeline. Processes ticks in chunks, aggregates to 1s on the fly,
    computes flow stats online, and finalizes per-session outputs immediately.
    """

    def __init__(self, sess: SessCfg, flow: FlowCfg, lab: LabCfg):
        self.sess, self.flow, self.lab = sess, flow, lab

    # simple RTH test on epoch seconds (avoid tz-aware arrays in the loop)
    def _is_rth(self, epoch_s: int) -> int:
        # convert to local time once per second for label/feature; cheap relative to ticks
        ts = pd.Timestamp(epoch_s, unit="s", tz="UTC").tz_convert(self.sess.tz)
        t = ts.time()
        return int(self._t_ge(t, self.sess.rth_start) and self._t_lt(t, self.sess.rth_end))

    @staticmethod
    def _t_ge(t, hhmmss: str): return t >= pd.to_datetime(hhmmss).time()
    @staticmethod
    def _t_lt(t, hhmmss: str): return t <  pd.to_datetime(hhmmss).time()

    def run_from_csv(self, path: str,
                     col_ts="ts", col_px="px", col_buy="buy", col_sell="sell", col_ntr="ntr",
                     chunksize=2_000_000) -> Dict[str, pd.DataFrame]:
        reader = pd.read_csv(path, usecols=[col_ts,col_px,col_buy,col_sell,col_ntr], chunksize=chunksize)
        return self._run_from_seconds(stream_seconds(reader, self.sess.tz, col_ts,col_px,col_buy,col_sell,col_ntr))

    def run_from_df(self, df: pd.DataFrame,
                    col_ts="ts", col_px="px", col_buy="buy", col_sell="sell", col_ntr="ntr",
                    row_chunks=2_000_000) -> Dict[str, pd.DataFrame]:
        # yield row chunks without copying large arrays
        for start in range(0, len(df), row_chunks):
            yield_df = df.iloc[start:start+row_chunks][[col_ts,col_px,col_buy,col_sell,col_ntr]]
            if start == 0:
                seconds_gen = stream_seconds([yield_df], self.sess.tz, col_ts,col_px,col_buy,col_sell,col_ntr)
            else:
                seconds_gen = stream_seconds([yield_df], self.sess.tz, col_ts,col_px,col_buy,col_sell,col_ntr)
        # simpler path when df fits once:
        seconds_gen = stream_seconds([df[[col_ts,col_px,col_buy,col_sell,col_ntr]]], self.sess.tz,
                                     col_ts,col_px,col_buy,col_sell,col_ntr)
        return self._run_from_seconds(seconds_gen)

    def _run_from_seconds(self, seconds_iter: Iterable[Tuple[int,float,float,float,float]]) -> Dict[str, pd.DataFrame]:
        sess_state: Optional[SessionState] = None
        entries_all = []   # small; only triggers stored
        labeled_all = []   # small
        tick = self.sess.tick
        band = self.flow.band_ticks * tick

        # per-session level placeholders (prior H/L/C computed from previous day’s last)
        prev_day = None
        prior_summary = {"High": np.nan, "Low": np.nan, "Close": np.nan}

        # store last price for labeling horizon scan: keep only last N seconds ring (not entire session)
        # for labeling we need up to horizon_s forward; we can post-label after session using the per-second px array

        # iterate 1-second aggregates
        for epoch_s, px_last, buy, sell, ntr in seconds_iter:
            # new session?
            local_date = pd.Timestamp(epoch_s, unit="s", tz="UTC").tz_convert(self.sess.tz).date()
            if sess_state is None:
                sess_state = SessionState(epoch_s, self.sess, self.flow)
                prev_day = local_date
            else:
                # day flip? finalize & reset
                if local_date != prev_day:
                    # finalize previous session outputs immediately (no long-term RAM)
                    levels = sess_state.finalize_levels()
                    # build triggers on the fly during session would be ideal; see below
                    # For brevity: we defer building entries until end-of-session using arrays.
                    entries, labeled = self._finalize_triggers_and_labels(sess_state, levels, band)
                    if len(entries):
                        entries_all.append(entries); labeled_all.append(labeled)
                    # update prior close
                    prior_summary["Close"] = sess_state.px_last[-1] if sess_state.px_last else np.nan
                    sess_state = SessionState(epoch_s, self.sess, self.flow)
                    prev_day = local_date

            # append second to session state (O(1))
            sess_state.push_second(epoch_s, px_last, buy, sell, ntr)

        # tail finalize
        if sess_state is not None and sess_state.px_last:
            levels = sess_state.finalize_levels()
            entries, labeled = self._finalize_triggers_and_labels(sess_state, levels, band)
            if len(entries):
                entries_all.append(entries); labeled_all.append(labeled)

        entries_df = pd.concat(entries_all, ignore_index=True) if entries_all else pd.DataFrame()
        labeled_df = pd.concat(labeled_all, ignore_index=True) if labeled_all else pd.DataFrame()
        return {"entries": entries_df, "labeled": labeled_df}

    # ---- build triggers & labels from compact per-second arrays (no big DF merges) ----
    def _finalize_triggers_and_labels(self, s: SessionState, levels: Dict[str,float], band: float) -> Tuple[pd.DataFrame,pd.DataFrame]:
        sec = np.asarray(s.sec_index, dtype=np.int64)
        px  = np.asarray(s.px_last, dtype=np.float32)
        buy = np.asarray(s.buy_sec, dtype=np.float32)
        sell= np.asarray(s.sell_sec, dtype=np.float32)
        # flow events per second
        n_buy = (buy > 0).astype(np.int8)
        n_sell = (sell > 0).astype(np.int8)

        delta_s = int(self.flow.delta_s)
        base_s = int(self.flow.base_s)

        lam_buy = rolling_mean_same(n_buy, delta_s)
        lam_sell = rolling_mean_same(n_sell, delta_s)
        mu_buy = rolling_mean_same(n_buy, base_s)
        mu_sell = rolling_mean_same(n_sell, base_s)

        sig_buy = np.sqrt(np.clip(mu_buy, 1e-6, None))
        sig_sell = np.sqrt(np.clip(mu_sell, 1e-6, None))
        z_buy = (lam_buy - mu_buy) / sig_buy
        z_sell = (lam_sell - mu_sell) / sig_sell

        imb = (lam_buy - lam_sell) / np.where((lam_buy + lam_sell)==0, np.nan, (lam_buy + lam_sell))

        # nearest-level test (vectorized against small level table)
        lvl_tbl = self._levels_to_table(levels)
        if lvl_tbl.size == 0:
            return pd.DataFrame(), pd.DataFrame()
        lvl_px = lvl_tbl[:,0].astype(np.float32)
        lvl_ty = lvl_tbl[:,1].astype(np.int8)

        # broadcast: for each second, find closest level
        # (vectorized via argmin over |px - lvl_px|; small matrix since #levels is tiny)
        diffs = np.abs(px.reshape(-1,1) - lvl_px.reshape(1,-1))
        j = np.nanargmin(diffs, axis=1)
        dmin = diffs[np.arange(len(px)), j]
        hit = dmin <= band
        # trigger condition
        zmax = np.maximum(z_buy, z_sell)
        dirn = np.where(z_buy >= z_sell, 1, -1).astype(np.int8)
        mask = hit & np.isfinite(zmax) & (zmax >= self.flow.z_thresh)

        if not mask.any():
            return pd.DataFrame(), pd.DataFrame()

        sel = np.where(mask)[0]
        entries = pd.DataFrame({
            "timestamp": pd.to_datetime(sec[sel], unit="s", utc=True).tz_convert(self.sess.tz),
            "price": px[sel],
            "level_px": lvl_px[j[sel]],
            "level_type": lvl_ty[j[sel]],
            "dist_ticks": (px[sel] - lvl_px[j[sel]])/self.sess.tick,
            "lam_buy": lam_buy[sel],
            "lam_sell": lam_sell[sel],
            "z_buy": z_buy[sel],
            "z_sell": z_sell[sel],
            "imb": imb[sel],
            "is_rth": np.array([self._is_rth(int(s_)) for s_ in sec[sel]], dtype=np.int8),
            "direction": dirn[sel],
        })

        labeled = self._label_from_arrays(entries, sec, px)
        return entries, labeled

    def _levels_to_table(self, levels: Dict[str,float]) -> np.ndarray:
        TYPE = {"IBH":1,"IBL":2,"VWAP":3,"PriorH":4,"PriorL":5,"PriorC":6,"POC":7,"VAH":8,"VAL":9}
        rows = []
        for k,v in levels.items():
            if v is not None and np.isfinite(v):
                rows.append((float(v), TYPE.get(k,0)))
        return np.asarray(rows, dtype=np.float64) if rows else np.empty((0,2))

    def _label_from_arrays(self, entries: pd.DataFrame, sec: np.ndarray, px: np.ndarray) -> pd.DataFrame:
        if entries.empty: return entries.assign(y=np.nan)
        # build quick index: sec -> idx
        # we assume sec is strictly increasing
        idx_by_sec = {int(s): i for i, s in enumerate(sec)}
        tgt = self.lab.tgt_ticks * self.sess.tick
        stp = self.lab.stp_ticks * self.sess.tick
        y_list, exit_ts = [], []

        for r in entries.itertuples(index=False):
            s0 = int(pd.Timestamp(r.timestamp).tz_convert("UTC").timestamp())
            i0 = idx_by_sec.get(s0, None)
            if i0 is None:
                y_list.append(np.nan); exit_ts.append(pd.NaT); continue
            end = min(i0 + self.lab.horizon_s, len(px)-1)
            entry_px = float(r.price)
            path = px[i0:end+1]
            # find first crossing
            if r.direction > 0:
                hit_tgt = np.where(path >= entry_px + tgt)[0]
                hit_stp = np.where(path <= entry_px - stp)[0]
            else:
                hit_tgt = np.where(path <= entry_px - tgt)[0]
                hit_stp = np.where(path >= entry_px + stp)[0]
            t_idx = None; ret_ticks = None
            if hit_tgt.size and hit_stp.size:
                t_idx = i0 + (hit_tgt[0] if hit_tgt[0] <= hit_stp[0] else hit_stp[0])
                ret_ticks = self.lab.tgt_ticks if hit_tgt[0] <= hit_stp[0] else -self.lab.stp_ticks
            elif hit_tgt.size:
                t_idx = i0 + hit_tgt[0]; ret_ticks = self.lab.tgt_ticks
            elif hit_stp.size:
                t_idx = i0 + hit_stp[0]; ret_ticks = -self.lab.stp_ticks
            else:
                t_idx = end
                exit_px = float(px[end])
                ret_ticks = (exit_px - entry_px)/self.sess.tick if r.direction>0 else (entry_px - exit_px)/self.sess.tick
            y_list.append(float(ret_ticks))
            exit_ts.append(pd.to_datetime(sec[t_idx], unit="s", utc=True).tz_convert(self.sess.tz))

        return entries.assign(y=y_list, exit_ts=exit_ts)
