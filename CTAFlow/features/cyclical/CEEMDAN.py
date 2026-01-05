# CTAFlow/features/cyclical/ceemdan_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Union

import numpy as np
import pandas as pd
from scipy.signal import hilbert

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    tqdm = None

try:
    # pip install EMD-signal
    from PyEMD import CEEMDAN
except Exception:  # pragma: no cover
    print("Failed to import PyEMD module")
    CEEMDAN = None  # type: ignore


AggMethod = Literal["median", "mean"]
EnvelopeMethod = Literal["hilbert", "quantile", "peak"]
SplitMode = Literal["year", "month", "vol_quantile"]


@dataclass(frozen=True)
class PeriodSlot:
    name: str
    min_period: float
    max_period: float


@dataclass
class RVConfig:
    """
    Realized volatility config.

    We compute daily realized variance from intraday returns:
        RV_day = sum_{intraday} r^2
    and realized volatility:
        sigma_day = sqrt(RV_day)

    You can then split days into volatility quantiles using sigma_day.
    """
    # If True, use daily log returns to compute RV (less ideal for intraday).
    # Default uses intraday returns and aggregates daily.
    use_intraday: bool = True

    # Annualization is optional; quantile splits are scale-invariant.
    annualize: bool = False
    trading_days_per_year: int = 252

    # Winsorize extreme daily sigmas before quantiles (robustness)
    winsor_q: Optional[float] = 0.01  # clip at [q, 1-q]; set None to disable

    # Number of quantile bins (e.g. 5 -> quintiles)
    n_bins: int = 5
    labels: Optional[List[str]] = None  # e.g. ["Q1","Q2","Q3","Q4","Q5"]


@dataclass
class CEEMDANCycleAnalysisConfig:
    # --- input transform ---
    return_mode: Literal["log", "diff", "pct"] = "log"
    demean: bool = True

    # --- CEEMDAN params ---
    trials: int = 20  # Reduced from 40 for better performance (40 is very slow)
    noise_width: float = 0.15
    random_state: Optional[int] = 7
    max_imfs: Optional[int] = None

    # --- rolling analysis ---
    window_days: int = 30
    anchor_time: Optional[str] = "10:00"  # local time; if None, last bar each day
    timezone: str = "America/New_York"
    min_points: int = 600
    mirror_pad: int = 200

    # --- time step units ---
    dt_minutes: Optional[float] = None
    period_clip_min: float = 5.0
    period_clip_max: float = 60.0 * 24.0 * 60.0  # 60 days in minutes

    # --- cycle bands to report ---
    slots: Tuple[PeriodSlot, ...] = (
        PeriodSlot("intra_30_120m", 30, 120),
        PeriodSlot("intra_2_6h", 120, 360),
        PeriodSlot("day_6_24h", 360, 1440),
        PeriodSlot("multiday_1_3d", 1440, 4320),
        PeriodSlot("weekly_3_10d", 4320, 14400),
        PeriodSlot("monthly_10_30d", 14400, 43200),
    )

    # --- prevalence definition ---
    envelope_method: EnvelopeMethod = "quantile"
    envelope_q: float = 0.10
    envelope_window_cycles: float = 1.5

    # --- aggregation ---
    prevalence_agg: AggMethod = "median"

    # --- performance ---
    verbose: bool = True  # Show progress bar during analysis
    anchor_stride: int = 1  # Process every Nth anchor (1 = all, 5 = every 5th day)

    # --- RV + segmentation ---
    rv: RVConfig = None
    split_mode: Optional[SplitMode] = None  # None -> no segmentation
    # For vol_quantile, split is done on daily sigma computed from intraday returns.


class CEEMDANCycleAnalyzer:
    """
    Rolling CEEMDAN cycle analysis + segmentation by year/month/vol quantiles.

    Main outputs:
      - window_table: per-window diagnostics (indexed by anchor timestamps)
      - current_cycles: latest snapshot ranked by slot prevalence
      - segment_summaries: dict of segment -> slot summary tables (optional)
      - segment_counts: counts of windows per segment (optional)
      - daily_rv: daily realized variance/vol table (optional)
    """

    def __init__(self, cfg: Optional[CEEMDANCycleAnalysisConfig] = None):
        self.cfg = cfg or CEEMDANCycleAnalysisConfig()
        if CEEMDAN is None:
            raise ImportError("PyEMD not installed. Install with: pip install EMD-signal")
        if self.cfg.rv is None:
            self.cfg.rv = RVConfig()

        self._ce = CEEMDAN(trials=self.cfg.trials, noise_width=self.cfg.noise_width)
        if self.cfg.random_state is not None:
            self._ce.noise_seed(self.cfg.random_state)

    # ---------------------------
    # Public API
    # ---------------------------
    def analyze(self, price: pd.Series) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        px = self._prep_prices(price)
        r = self._to_returns(px)

        dt = self._infer_dt_minutes(r) if self.cfg.dt_minutes is None else float(self.cfg.dt_minutes)

        anchors = self._build_anchors(r.index)
        window_table = self._compute_windows(r, anchors, dt)
        current_cycles = self._current_cycles(window_table)

        out: Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = {
            "window_table": window_table,
            "current_cycles": current_cycles,
        }

        # RV + segmentation
        daily_rv = self.compute_daily_rv(r)
        out["daily_rv"] = daily_rv

        if self.cfg.split_mode is not None:
            seg = self.segment_windows(window_table, daily_rv=daily_rv)
            out.update(seg)

        return out

    def compute_daily_rv(self, returns: pd.Series) -> pd.DataFrame:
        """
        Compute daily realized variance/volatility from intraday returns.
        Returns dataframe indexed by normalized date with columns: rv, sigma, sigma_ann (optional), vol_bin (optional).
        """
        cfg = self.cfg.rv
        r = returns.dropna().astype(float)

        daily = r.groupby(r.index.normalize()).apply(lambda x: np.sum(np.square(x.values)))
        rv = daily.rename("rv")

        sigma = np.sqrt(rv).rename("sigma")

        df = pd.concat([rv, sigma], axis=1)

        if cfg.annualize:
            df["sigma_ann"] = df["sigma"] * np.sqrt(cfg.trading_days_per_year)

        # winsorize (robust)
        if cfg.winsor_q is not None:
            lo = df["sigma"].quantile(cfg.winsor_q)
            hi = df["sigma"].quantile(1 - cfg.winsor_q)
            df["sigma_w"] = df["sigma"].clip(lo, hi)
        else:
            df["sigma_w"] = df["sigma"]

        # bins (vol quantiles)
        labels = cfg.labels
        if labels is None:
            labels = [f"Q{i+1}" for i in range(cfg.n_bins)]
        df["vol_bin"] = pd.qcut(df["sigma_w"], q=cfg.n_bins, labels=labels, duplicates="drop")

        return df

    def segment_windows(
        self,
        window_table: pd.DataFrame,
        daily_rv: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Union[Dict[str, pd.DataFrame], pd.DataFrame]]:
        """
        Split window_table into segments (year, month, vol_quantile) and summarize each segment.

        Returns dict with:
          - segment_summaries: dict[str -> pd.DataFrame]
          - segment_counts: pd.DataFrame (counts per segment)
        """
        mode = self.cfg.split_mode
        if mode is None:
            return {"segment_summaries": {}, "segment_counts": pd.DataFrame()}

        wt = window_table.copy()
        if wt.empty:
            return {"segment_summaries": {}, "segment_counts": pd.DataFrame()}

        if mode == "year":
            keys = wt.index.to_period("Y").astype(str)

        elif mode == "month":
            keys = wt.index.to_period("M").astype(str)

        elif mode == "vol_quantile":
            if daily_rv is None:
                raise ValueError("daily_rv required for vol_quantile segmentation.")
            # Map each anchor timestamp to its day's vol_bin
            day = wt.index.normalize()
            vb = daily_rv.reindex(day)["vol_bin"]
            keys = vb.astype(str).fillna("NA")

        else:
            raise ValueError(f"Unknown split_mode: {mode}")

        summaries: Dict[str, pd.DataFrame] = {}
        counts = keys.value_counts().sort_index().rename("n_windows").to_frame()

        for seg_name in sorted(keys.unique()):
            seg_mask = keys == seg_name
            seg_wt = wt.loc[seg_mask]
            if seg_wt.empty:
                continue
            summaries[str(seg_name)] = self.summarize_era(seg_wt)

        return {"segment_summaries": summaries, "segment_counts": counts}

    def summarize_era(self, window_table_or_slice: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate prevalence/stability across an already-sliced window_table.
        """
        wt = window_table_or_slice
        if wt.empty:
            raise ValueError("Empty era slice.")

        agg = np.nanmedian if self.cfg.prevalence_agg == "median" else np.nanmean

        rows = []
        for slot in self.cfg.slots:
            amp_col = f"slot.{slot.name}.amp"
            pmed_col = f"slot.{slot.name}.period_med_min"
            prev_col = f"slot.{slot.name}.prevalence"

            p = wt[pmed_col].values
            rows.append({
                "slot": slot.name,
                "period_med_min": float(agg(p)),
                "amp": float(agg(wt[amp_col].values)),
                "prevalence": float(agg(wt[prev_col].values)),
                "period_iqr_min": float(np.nanpercentile(p, 75) - np.nanpercentile(p, 25)),
                "coverage": float(np.mean(np.isfinite(p))),
            })

        out = pd.DataFrame(rows).sort_values("prevalence", ascending=False).reset_index(drop=True)
        return out

    # ---------------------------
    # Prep
    # ---------------------------
    def _prep_prices(self, px: pd.Series) -> pd.Series:
        cfg = self.cfg
        s = px.dropna().astype(float).sort_index()
        if s.index.tz is None:
            s = s.tz_localize(cfg.timezone)
        else:
            s = s.tz_convert(cfg.timezone)
        return s

    def _to_returns(self, px: pd.Series) -> pd.Series:
        cfg = self.cfg
        if cfg.return_mode == "log":
            if (px <= 0).any():
                r = px.diff()
            else:
                r = np.log(px).diff()
        elif cfg.return_mode == "pct":
            r = px.pct_change()
        else:
            r = px.diff()

        r = r.dropna()
        if cfg.demean:
            r = r - r.mean()
        return r

    def _infer_dt_minutes(self, r: pd.Series) -> float:
        diffs = r.index.to_series().diff().dropna()
        if diffs.empty:
            return 5.0
        return float(max(1.0, diffs.median().total_seconds() / 60.0))

    # ---------------------------
    # Anchors: once per day at anchor_time
    # ---------------------------
    def _build_anchors(self, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        cfg = self.cfg
        days = pd.DatetimeIndex(idx.normalize().unique())
        # Only localize if idx has timezone and days doesn't (avoid double-localization)
        if idx.tz is not None and days.tz is None:
            days = days.tz_localize(idx.tz)
        anchors = []

        if cfg.anchor_time is None:
            for d in days:
                day_idx = idx[(idx >= d) & (idx < d + pd.Timedelta(days=1))]
                if len(day_idx):
                    anchors.append(day_idx.max())
            return pd.DatetimeIndex(anchors)

        hh, mm = map(int, cfg.anchor_time.split(":"))
        target_m = hh * 60 + mm

        for d in days:
            day_idx = idx[(idx >= d) & (idx < d + pd.Timedelta(days=1))]
            if not len(day_idx):
                continue
            mins = day_idx.hour * 60 + day_idx.minute
            dist = np.abs(mins - target_m)
            anchors.append(day_idx[np.argmin(dist)])
        return pd.DatetimeIndex(anchors)

    # ---------------------------
    # Rolling windows
    # ---------------------------
    def _compute_windows(self, r: pd.Series, anchors: pd.DatetimeIndex, dt_minutes: float) -> pd.DataFrame:
        cfg = self.cfg
        rows = []

        # Apply anchor stride to reduce computation
        if cfg.anchor_stride > 1:
            anchors = anchors[::cfg.anchor_stride]

        # Setup progress bar if verbose and tqdm available
        iterator = anchors
        if cfg.verbose and tqdm is not None:
            iterator = tqdm(anchors, desc="CEEMDAN windows", unit="window")

        for a in iterator:
            start = a - pd.Timedelta(days=cfg.window_days)
            w = r.loc[start:a].values
            if w.shape[0] < cfg.min_points:
                continue

            imfs = self._ceemdan_window(w)
            if imfs.size == 0:
                continue

            imf_stats = [self._imf_stats(imfs[k], dt_minutes) for k in range(imfs.shape[0])]
            slot_map = self._assign_slots(imf_stats)
            dominant = self._dominant_cycle(imf_stats)

            row = {
                "anchor": a,
                "dominant_period_min": dominant["period_med_min"],
                "dominant_amp": dominant["amp"],
                "dominant_prevalence": dominant["prevalence"],
            }

            for slot in cfg.slots:
                k = slot_map.get(slot.name)
                if k is None:
                    row[f"slot.{slot.name}.period_med_min"] = np.nan
                    row[f"slot.{slot.name}.amp"] = np.nan
                    row[f"slot.{slot.name}.prevalence"] = np.nan
                else:
                    st = imf_stats[k]
                    row[f"slot.{slot.name}.period_med_min"] = st["period_med_min"]
                    row[f"slot.{slot.name}.amp"] = st["amp"]
                    row[f"slot.{slot.name}.prevalence"] = st["prevalence"]

            rows.append(row)

        out = pd.DataFrame(rows).set_index("anchor").sort_index()
        return out

    def _ceemdan_window(self, w: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        x = np.asarray(w, float)

        if cfg.mirror_pad > 0 and x.shape[0] > cfg.mirror_pad + 5:
            p = cfg.mirror_pad
            left = x[1:p + 1][::-1]
            right = x[-p - 1:-1][::-1]
            x_pad = np.r_[left, x, right]
            pad_offset = p
        else:
            x_pad = x
            pad_offset = 0

        # Call CEEMDAN with max_imf parameter
        if cfg.max_imfs is not None:
            imfs = self._ce.ceemdan(x_pad, max_imf=cfg.max_imfs)
        else:
            imfs = self._ce.ceemdan(x_pad)
        imfs = np.asarray(imfs)
        if imfs.size == 0:
            return imfs

        if pad_offset:
            return imfs[:, pad_offset:pad_offset + x.shape[0]]
        return imfs

    # ---------------------------
    # IMF stats
    # ---------------------------
    def _imf_stats(self, imf: np.ndarray, dt_minutes: float) -> Dict[str, float]:
        cfg = self.cfg
        c = np.asarray(imf, float)

        z = hilbert(c)
        phase = np.unwrap(np.angle(z))
        dphi = np.diff(phase)

        freq = np.abs(dphi) / (2 * np.pi * dt_minutes)  # cycles/min
        freq[freq == 0] = np.nan

        per = 1.0 / freq
        per = per[np.isfinite(per) & (per >= cfg.period_clip_min) & (per <= cfg.period_clip_max)]
        period_med = float(np.nanmedian(per)) if per.size else np.nan

        amp_now, prev = self._envelope_prevalence(c)
        return {"period_med_min": period_med, "amp": amp_now, "prevalence": prev}

    def _envelope_prevalence(self, c: np.ndarray) -> Tuple[float, float]:
        """
        Returns:
          amp_now: "current" amplitude proxy at end of window
          prev: prevalence proxy across window (typical amplitude)
        """
        cfg = self.cfg
        n = len(c)
        # local window size: robust default
        w = max(50, int(0.25 * n))

        if cfg.envelope_method == "hilbert":
            env = np.abs(hilbert(c))
            return float(env[-1]), float(np.nanmedian(env))

        if cfg.envelope_method == "peak":
            amp_now = 0.5 * (np.max(c[-w:]) - np.min(c[-w:]))
            blocks = np.array_split(c, max(5, n // w))
            amps = [0.5 * (np.max(b) - np.min(b)) for b in blocks if len(b) > 10]
            prev = float(np.nanmedian(amps)) if amps else np.nan
            return float(amp_now), prev

        # quantile envelope (robust full-move amplitude proxy)
        q = cfg.envelope_q
        amp_now = 0.5 * (np.quantile(c[-w:], 1 - q) - np.quantile(c[-w:], q))
        blocks = np.array_split(c, max(5, n // w))
        amps = []
        for b in blocks:
            if len(b) < 10:
                continue
            amps.append(0.5 * (np.quantile(b, 1 - q) - np.quantile(b, q)))
        prev = float(np.nanmedian(amps)) if amps else np.nan
        return float(amp_now), prev

    # ---------------------------
    # Slot assignment + dominance
    # ---------------------------
    def _assign_slots(self, imf_stats: List[Dict[str, float]]) -> Dict[str, int]:
        cfg = self.cfg
        assign: Dict[str, int] = {}
        used = set()

        for slot in cfg.slots:
            center = 0.5 * (slot.min_period + slot.max_period)
            cands = []
            for k, st in enumerate(imf_stats):
                p = st["period_med_min"]
                if np.isnan(p):
                    continue
                if slot.min_period <= p <= slot.max_period:
                    cands.append((k, abs(p - center)))
            cands.sort(key=lambda kv: kv[1])
            for k, _ in cands:
                if k not in used:
                    assign[slot.name] = k
                    used.add(k)
                    break
        return assign

    def _dominant_cycle(self, imf_stats: List[Dict[str, float]]) -> Dict[str, float]:
        best = None
        for st in imf_stats:
            if np.isnan(st["prevalence"]) or np.isnan(st["period_med_min"]):
                continue
            if best is None or st["prevalence"] > best["prevalence"]:
                best = st
        return best or {"period_med_min": np.nan, "amp": np.nan, "prevalence": np.nan}

    def _current_cycles(self, window_table: pd.DataFrame) -> pd.DataFrame:
        if window_table.empty:
            return pd.DataFrame()
        last = window_table.iloc[-1]
        rows = []
        for slot in self.cfg.slots:
            rows.append({
                "slot": slot.name,
                "period_med_min": float(last.get(f"slot.{slot.name}.period_med_min", np.nan)),
                "amp": float(last.get(f"slot.{slot.name}.amp", np.nan)),
                "prevalence": float(last.get(f"slot.{slot.name}.prevalence", np.nan)),
            })
        return pd.DataFrame(rows).sort_values("prevalence", ascending=False).reset_index(drop=True)
