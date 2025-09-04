
import os
import re
import calendar
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

MONTH_CODE_MAP = {
    "F": 1,  # Jan
    "G": 2,  # Feb
    "H": 3,  # Mar
    "J": 4,  # Apr
    "K": 5,  # May
    "M": 6,  # Jun
    "N": 7,  # Jul
    "Q": 8,  # Aug
    "U": 9,  # Sep
    "V": 10, # Oct
    "X": 11, # Nov
    "Z": 12, # Dec
}
ORDER_LETTERS = ["F","G","H","J","K","M","N","Q","U","V","X","Z"]

def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _find_columns_ci(cols, name_candidates) -> Optional[str]:
    cols_lower = {str(c).lower(): c for c in cols}
    for cand in name_candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

class FuturesCurveManager:
    """
    Reusable manager for futures forward curves.
    - Reads per-contract CSVs named like 'CL_K.csv'
    - Builds a wide curve (columns = month code letters) indexed by date
    - Computes symbol-aware days-to-expiry (DTE) and detects front month daily
    - Refines rolls with a jump detector on M0/M1 spread
    - Creates sequential forward curve views (M0/M1/...) and spreads vs M0
    - Saves artifacts into HDF (e.g., 'CL/curve', 'CL/front_month', etc.)
    """

    def __init__(
        self,
        root: str,
        symbol: str,
        hdf_path: str,
        expiry_rules: Optional[Dict[str, Callable[[int, int], pd.Timestamp]]] = None,
    ):
        self.root = root
        self.symbol = symbol.upper()
        self.hdf_path = hdf_path
        self.expiry_rules = expiry_rules or {}
        # Preload if already built by caller
        self.curve: Optional[pd.DataFrame] = None
        self.dte: Optional[pd.DataFrame] = None
        self.front: Optional[pd.Series] = None
        self.seq_prices: Optional[pd.DataFrame] = None
        self.seq_labels: Optional[pd.DataFrame] = None
        self.seq_dte: Optional[pd.DataFrame] = None
        self.seq_spreads: Optional[pd.DataFrame] = None

    # ------------------------------
    # Ingestion
    # ------------------------------

    def parse_contract_from_filename(self, filename: str) -> Tuple[str, Optional[str]]:
        base = os.path.basename(filename)
        base = re.sub(r"\.csv$", "", base, flags=re.IGNORECASE)
        m = re.match(r"^([A-Z]+)[\-_]?([FGHJKMNQUVXZ])$", base)
        if m:
            return m.group(1).upper(), m.group(2).upper()
        parts = re.split(r"[\-_]", base)
        if parts and parts[-1].upper() in MONTH_CODE_MAP:
            return parts[0].upper(), parts[-1].upper()
        return base.upper(), None

    def collect_symbol_contract_files(self) -> Dict[str, str]:
        files: Dict[str, str] = {}
        for fn in os.listdir(self.root):
            if not fn.lower().endswith(".csv"):
                continue
            path = os.path.join(self.root, fn)
            sym, mcode = self.parse_contract_from_filename(fn)
            if sym == self.symbol and mcode in MONTH_CODE_MAP:
                files[mcode] = path
        return dict(sorted(files.items(), key=lambda kv: MONTH_CODE_MAP[kv[0]]))

    def read_contract_csv(self, fp: str) -> pd.DataFrame:
        df = pd.read_csv(fp)
        df = _strip_cols(df)
        date_col = _find_columns_ci(df.columns, ["date", "datetime", "timestamp"])
        if date_col is None:
            raise ValueError(f"[{fp}] No date-like column found.")
        close_col = _find_columns_ci(
            df.columns,
            # Treat "Last" as the close/settle; include variants
            ["settle", "settlement", "close", "last", "adj close", "price", "Last"]
        )
        if close_col is None:
            raise ValueError(f"[{fp}] No close/settle-like column found. Columns={list(df.columns)}")
        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        })
        out = out.dropna(subset=["date"]).sort_values("date").set_index("date")
        # Deduplicate dates
        out = out[~out.index.duplicated(keep="last")]
        return out

    def load_front_month_series(self) -> Optional[pd.Series]:
        candidates = [
            os.path.join(self.root, f"{self.symbol}_front_month.csv"),
            os.path.join(self.root, f"{self.symbol}-front-month.csv"),
            os.path.join(self.root, f"{self.symbol}_front.csv"),
        ]
        for fp in candidates:
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                df = _strip_cols(df)
                dcol = _find_columns_ci(df.columns, ["date", "datetime", "timestamp"])
                pcol = _find_columns_ci(df.columns, ["close", "settle", "settlement", "last", "price"])
                if dcol and pcol:
                    ser = pd.Series(pd.to_numeric(df[pcol], errors="coerce").values,
                                    index=pd.to_datetime(df[dcol], errors="coerce"))
                    ser = ser.dropna().sort_index()
                    ser = ser[~ser.index.duplicated(keep="last")]
                    ser.name = f"{self.symbol}_front"
                    return ser
        return None

    def build_curve(self) -> pd.DataFrame:
        fmap = self.collect_symbol_contract_files()
        if not fmap:
            raise RuntimeError(f"No contract CSVs found for {self.symbol} in {self.root}")
        frames: List[pd.DataFrame] = []
        for mcode, fp in fmap.items():
            df = self.read_contract_csv(fp).rename(columns={"close": mcode})
            frames.append(df[[mcode]])
        curve = pd.concat(frames, axis=1).sort_index()
        self.curve = curve
        return curve

    # ------------------------------
    # Expiry & DTE
    # ------------------------------

    @staticmethod
    def last_business_day(year: int, month: int) -> pd.Timestamp:
        last_day = calendar.monthrange(year, month)[1]
        dt = datetime(year, month, last_day)
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)
        return pd.Timestamp(dt.date())

    @staticmethod
    def business_day_before(date: pd.Timestamp, n: int) -> pd.Timestamp:
        d = pd.Timestamp(date.date())
        count = 0
        while count < n:
            d -= pd.Timedelta(days=1)
            if d.weekday() < 5:
                count += 1
        return d

    @staticmethod
    def cme_wti_last_trading_day(delivery_year: int, delivery_month: int) -> pd.Timestamp:
        # 3 business days prior to the 25th of the month preceding delivery
        if delivery_month == 1:
            y2, m2 = delivery_year - 1, 12
        else:
            y2, m2 = delivery_year, delivery_month - 1
        date_25 = pd.Timestamp(datetime(y2, m2, 25))
        return FuturesCurveManager.business_day_before(date_25, 3)

    @staticmethod
    def effective_delivery_year_for_date(d: pd.Timestamp, delivery_month: int) -> int:
        return d.year if d.month <= delivery_month else d.year + 1

    def contract_expiry(self, delivery_year: int, delivery_month: int) -> pd.Timestamp:
        # Symbol-specific rule if provided
        if self.symbol in self.expiry_rules:
            return self.expiry_rules[self.symbol](delivery_year, delivery_month)
        # Built-in special case for CL
        if self.symbol == "CL":
            return self.cme_wti_last_trading_day(delivery_year, delivery_month)
        # Generic fallback: last business day of prior month
        if delivery_month == 1:
            y, m = delivery_year - 1, 12
        else:
            y, m = delivery_year, delivery_month - 1
        return self.last_business_day(y, m)

    def days_to_expiry_for_date(self, d: pd.Timestamp, month_code: str) -> int:
        m = MONTH_CODE_MAP[month_code]
        y = self.effective_delivery_year_for_date(d, m)
        exp = self.contract_expiry(y, m)
        return int((exp - d.normalize()).days)

    def compute_dte_matrix(self, curve: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        curve = curve if curve is not None else self.curve
        if curve is None:
            raise RuntimeError("Curve not built. Call build_curve() first.")
        mat = pd.DataFrame(index=curve.index, columns=curve.columns, dtype="float")
        for mcode in curve.columns:
            mat[mcode] = [self.days_to_expiry_for_date(d, mcode) for d in curve.index]
        self.dte = mat
        return mat

    # ------------------------------
    # Front-month detection
    # ------------------------------

    def assign_front_calendar(self, curve: Optional[pd.DataFrame] = None) -> pd.Series:
        curve = curve if curve is not None else self.curve
        if curve is None:
            raise RuntimeError("Curve not built. Call build_curve() first.")
        if self.dte is None:
            self.compute_dte_matrix(curve)

        fm = []
        for d in curve.index:
            candidates = []
            for mcode in curve.columns:
                px = curve.at[d, mcode]
                dd = self.dte.at[d, mcode]
                if pd.notna(px) and pd.notna(dd) and dd >= 0:
                    candidates.append((dd, mcode))
            if candidates:
                candidates.sort(key=lambda x: (x[0], ORDER_LETTERS.index(x[1]) if x[1] in ORDER_LETTERS else 99))
                fm.append(candidates[0][1])
            else:
                fm.append(None)
        s = pd.Series(fm, index=curve.index, name="front_month")
        return s

    def assign_front_by_match(self, front_series: pd.Series, tol: float = 0.01) -> pd.Series:
        curve = self.curve
        if curve is None:
            raise RuntimeError("Curve not built. Call build_curve() first.")
        join = curve.join(front_series.rename("front"), how="left")
        fm = pd.Series(index=curve.index, dtype="object")
        for d, row in join.iterrows():
            f = row.get("front", np.nan)
            if not np.isfinite(f):
                continue
            errs = {}
            for mcode in curve.columns:
                c = row.get(mcode, np.nan)
                if not np.isfinite(c):
                    continue
                rel_err = abs(c - f) / (abs(f) if f != 0 else np.nan)
                if np.isfinite(rel_err) and rel_err <= tol:
                    errs[mcode] = rel_err
            if errs:
                fm.at[d] = min(errs.items(), key=lambda kv: kv[1])[0]
        return fm.ffill().bfill().rename("front_month")

    def refine_front_by_jump(
        self,
        base_front: pd.Series,
        rel_jump_thresh: float = 0.01,
        robust_k: float = 4.0,
        lookback: int = 10,
        near_expiry_days: int = 15,
    ) -> pd.Series:
        curve = self.curve
        if curve is None:
            raise RuntimeError("Curve not built. Call build_curve() first.")
        if self.dte is None:
            self.compute_dte_matrix(curve)

        # Build M0/M1 calendar order (non-expired)
        M0, M1 = [], []
        for d in curve.index:
            cands = []
            for m in curve.columns:
                px = curve.at[d, m]
                dd = self.dte.at[d, m]
                if pd.notna(px) and pd.notna(dd) and dd >= 0:
                    cands.append((dd, m))
            cands.sort(key=lambda x: x[0])
            M0.append(cands[0][1] if len(cands) >= 1 else None)
            M1.append(cands[1][1] if len(cands) >= 2 else None)
        M0 = pd.Series(M0, index=curve.index, name="M0cal")
        M1 = pd.Series(M1, index=curve.index, name="M1cal")

        # Spread s = p(M1) - p(M0)
        s = pd.Series(index=curve.index, dtype="float")
        for d in curve.index:
            m0, m1 = M0.at[d], M1.at[d]
            if m0 and m1:
                p0 = curve.at[d, m0]
                p1 = curve.at[d, m1]
                if pd.notna(p0) and pd.notna(p1):
                    s.at[d] = p1 - p0

        ds = s.diff()
        rolling_mad = ds.rolling(lookback).apply(
            lambda x: np.median(np.abs(x - np.nanmedian(x))), raw=True
        ).fillna(0)

        refined = base_front.copy()
        for d in curve.index[1:]:
            m0 = M0.at[d]
            m1 = M1.at[d]
            if not m0 or not m1:
                continue
            p0 = curve.at[d, m0]
            jump = abs(ds.at[d]) if pd.notna(ds.at[d]) else 0.0
            mad = rolling_mad.at[d] if pd.notna(rolling_mad.at[d]) else 0.0
            rel_gate = rel_jump_thresh * (p0 if pd.notna(p0) else 0.0)
            gate = max(robust_k * mad, rel_gate)
            dd0 = self.dte.at[d, m0] if (m0 in self.dte.columns) else 9999
            if (jump > gate) and (dd0 <= near_expiry_days):
                refined.at[d] = m1
        refined = refined.ffill().where(base_front.notna(), refined)
        return refined.rename("front_month")

    # ------------------------------
    # Sequential forward curve
    # ------------------------------

    def sequentialize(self, fm_labels: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        curve = self.curve
        if curve is None:
            raise RuntimeError("Curve not built. Call build_curve() first.")
        if self.dte is None:
            self.compute_dte_matrix(curve)

        max_cols = curve.shape[1]
        mcols = [f"M{i}" for i in range(max_cols)]
        seq_prices = pd.DataFrame(index=curve.index, columns=mcols, dtype="float")
        seq_labels = pd.DataFrame(index=curve.index, columns=mcols, dtype="object")
        seq_dte    = pd.DataFrame(index=curve.index, columns=mcols, dtype="float")

        # Pre-rank available non-expired by DTE for each date
        ranked = {}
        for d in curve.index:
            recs = []
            for m in curve.columns:
                px = curve.at[d, m]
                dd = self.dte.at[d, m]
                if pd.notna(px) and pd.notna(dd) and dd >= 0:
                    recs.append((dd, m, px))
            recs.sort(key=lambda x: (x[0], ORDER_LETTERS.index(x[1]) if x[1] in ORDER_LETTERS else 99))
            ranked[d] = recs

        for d in curve.index:
            recs = ranked[d]
            if not recs:
                continue
            fm = fm_labels.at[d] if d in fm_labels.index else None
            ordered = []
            if fm:
                present = [r for r in recs if r[1] == fm]
                if present:
                    ordered.append(present[0])
                    for r in recs:
                        if r[1] != fm:
                            ordered.append(r)
                else:
                    ordered = recs
            else:
                ordered = recs

            for i, (t, m, px) in enumerate(ordered[:max_cols]):
                j = i
                seq_prices.iat[curve.index.get_loc(d), j] = px
                seq_labels.iat[curve.index.get_loc(d), j] = m
                seq_dte.iat[curve.index.get_loc(d), j]    = t

        self.seq_prices, self.seq_labels, self.seq_dte = seq_prices, seq_labels, seq_dte
        return seq_prices, seq_labels, seq_dte

    # ------------------------------
    # Spreads and persistence
    # ------------------------------

    @staticmethod
    def spreads_vs_front(seq_prices: pd.DataFrame) -> pd.DataFrame:
        return seq_prices.sub(seq_prices["M0"], axis=0)

    def save_to_hdf(self):
        if self.curve is None:
            raise RuntimeError("Nothing to save; curve is None.")
        with pd.HDFStore(self.hdf_path) as store:
            store.put(f"{self.symbol}/curve", self.curve, format="table", data_columns=True)
            if self.front is not None:
                store.put(f"{self.symbol}/front_month", self.front.to_frame(), format="table", data_columns=True)
            if self.dte is not None:
                store.put(f"{self.symbol}/days_to_expiry", self.dte, format="table", data_columns=True)
            if self.seq_prices is not None:
                store.put(f"{self.symbol}/curve_seq", self.seq_prices, format="table", data_columns=True)
            if self.seq_labels is not None:
                store.put(f"{self.symbol}/curve_seq_labels", self.seq_labels, format="table", data_columns=True)
            if self.seq_dte is not None:
                store.put(f"{self.symbol}/days_to_expiry_seq", self.seq_dte, format="table", data_columns=True)
            if self.seq_prices is not None:
                store.put(f"{self.symbol}/spreads_seq", self.spreads_vs_front(self.seq_prices), format="table", data_columns=True)

    # ------------------------------
    # Orchestrate end-to-end
    # ------------------------------

    def run(
        self,
        prefer_front_series: bool = True,
        match_tol: float = 0.01,
        rel_jump_thresh: float = 0.01,
        robust_k: float = 4.0,
        lookback: int = 10,
        near_expiry_days: int = 15,
        save: bool = True,
    ) -> Dict[str, str]:
        """
        Build curve -> DTE -> front-month (match or calendar) -> refine by jump -> sequentialize -> save
        Returns HDF keys written.
        """
        self.build_curve()
        self.compute_dte_matrix(self.curve)

        # Front-month selection
        front_series = self.load_front_month_series() if prefer_front_series else None
        if front_series is not None:
            fm0 = self.assign_front_by_match(front_series, tol=match_tol)
            # Ensure not expired; fallback to calendar where necessary
            cal = self.assign_front_calendar(self.curve)
            # Use calendar where fm0 is missing
            fm0 = fm0.where(fm0.notna(), cal)
            fm = self.refine_front_by_jump(
                fm0, rel_jump_thresh=rel_jump_thresh, robust_k=robust_k,
                lookback=lookback, near_expiry_days=near_expiry_days
            )
        else:
            cal = self.assign_front_calendar(self.curve)
            fm = self.refine_front_by_jump(
                cal, rel_jump_thresh=rel_jump_thresh, robust_k=robust_k,
                lookback=lookback, near_expiry_days=near_expiry_days
            )
        self.front = fm

        # Sequentialize and spreads
        seq_prices, seq_labels, seq_dte = self.sequentialize(self.front)
        self.seq_prices, self.seq_labels, self.seq_dte = seq_prices, seq_labels, seq_dte

        if save:
            self.save_to_hdf()

        return {
            "curve": f"{self.symbol}/curve",
            "front": f"{self.symbol}/front_month",
            "dte": f"{self.symbol}/days_to_expiry",
            "seq_curve": f"{self.symbol}/curve_seq",
            "seq_labels": f"{self.symbol}/curve_seq_labels",
            "seq_dte": f"{self.symbol}/days_to_expiry_seq",
            "seq_spreads": f"{self.symbol}/spreads_seq",
        }
