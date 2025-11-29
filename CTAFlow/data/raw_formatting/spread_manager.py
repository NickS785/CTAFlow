import os
import re
import calendar
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple, List, Any, Union, Sequence

from scipy import stats
from scipy.stats import skew, kurtosis

from ..data_client import DataClient
from ...config import *

import numpy as np
import pandas as pd

# Local definition to avoid circular imports
MONTH_CODE_ORDER = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']

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
    "V": 10,  # Oct
    "X": 11,  # Nov
    "Z": 12,  # Dec
}
ORDER_LETTERS = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]


_CONTRACT_CODE_PATTERN = re.compile(r"([FGHJKMNQUVXZ])", re.IGNORECASE)
_YEAR_SUFFIX_PATTERN = re.compile(r"(\d{2,4})")


def normalize_contract_code(contract_code: Optional[str]) -> Optional[str]:
    """Return the month letter embedded in a contract label.

    Handles labels that include year information (e.g. "H2025", "M_25").
    Returns ``None`` if no valid month code is found.
    """
    if contract_code is None:
        return None

    match = _CONTRACT_CODE_PATTERN.search(str(contract_code).upper())
    if match:
        return match.group(1)
    return None


def infer_contract_year(contract_code: Optional[str], reference_year: int) -> Optional[int]:
    """Infer contract year from label if a numeric suffix is provided."""
    if not contract_code:
        return None

    match = _YEAR_SUFFIX_PATTERN.search(str(contract_code))
    if not match:
        return None

    year_str = match.group(1)
    try:
        year_val = int(year_str)
    except ValueError:
        return None

    if len(year_str) == 2:
        # Map to the closest year in the same century as reference_year
        century = (reference_year // 100) * 100
        candidate = century + year_val
        if candidate < reference_year - 50:
            candidate += 100
        elif candidate > reference_year + 50:
            candidate -= 100
        return candidate

    if len(year_str) == 4:
        return year_val

    return None


def dte_to_months(dte_value: Optional[float]) -> Optional[float]:
    """Convert days-to-expiry to approximate months."""
    if dte_value is None:
        return None
    try:
        if np.isnan(dte_value):
            return None
    except TypeError:
        pass
    return float(dte_value) / 30.4375


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
            symbol: str,
            root: str = None,
            hdf_path: str = None,
            expiry_rules: Optional[Dict[str, Callable[[int, int], pd.Timestamp]]] = None,

    ):
        # For passing symbols denoted as "_F", examples: "ZC_F", "ZCH25", "ZCH2025"
        if len(symbol) > 2:
            symbol = symbol[:2]

        self.root = root or RAW_MARKET_DATA_PATH / 'daily' / symbol
        self.symbol = symbol.upper()
        self.hdf_path = hdf_path or MARKET_DATA_PATH
        self.expiry_rules = expiry_rules or {}
        # Preload if already built by caller
        self.curve: Optional[pd.DataFrame] = None
        self.volume_curve: Optional[pd.DataFrame] = None
        self.oi_curve: Optional[pd.DataFrame] = None
        self.dte: Optional[pd.DataFrame] = None
        self.front: Optional[pd.Series] = None
        self.seq_prices: Optional[pd.DataFrame] = None
        self.seq_volume: Optional[pd.DataFrame] = None
        self.seq_oi: Optional[pd.DataFrame] = None
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

        # Look for volume and open interest columns
        volume_col = _find_columns_ci(
            df.columns,
            ["volume", "vol", "total volume", "totalvolume", "Volume"]
        )
        oi_col = _find_columns_ci(
            df.columns,
            ["open interest", "openinterest", "oi", "open_interest", "OpenInterest", "OI"]
        )

        # Build output DataFrame with available columns
        out_data = {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }

        if volume_col is not None:
            out_data["volume"] = pd.to_numeric(df[volume_col], errors="coerce")

        if oi_col is not None:
            out_data["oi"] = pd.to_numeric(df[oi_col], errors="coerce")

        out = pd.DataFrame(out_data)
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

        price_frames: List[pd.DataFrame] = []
        volume_frames: List[pd.DataFrame] = []
        oi_frames: List[pd.DataFrame] = []

        has_volume = False
        has_oi = False

        for mcode, fp in fmap.items():
            df = self.read_contract_csv(fp)

            # Price data (required)
            price_df = df[["close"]].rename(columns={"close": mcode})
            price_frames.append(price_df)

            # Volume data (optional)
            if "volume" in df.columns:
                has_volume = True
                volume_df = df[["volume"]].rename(columns={"volume": mcode})
                volume_frames.append(volume_df)

            # Open Interest data (optional)
            if "oi" in df.columns:
                has_oi = True
                oi_df = df[["oi"]].rename(columns={"oi": mcode})
                oi_frames.append(oi_df)

        # Build curves
        self.curve = pd.concat(price_frames, axis=1).sort_index()

        if has_volume and volume_frames:
            self.volume_curve = pd.concat(volume_frames, axis=1).sort_index()

        if has_oi and oi_frames:
            self.oi_curve = pd.concat(oi_frames, axis=1).sort_index()

        return self.curve

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
        """
        Return the last trading day / expiry approximation for the given (year, delivery_month)
        using built-ins for common NYMEX energy contracts:
          - CL (WTI): 3 business days prior to the 25th calendar day of the month preceding delivery
          - NG (Henry Hub): 3 business days prior to the FIRST day of the delivery month
          - HO (NY Harbor ULSD): last business day of the month PRIOR to the delivery month
          - RB (RBOB Gasoline): last business day of the month PRIOR to the delivery month
        You may override/extend behavior by passing expiry_rules at init.
        """
        # 1) Explicit user-provided override
        if self.symbol in self.expiry_rules:
            return self.expiry_rules[self.symbol](delivery_year, delivery_month)

        # 2) Built-ins
        sym = self.symbol
        if sym == "CL":
            return self.cme_wti_last_trading_day(delivery_year, delivery_month)

        if sym == "NG":
            # 3 business days prior to the first calendar day of the delivery month
            first = pd.Timestamp(datetime(delivery_year, delivery_month, 1))
            return self.business_day_before(first, 3)

        if sym in {"HO", "RB"}:
            # Last business day of the month prior to the delivery month
            if delivery_month == 1:
                y, m = delivery_year - 1, 12
            else:
                y, m = delivery_year, delivery_month - 1
            return self.last_business_day(y, m)

        # 3) Generic fallback: last business day of prior month
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

    def refine_front_by_jump_fixed(
            self,
            base_front: pd.Series,
            rel_jump_thresh: float = 0.01,
            robust_k: float = 4.0,
            lookback: int = 10,
            near_expiry_days: int = 15,
            min_persistence_days: int = 2,  # NEW: Require persistence
            smooth_window: int = 3,  # NEW: Smooth spread before detection
    ) -> pd.Series:
        """
        FIXED VERSION: Refine front month detection with persistence check to avoid single-day flip-flops.

        Key fixes:
        1. Smooth the spread before detecting jumps to reduce noise
        2. Require jumps to persist for min_persistence_days before switching
        3. Add hysteresis to prevent immediate roll-back
        4. Better handling of edge cases
        """
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
            cands.sort(key=lambda x: (x[0], ORDER_LETTERS.index(x[1]) if x[1] in ORDER_LETTERS else 99))
            M0.append(cands[0][1] if len(cands) >= 1 else None)
            M1.append(cands[1][1] if len(cands) >= 2 else None)

        M0 = pd.Series(M0, index=curve.index, name="M0cal")
        M1 = pd.Series(M1, index=curve.index, name="M1cal")

        # Calculate spread s = p(M1) - p(M0)
        s = pd.Series(index=curve.index, dtype="float")
        for d in curve.index:
            m0, m1 = M0.at[d], M1.at[d]
            if m0 and m1:
                p0 = curve.at[d, m0]
                p1 = curve.at[d, m1]
                if pd.notna(p0) and pd.notna(p1):
                    s.at[d] = p1 - p0

        # FIX 1: Smooth the spread to reduce noise
        if smooth_window > 1:
            s_smoothed = s.rolling(window=smooth_window, center=True, min_periods=1).mean()
        else:
            s_smoothed = s

        # Calculate spread changes on smoothed data
        ds = s_smoothed.diff()

        # Robust MAD calculation
        rolling_mad = ds.rolling(lookback).apply(
            lambda x: np.median(np.abs(x - np.nanmedian(x))) if len(x) > 0 else 0,
            raw=True
        ).fillna(0)

        # FIX 2: Track roll signals with persistence
        roll_signal = pd.Series(False, index=curve.index)

        for d in curve.index[1:]:
            m0 = M0.at[d]
            m1 = M1.at[d]
            if not m0 or not m1:
                continue

            p0 = curve.at[d, m0]
            jump = abs(ds.at[d]) if pd.notna(ds.at[d]) else 0.0
            mad = rolling_mad.at[d] if pd.notna(rolling_mad.at[d]) else 0.0
            rel_gate = rel_jump_thresh * (abs(p0) if pd.notna(p0) else 1.0)
            gate = max(robust_k * mad, rel_gate)
            dd0 = self.dte.at[d, m0] if (m0 in self.dte.columns) else 9999

            # Signal a potential roll if jump exceeds threshold and near expiry
            if (jump > gate) and (dd0 <= near_expiry_days):
                roll_signal.at[d] = True

        # FIX 3: Apply persistence filter - require consecutive signals
        if min_persistence_days > 1:
            # Use rolling window to check for persistent signals
            roll_confirmed = roll_signal.rolling(
                window=min_persistence_days,
                min_periods=min_persistence_days
            ).sum() >= min_persistence_days
        else:
            roll_confirmed = roll_signal

        # FIX 4: Apply rolls with hysteresis to prevent flip-flopping
        refined = base_front.copy()
        rolled = False  # Track if we've already rolled to prevent immediate roll-back
        last_roll_date = None

        for i, d in enumerate(curve.index[1:], 1):
            m0 = M0.at[d]
            m1 = M1.at[d]

            if not m0 or not m1:
                continue

            # Check if we should roll
            if roll_confirmed.at[d]:
                # Only roll if we haven't rolled recently (hysteresis)
                if last_roll_date is None or (d - last_roll_date).days > 5:
                    refined.at[d] = m1
                    rolled = True
                    last_roll_date = d

                    # FIX 5: Forward fill the roll for at least a few days
                    # This prevents single-day reversions
                    future_dates = curve.index[i:min(i + 3, len(curve.index))]
                    for future_d in future_dates:
                        if self.dte.at[future_d, m1] >= 0:  # Only if M1 hasn't expired
                            refined.at[future_d] = m1

            # If we've rolled and M1 becomes the natural front month, maintain it
            elif rolled and base_front.at[d] == m1:
                refined.at[d] = m1

        # Forward fill and handle NaNs
        refined = refined.ffill().where(base_front.notna(), base_front)

        return refined.rename("front_month")

    # Additional helper method to diagnose roll issues
    def diagnose_rolls(self, front_series: pd.Series) -> pd.DataFrame:
        """
        Diagnose potential roll issues by identifying single-day changes.
        Returns DataFrame with dates where front month changes for only 1 day.
        """
        if front_series is None or len(front_series) < 3:
            return pd.DataFrame()

        issues = []

        for i in range(1, len(front_series) - 1):
            prev_val = front_series.iloc[i - 1]
            curr_val = front_series.iloc[i]
            next_val = front_series.iloc[i + 1]

            # Check for single-day change (flip-flop pattern)
            if (prev_val != curr_val) and (curr_val != next_val) and (prev_val == next_val):
                issues.append({
                    'date': front_series.index[i],
                    'prev_contract': prev_val,
                    'single_day_contract': curr_val,
                    'next_contract': next_val
                })

        return pd.DataFrame(issues)

    def refine_front_by_jump(
            self,
            base_front: pd.Series,
            rel_jump_thresh: float = 0.01,
            robust_k: float = 4.0,
            lookback: int = 10,
            near_expiry_days: int = 15,
            min_persistence_days: int = 2,  # NEW: Require persistence
            smooth_window: int = 3,  # NEW: Smooth spread before detection
    ) -> pd.Series:
        """
        FIXED VERSION: Refine front month detection with persistence check to avoid single-day flip-flops.

        Key fixes:
        1. Smooth the spread before detecting jumps to reduce noise
        2. Require jumps to persist for min_persistence_days before switching
        3. Add hysteresis to prevent immediate roll-back
        4. Better handling of edge cases
        """
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
            cands.sort(key=lambda x: (x[0], ORDER_LETTERS.index(x[1]) if x[1] in ORDER_LETTERS else 99))
            M0.append(cands[0][1] if len(cands) >= 1 else None)
            M1.append(cands[1][1] if len(cands) >= 2 else None)

        M0 = pd.Series(M0, index=curve.index, name="M0cal")
        M1 = pd.Series(M1, index=curve.index, name="M1cal")

        # Calculate spread s = p(M1) - p(M0)
        s = pd.Series(index=curve.index, dtype="float")
        for d in curve.index:
            m0, m1 = M0.at[d], M1.at[d]
            if m0 and m1:
                p0 = curve.at[d, m0]
                p1 = curve.at[d, m1]
                if pd.notna(p0) and pd.notna(p1):
                    s.at[d] = p1 - p0

        # FIX 1: Smooth the spread to reduce noise
        if smooth_window > 1:
            s_smoothed = s.rolling(window=smooth_window, center=True, min_periods=1).mean()
        else:
            s_smoothed = s

        # Calculate spread changes on smoothed data
        ds = s_smoothed.diff()

        # Robust MAD calculation
        rolling_mad = ds.rolling(lookback).apply(
            lambda x: np.median(np.abs(x - np.nanmedian(x))) if len(x) > 0 else 0,
            raw=True
        ).fillna(0)

        # FIX 2: Track roll signals with persistence
        roll_signal = pd.Series(False, index=curve.index)

        for d in curve.index[1:]:
            m0 = M0.at[d]
            m1 = M1.at[d]
            if not m0 or not m1:
                continue

            p0 = curve.at[d, m0]
            jump = abs(ds.at[d]) if pd.notna(ds.at[d]) else 0.0
            mad = rolling_mad.at[d] if pd.notna(rolling_mad.at[d]) else 0.0
            rel_gate = rel_jump_thresh * (abs(p0) if pd.notna(p0) else 1.0)
            gate = max(robust_k * mad, rel_gate)
            dd0 = self.dte.at[d, m0] if (m0 in self.dte.columns) else 9999

            # Signal a potential roll if jump exceeds threshold and near expiry
            if (jump > gate) and (dd0 <= near_expiry_days):
                roll_signal.at[d] = True

        # FIX 3: Apply persistence filter - require consecutive signals
        if min_persistence_days > 1:
            # Use rolling window to check for persistent signals
            roll_confirmed = roll_signal.rolling(
                window=min_persistence_days,
                min_periods=min_persistence_days
            ).sum() >= min_persistence_days
        else:
            roll_confirmed = roll_signal

        # FIX 4: Apply rolls with hysteresis to prevent flip-flopping
        refined = base_front.copy()
        rolled = False  # Track if we've already rolled to prevent immediate roll-back
        last_roll_date = None

        for i, d in enumerate(curve.index[1:], 1):
            m0 = M0.at[d]
            m1 = M1.at[d]

            if not m0 or not m1:
                continue

            # Check if we should roll
            if roll_confirmed.at[d]:
                # Only roll if we haven't rolled recently (hysteresis)
                if last_roll_date is None or (d - last_roll_date).days > 5:
                    refined.at[d] = m1
                    rolled = True
                    last_roll_date = d

                    # FIX 5: Forward fill the roll for at least a few days
                    # This prevents single-day reversions
                    future_dates = curve.index[i:min(i + 3, len(curve.index))]
                    for future_d in future_dates:
                        if self.dte.at[future_d, m1] >= 0:  # Only if M1 hasn't expired
                            refined.at[future_d] = m1

            # If we've rolled and M1 becomes the natural front month, maintain it
            elif rolled and base_front.at[d] == m1:
                refined.at[d] = m1

        # Forward fill and handle NaNs
        refined = refined.ffill().where(base_front.notna(), base_front)

        return refined.rename("front_month")

    # Additional helper method to diagnose roll issues
    def diagnose_rolls(self, front_series: pd.Series) -> pd.DataFrame:
        """
        Diagnose potential roll issues by identifying single-day changes.
        Returns DataFrame with dates where front month changes for only 1 day.
        """
        if front_series is None or len(front_series) < 3:
            return pd.DataFrame()

        issues = []

        for i in range(1, len(front_series) - 1):
            prev_val = front_series.iloc[i - 1]
            curr_val = front_series.iloc[i]
            next_val = front_series.iloc[i + 1]

            # Check for single-day change (flip-flop pattern)
            if (prev_val != curr_val) and (curr_val != next_val) and (prev_val == next_val):
                issues.append({
                    'date': front_series.index[i],
                    'prev_contract': prev_val,
                    'single_day_contract': curr_val,
                    'next_contract': next_val
                })

        return pd.DataFrame(issues)

    # Alternative sequentialize method with smoothing
    def sequentialize(
            self,
            fm_labels: pd.Series,
            enforce_calendar_order: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Enhanced sequencing that maintains proper calendar order and handles volume/OI data.
        Prevents rolled contracts from appearing in wrong positions.
        """
        curve = self.curve
        if curve is None:
            raise RuntimeError("Curve not built. Call build_curve() first.")
        if self.dte is None:
            self.compute_dte_matrix(curve)

        # Cap sequential columns at reasonable limit to prevent excessive memory usage
        max_cols = min(curve.shape[1], 48)  # Cap at 48 sequential contracts (4 years)
        mcols = [f"M{i}" for i in range(max_cols)]
        seq_prices = pd.DataFrame(index=curve.index, columns=mcols, dtype="float")
        seq_labels = pd.DataFrame(index=curve.index, columns=mcols, dtype="object")
        seq_dte = pd.DataFrame(index=curve.index, columns=mcols, dtype="float")

        # Initialize volume and OI DataFrames if data is available
        seq_volume = None
        seq_oi = None
        if self.volume_curve is not None:
            seq_volume = pd.DataFrame(index=curve.index, columns=mcols, dtype="float")
        if self.oi_curve is not None:
            seq_oi = pd.DataFrame(index=curve.index, columns=mcols, dtype="float")

        def get_calendar_position(month_code):
            if month_code in ORDER_LETTERS:
                return ORDER_LETTERS.index(month_code)
            return 999

        for date_idx, d in enumerate(curve.index):
            # Collect available contracts
            available_contracts = []

            for month_code in curve.columns:
                px = curve.at[d, month_code]
                dd = self.dte.at[d, month_code]

                if pd.notna(px) and pd.notna(dd) and dd >= -5:
                    available_contracts.append({
                        'code': month_code,
                        'price': px,
                        'dte': dd,
                        'cal_pos': get_calendar_position(month_code),
                        'is_expired': dd < 0
                    })

            if not available_contracts:
                continue

            # Get front month designation
            front_month = None
            if fm_labels is not None and d in fm_labels.index:
                front_month = fm_labels.at[d]
                if pd.isna(front_month):
                    front_month = None

            # Separate and organize contracts
            ordered_sequence = []

            # Find front month contract
            front_contract = None
            other_contracts = []

            for contract in available_contracts:
                if not contract['is_expired']:
                    if contract['code'] == front_month:
                        front_contract = contract
                    else:
                        other_contracts.append(contract)

            if front_contract:
                ordered_sequence.append(front_contract)

                if enforce_calendar_order:
                    # Sort remaining contracts by calendar position
                    # Handle year wrap-around
                    front_pos = front_contract['cal_pos']

                    for contract in other_contracts:
                        if contract['cal_pos'] > front_pos:
                            contract['sort_key'] = (0, contract['cal_pos'])
                        else:
                            contract['sort_key'] = (1, contract['cal_pos'])

                    other_contracts.sort(key=lambda x: x['sort_key'])
                    ordered_sequence.extend(other_contracts)
                else:
                    other_contracts.sort(key=lambda x: x['dte'])
                    ordered_sequence.extend(other_contracts)
            else:
                # No front month specified, use DTE ordering
                other_contracts.sort(key=lambda x: x['dte'])
                ordered_sequence = other_contracts

            # Fill arrays
            for i, contract in enumerate(ordered_sequence[:max_cols]):
                seq_prices.iat[date_idx, i] = contract['price']
                seq_labels.iat[date_idx, i] = contract['code']
                seq_dte.iat[date_idx, i] = contract['dte']

                # Fill volume data if available
                if seq_volume is not None and self.volume_curve is not None:
                    vol_value = self.volume_curve.at[d, contract['code']] if contract[
                                                                                 'code'] in self.volume_curve.columns else np.nan
                    if pd.notna(vol_value):
                        seq_volume.iat[date_idx, i] = vol_value

                # Fill OI data if available
                if seq_oi is not None and self.oi_curve is not None:
                    oi_value = self.oi_curve.at[d, contract['code']] if contract[
                                                                            'code'] in self.oi_curve.columns else np.nan
                    if pd.notna(oi_value):
                        seq_oi.iat[date_idx, i] = oi_value

        # Store volume and OI sequences as instance variables
        if seq_volume is not None:
            self.seq_volume = seq_volume
        if seq_oi is not None:
            self.seq_oi = seq_oi

        return seq_prices, seq_labels, seq_dte

    # ------------------------------
    # Spreads and persistence
    # ------------------------------

    @staticmethod
    def spreads_vs_front(seq_prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate spreads relative to front month (M0) prices."""
        if seq_prices is None or seq_prices.empty:
            return None
        if "M0" not in seq_prices.columns:
            print(f"Warning: M0 column not found in sequential prices. Columns: {list(seq_prices.columns)}")
            return None
        return seq_prices.sub(seq_prices["M0"], axis=0)

    def save_to_hdf(self):
        if self.curve is None:
            raise RuntimeError("Nothing to save; curve is None.")

        # Use DataClient for consistent path structure and proper writing
        try:
            from ..data_client import DataClient
            client = DataClient(market_path=self.hdf_path)

            # Ensure symbol has _F suffix for consistency
            symbol_key = f"{self.symbol}_F" if not self.symbol.endswith('_F') else self.symbol

            # Write curve data using DataClient
            client.write_market(self.curve, f"market/{symbol_key}/curve")

            # Write volume and OI curves if available
            if self.volume_curve is not None:
                client.write_market(self.volume_curve, f"market/{symbol_key}/volume_curve")

            if self.oi_curve is not None:
                client.write_market(self.oi_curve, f"market/{symbol_key}/oi_curve")
            if self.front is not None:
                client.write_market(self.front.to_frame(), f"market/{symbol_key}/front_month")

            if self.dte is not None:
                client.write_market(self.dte, f"market/{symbol_key}/days_to_expiry")

            if self.seq_prices is not None:
                client.write_market(self.seq_prices, f"market/{symbol_key}/curve_seq")

            if self.seq_volume is not None:
                client.write_market(self.seq_volume, f"market/{symbol_key}/volume_seq")

            if self.seq_oi is not None:
                client.write_market(self.seq_oi, f"market/{symbol_key}/oi_seq")

            if self.seq_labels is not None:
                client.write_market(self.seq_labels, f"market/{symbol_key}/curve_seq_labels")

            if self.seq_dte is not None:
                client.write_market(self.seq_dte, f"market/{symbol_key}/days_to_expiry_seq")

            if self.seq_prices is not None:
                spreads_df = self.spreads_vs_front(self.seq_prices)
                client.write_market(spreads_df, f"market/{symbol_key}/spreads_seq")

        except ImportError:
            # Fallback to direct HDF5 writing with corrected paths
            symbol_key = f"{self.symbol}_F" if not self.symbol.endswith('_F') else self.symbol

            with pd.HDFStore(self.hdf_path) as store:
                store.put(f"market/{symbol_key}/curve", self.curve, format="table", data_columns=True)

                # Write volume and OI curves if available
                if self.volume_curve is not None:
                    store.put(f"market/{symbol_key}/volume_curve", self.volume_curve, format="table", data_columns=True)

                if self.oi_curve is not None:
                    store.put(f"market/{symbol_key}/oi_curve", self.oi_curve, format="table", data_columns=True)

                if self.front is not None:
                    store.put(f"market/{symbol_key}/front_month", self.front.to_frame(), format="table",
                              data_columns=True)
                if self.dte is not None:
                    store.put(f"market/{symbol_key}/days_to_expiry", self.dte, format="table", data_columns=True)
                if self.seq_prices is not None:
                    store.put(f"market/{symbol_key}/curve_seq", self.seq_prices, format="table", data_columns=True)
                if self.seq_volume is not None:
                    store.put(f"market/{symbol_key}/volume_seq", self.seq_volume, format="table", data_columns=True)
                if self.seq_oi is not None:
                    store.put(f"market/{symbol_key}/oi_seq", self.seq_oi, format="table", data_columns=True)
                if self.seq_labels is not None:
                    store.put(f"market/{symbol_key}/curve_seq_labels", self.seq_labels, format="table",
                              data_columns=True)
                if self.seq_dte is not None:
                    store.put(f"market/{symbol_key}/days_to_expiry_seq", self.seq_dte, format="table",
                              data_columns=True)
                if self.seq_prices is not None:
                    store.put(f"market/{symbol_key}/spreads_seq", self.spreads_vs_front(self.seq_prices),
                              format="table", data_columns=True)

    # ------------------------------
    # Debugging and diagnostics
    # ------------------------------

    def diagnose_data_state(self) -> Dict[str, any]:
        """Diagnose the current state of curve data for debugging."""
        diagnosis = {
            'curve_exists': self.curve is not None,
            'curve_shape': self.curve.shape if self.curve is not None else None,
            'curve_columns': list(self.curve.columns) if self.curve is not None else None,
            'dte_exists': self.dte is not None,
            'dte_shape': self.dte.shape if self.dte is not None else None,
            'front_exists': self.front is not None,
            'front_length': len(self.front) if self.front is not None else None,
            'front_non_null': len(self.front.dropna()) if self.front is not None else None,
            'seq_prices_exists': self.seq_prices is not None,
            'seq_prices_shape': self.seq_prices.shape if self.seq_prices is not None else None,
            'seq_prices_non_null': self.seq_prices.count().sum() if self.seq_prices is not None else None,
            'seq_spreads_exists': self.seq_spreads is not None,
            'seq_spreads_shape': self.seq_spreads.shape if self.seq_spreads is not None else None,
            'seq_spreads_non_null': self.seq_spreads.count().sum() if self.seq_spreads is not None else None
        }

        if self.curve is not None:
            # Sample of data availability per column
            diagnosis['curve_data_sample'] = {}
            for col in self.curve.columns[:5]:  # First 5 columns
                non_null = self.curve[col].count()
                total = len(self.curve[col])
                diagnosis['curve_data_sample'][col] = f"{non_null}/{total}"

        return diagnosis

    def detect_sequencing_issues(self) -> pd.DataFrame:
        """Detect sequencing issues in forward curve."""
        issues = []

        seq_dte, seq_spreads, seq_labels = self.dte, self.seq_spreads, self.seq_labels,

        for idx, date in enumerate(seq_labels.index):
            row_labels = seq_labels.iloc[idx]
            row_dte = seq_dte.iloc[idx]

            active_codes = [c for c in row_labels if pd.notna(c)]
            active_dtes = [row_dte[f'M{i}'] for i, c in enumerate(row_labels) if pd.notna(c)]

            if len(active_codes) < 2:
                continue

            # Check calendar order violations
            for i in range(len(active_codes) - 1):
                curr_code = active_codes[i]
                next_code = active_codes[i + 1]

                curr_pos = ORDER_LETTERS.index(curr_code) if curr_code in ORDER_LETTERS else -1
                next_pos = ORDER_LETTERS.index(next_code) if next_code in ORDER_LETTERS else -1

                if curr_pos >= 0 and next_pos >= 0:
                    if next_pos < curr_pos and not (curr_pos >= 9 and next_pos <= 2):
                        issues.append({
                            'date': date,
                            'issue_type': 'calendar_order_violation',
                            'M_index': i,
                            'current': curr_code,
                            'next': next_code
                        })

            # Check DTE monotonicity
            for i in range(len(active_dtes) - 1):
                if pd.notna(active_dtes[i]) and pd.notna(active_dtes[i + 1]):
                    if active_dtes[i + 1] < active_dtes[i]:
                        issues.append({
                            'date': date,
                            'issue_type': 'dte_not_monotonic',
                            'M_index': i,
                            'current_dte': active_dtes[i],
                            'next_dte': active_dtes[i + 1]
                        })

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
            min_persistence_days: int = 2,
            smooth_window: int = 3,
            enforce_calendar_order: bool = True,
            save: bool = True,
            debug: bool = False
    ):
        """
        Enhanced run method with all fixes applied.
        """
        print(f"Building forward curve for {self.symbol} with fixes applied...")

        # Build curve and compute DTE
        self.build_curve()
        self.compute_dte_matrix(self.curve)

        # Front-month selection with fixed roll detection
        front_series = self.load_front_month_series() if prefer_front_series else None

        if front_series is not None:
            fm0 = self.assign_front_by_match(front_series, tol=match_tol)
            cal = self.assign_front_calendar(self.curve)
            fm0 = fm0.where(fm0.notna(), cal)
            fm = self.refine_front_by_jump(
                fm0,
                rel_jump_thresh=rel_jump_thresh,
                robust_k=robust_k,
                lookback=lookback,
                near_expiry_days=near_expiry_days,
                min_persistence_days=min_persistence_days,
                smooth_window=smooth_window
            )
        else:
            cal = self.assign_front_calendar(self.curve)
            fm = self.refine_front_by_jump(
                cal,
                rel_jump_thresh=rel_jump_thresh,
                robust_k=robust_k,
                lookback=lookback,
                near_expiry_days=near_expiry_days,
                min_persistence_days=min_persistence_days,
                smooth_window=smooth_window
            )

        self.front = fm

        # Check for roll issues
        roll_issues = self.diagnose_rolls(fm)
        if not roll_issues.empty and debug:
            print(f"WARNING: Found {len(roll_issues)} potential single-day roll issues")
            if debug:
                print(roll_issues.head())

        # Sequentialize with fixed ordering
        seq_prices, seq_labels, seq_dte = self.sequentialize(
            self.front,
            enforce_calendar_order=enforce_calendar_order
        )

        self.seq_prices = seq_prices
        self.seq_labels = seq_labels
        self.seq_dte = seq_dte

        # Calculate spreads
        if self.seq_prices is not None and not self.seq_prices.empty:
            self.seq_spreads = self.spreads_vs_front(self.seq_prices)

            if debug:
                # Validate the sequencing
                issues = self.detect_sequencing_issues()
                if not issues.empty:
                    print(f"WARNING: Found {len(issues)} sequencing issues")
                    print(issues.head())

        if save:
            self.save_to_hdf()

        # Return paths
        symbol_key = f"{self.symbol}_F" if not self.symbol.endswith('_F') else self.symbol

        # Return paths for all data including volume and OI
        results = {
            "curve": f"market/{symbol_key}/curve",
            "front": f"market/{symbol_key}/front_month",
            "dte": f"market/{symbol_key}/days_to_expiry",
            "seq_curve": f"market/{symbol_key}/curve_seq",
            "seq_labels": f"market/{symbol_key}/curve_seq_labels",
            "seq_dte": f"market/{symbol_key}/days_to_expiry_seq",
            "seq_spreads": f"market/{symbol_key}/spreads_seq",
        }

        # Add volume and OI paths if data exists
        if self.volume_curve is not None:
            results["volume_curve"] = f"market/{symbol_key}/volume_curve"
        if self.oi_curve is not None:
            results["oi_curve"] = f"market/{symbol_key}/oi_curve"
        if self.seq_volume is not None:
            results["seq_volume"] = f"market/{symbol_key}/volume_seq"
        if self.seq_oi is not None:
            results["seq_oi"] = f"market/{symbol_key}/oi_seq"

        return results


def _is_empty(data) -> bool:
    """Check if data structure is empty (works for both DataFrames and numpy arrays)"""
    if hasattr(data, 'empty'):
        return data.empty
    elif hasattr(data, 'size'):
        return data.size == 0
    elif hasattr(data, '__len__'):
        return len(data) == 0
    else:
        return data is None


def _safe_get_value(data, row_idx, col_key, default=np.nan):
    """Safely get value from data structure (DataFrame or array)"""
    try:
        if hasattr(data, 'index'):
            return data.loc[row_idx, col_key]
        elif hasattr(data, '__getitem__'):
            return data[row_idx, col_key] if hasattr(data, 'ndim') and data.ndim > 1 else data[row_idx]
        else:
            return default
    except (KeyError, IndexError, TypeError):
        return default


def _safe_get_columns(data):
    """Safely get columns from data structure"""
    if hasattr(data, 'columns'):
        return data.columns
    elif hasattr(data, 'shape') and len(data.shape) > 1:
        return list(range(data.shape[1]))
    else:
        return []


def _safe_get_index(data):
    """Safely get index from data structure"""
    if hasattr(data, 'index'):
        return data.index
    elif hasattr(data, 'shape'):
        return list(range(data.shape[0]))
    else:
        return []


def _safe_check_column(data, col_key) -> bool:
    """Check if column exists in data structure"""
    if hasattr(data, 'columns'):
        return col_key in data.columns
    elif hasattr(data, 'shape') and len(data.shape) > 1:
        return isinstance(col_key, int) and 0 <= col_key < data.shape[1]
    else:
        return False


@dataclass
class ExpiryTracker:
    """
    Tracks expiry dates and rolls for futures contracts
    """
    symbol: str
    month_code: str
    year: int
    expiry_date: Optional[datetime] = None
    days_to_expiry: Optional[int] = None
    is_active: bool = True
    roll_date: Optional[datetime] = None

    def __post_init__(self):
        self._base_month_code = normalize_contract_code(self.month_code) or self.month_code

        inferred_year = infer_contract_year(self.month_code, self.year)
        if inferred_year is not None:
            self.year = inferred_year

        if self.expiry_date is None:
            self.expiry_date = self._calculate_expiry_date()

        if self.days_to_expiry is None and self.expiry_date:
            self.days_to_expiry = (self.expiry_date - datetime.now()).days

    def _calculate_expiry_date(self) -> datetime:
        """Calculate expiry date based on standard rules"""
        month_key = normalize_contract_code(self.month_code)
        month_num = MONTH_CODE_MAP.get(month_key, 1)
        year = self.year

        inferred_year = infer_contract_year(self.month_code, self.year)
        if inferred_year is not None:
            year = inferred_year

        # Default expiry rules (can be customized per symbol)
        expiry_rules = {
            'CL': {'day': 25, 'offset': -3},  # Crude: 3 business days before 25th
            'NG': {'day': -3, 'offset': 0},  # Natural Gas: 3 business days before last
            'C': {'day': 15, 'offset': -1},  # Corn: business day prior to 15th
            'S': {'day': 15, 'offset': -1},  # Soybeans
            'W': {'day': 15, 'offset': -1},  # Wheat
        }

        # Extract symbol code from symbol
        commodity = ''.join([c for c in self.symbol if c.isalpha()])[:2]

        if commodity in expiry_rules:
            rule = expiry_rules[commodity]
            if rule['day'] > 0:
                exp_date = datetime(year, month_num, rule['day'])
            else:
                last_day = calendar.monthrange(year, month_num)[1]
                exp_date = datetime(year, month_num, last_day) + timedelta(days=rule['day'])

            # Apply business day offset
            if rule['offset'] != 0:
                exp_date = self._add_business_days(exp_date, rule['offset'])

            return exp_date

        # Default: 15th of the month
        return datetime(year, month_num, 15)

    def _add_business_days(self, date: datetime, days: int) -> datetime:
        """Add business days to a date"""
        delta = abs(days)
        direction = 1 if days > 0 else -1

        while delta > 0:
            date += timedelta(days=direction)
            if date.weekday() < 5:  # Monday = 0, Friday = 4
                delta -= 1

        return date


@dataclass
class SpreadFeature:
    dtype: type = float
    sequential: bool = False
    data: np.ndarray = None
    labels: List[str] = None
    direction: str = None
    index: pd.Index = None
    frame: Optional[pd.DataFrame] = None

    def __post_init__(self):
        # Preserve an associated DataFrame when provided so utilities that
        # expect pandas objects can retrieve the raw structure without
        # repeated conversions.
        if isinstance(self.data, pd.DataFrame):
            self.frame = self.data.copy()
            self.index = self.frame.index if self.index is None else self.index
            self.labels = list(self.frame.columns) if self.labels is None else self.labels
            self.data = self.frame.to_numpy(dtype=float)
            return

        if self.frame is not None:
            if not isinstance(self.frame, pd.DataFrame):
                raise TypeError("frame must be a pandas DataFrame when provided")
            if self.data is None:
                self.data = self.frame.to_numpy(dtype=float)
            self.index = self.frame.index if self.index is None else self.index
            self.labels = list(self.frame.columns) if self.labels is None else self.labels

        if self.data is not None and not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=float)

    def _calculate_slopes(self) -> np.ndarray:
        """
        Calculate slopes for all curves simultaneously
        For horizontal sequential data, calculates slope across contracts for each time point
        """
        if self.data is None:
            return np.array([])

        # Handle 1D data
        if len(self.data.shape) == 1:
            if len(self.data) >= 2:
                return np.array([np.polyfit(range(len(self.data)), self.data, 1)[0]])
            else:
                return np.array([])

        # Handle 2D data
        if len(self.data.shape) == 2:
            if self.direction == "horizontal" and self.sequential:
                # For horizontal sequential data: calculate slope across contracts for each time point
                slopes = []
                for row in self.data:
                    valid_mask = ~np.isnan(row)
                    if np.sum(valid_mask) >= 2:  # Need at least 2 points for slope
                        valid_prices = row[valid_mask]
                        valid_positions = np.arange(len(row))[valid_mask]
                        slope = np.polyfit(valid_positions, valid_prices, 1)[0]
                        slopes.append(slope)
                    else:
                        slopes.append(np.nan)
                return np.array(slopes)

            elif self.direction == "horizontal":
                # For horizontal non-sequential data: calculate slope across contracts for each time point
                slopes = []
                for row in self.data:
                    valid_mask = ~np.isnan(row)
                    if np.sum(valid_mask) >= 2:
                        valid_prices = row[valid_mask]
                        slope = np.polyfit(range(len(valid_prices)), valid_prices, 1)[0]
                        slopes.append(slope)
                    else:
                        slopes.append(np.nan)
                return np.array(slopes)

            else:
                # For vertical data: calculate slopes over time for each contract
                slopes = []
                for col in self.data.T:
                    valid_mask = ~np.isnan(col)
                    if np.sum(valid_mask) >= 2:
                        valid_prices = col[valid_mask]
                        slope = np.polyfit(range(len(valid_prices)), valid_prices, 1)[0]
                        slopes.append(slope)
                    else:
                        slopes.append(np.nan)
                return np.array(slopes)

        return np.array([])

    def get_slopes(self) -> np.ndarray:
        """
        Public method to get slopes for horizontal sequential SpreadFeature
        Returns slopes representing curve shape at each time point
        """
        if self.direction == "horizontal" and self.sequential:
            return self._calculate_slopes()
        else:
            raise ValueError(f"get_slopes() is only available for horizontal sequential SpreadFeatures. "
                             f"Current: direction='{self.direction}', sequential={self.sequential}")

    def get_market_structure(self) -> List[str]:
        """
        Get market structure interpretation for each time point
        Returns list of strings: 'Contango', 'Backwardation', or 'Flat'
        """
        if not (self.direction == "horizontal" and self.sequential):
            raise ValueError("get_market_structure() is only available for horizontal sequential SpreadFeatures")

        slopes = self.get_slopes()
        structures = []

        for slope in slopes:
            if np.isnan(slope):
                structures.append('Unknown')
            elif slope > 0.01:  # Small threshold to avoid noise
                structures.append('Contango')
            elif slope < -0.01:
                structures.append('Backwardation')
            else:
                structures.append('Flat')

        return structures

    def calculate_moving_average(self, window: int = 20) -> np.ndarray:
        """
        Calculate moving average for vertical SpreadFeature (along time axis)
        Only works on sequentialized data, not raw curve data
        """
        if self.direction != "vertical":
            raise ValueError("calculate_moving_average() is only available for vertical SpreadFeatures")

        if self.data is None:
            return np.array([])

        # For 1D data (single contract time series)
        if len(self.data.shape) == 1:
            return self._rolling_window_calc(self.data, window, np.nanmean)

        # For 2D data (multiple contract time series)
        if len(self.data.shape) == 2:
            # Apply moving average to each contract (column) separately
            ma_data = np.zeros_like(self.data)
            for i in range(self.data.shape[1]):
                ma_data[:, i] = self._rolling_window_calc(self.data[:, i], window, np.nanmean)
            return ma_data

        return np.array([])

    def calculate_volatility(self, window: int = 20) -> np.ndarray:
        """
        Calculate rolling volatility for vertical SpreadFeature (along time axis)
        Only works on sequentialized data
        """
        if self.direction != "vertical":
            raise ValueError("calculate_volatility() is only available for vertical SpreadFeatures")

        if self.data is None:
            return np.array([])

        # For 1D data (single contract time series)
        if len(self.data.shape) == 1:
            returns = np.diff(self.data) / self.data[:-1]
            padded_returns = np.concatenate([[np.nan], returns])
            return self._rolling_window_calc(padded_returns, window, np.nanstd)

        # For 2D data (multiple contract time series)
        if len(self.data.shape) == 2:
            vol_data = np.zeros_like(self.data)
            for i in range(self.data.shape[1]):
                contract_data = self.data[:, i]
                returns = np.diff(contract_data) / contract_data[:-1]
                padded_returns = np.concatenate([[np.nan], returns])
                vol_data[:, i] = self._rolling_window_calc(padded_returns, window, np.nanstd)
            return vol_data

        return np.array([])

    def _rolling_window_calc(self, data: np.ndarray, window: int, func) -> np.ndarray:
        """Helper function to calculate rolling window statistics"""
        result = np.full_like(data, np.nan)

        for i in range(window - 1, len(data)):
            window_data = data[i - window + 1:i + 1]
            if not np.all(np.isnan(window_data)):
                result[i] = func(window_data)

        return result

    def analyze_trend(self, method: str = 'linear') -> Dict[str, Any]:
        """
        Analyze trend for vertical SpreadFeature using sequentialized data
        """
        if self.direction != "vertical":
            raise ValueError("analyze_trend() is only available for vertical SpreadFeatures")

        if self.data is None or len(self.data) < 3:
            return {'trend': 'insufficient_data', 'slope': np.nan, 'r_squared': np.nan}

        results = {}

        # For 1D data (single contract)
        if len(self.data.shape) == 1:
            valid_mask = ~np.isnan(self.data)
            if np.sum(valid_mask) >= 3:
                x = np.arange(len(self.data))[valid_mask]
                y = self.data[valid_mask]

                if method == 'linear':
                    coeffs = np.polyfit(x, y, 1)
                    slope, intercept = coeffs
                    y_pred = np.polyval(coeffs, x)
                    r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

                    results = {
                        'trend': 'upward' if slope > 0 else 'downward' if slope < 0 else 'flat',
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_squared
                    }

        # For 2D data (multiple contracts)
        elif len(self.data.shape) == 2:
            contract_results = {}
            for i in range(self.data.shape[1]):
                contract_data = self.data[:, i]
                valid_mask = ~np.isnan(contract_data)

                if np.sum(valid_mask) >= 3:
                    x = np.arange(len(contract_data))[valid_mask]
                    y = contract_data[valid_mask]

                    if method == 'linear':
                        coeffs = np.polyfit(x, y, 1)
                        slope, intercept = coeffs
                        y_pred = np.polyval(coeffs, x)
                        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

                        contract_results[f'contract_{i}'] = {
                            'trend': 'upward' if slope > 0 else 'downward' if slope < 0 else 'flat',
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_squared
                        }

            results = contract_results

        return results


# noinspection PyTypeChecker
@dataclass
class SpreadReturns(SpreadFeature):
    long_price: np.array = None
    short_price: np.array = None
    long_returns: np.array = None
    short_returns: np.array = None
    long_volume: np.array = None
    long_ct_oi: np.array = None
    short_volume: np.array = None
    short_ct_oi: np.array = None
    index: pd.Index = None
    rolling_dynamic: bool = True
    long_label: str = None
    short_label: str = None
    
    # PnL calculation parameters
    position_size: float = 1.0
    fees_per_contract: float = 0.0
    roll_yield: Optional[np.array] = None
    roll_dates: Optional[np.array] = None
    contract_multiplier: float = 1.0
    dte: Optional[np.ndarray] = None
    seq_dte: Optional[np.ndarray] = None

    def __post_init__(self):
        self.direction = "vertical"
        
        if not hasattr(self, "long_price") or not hasattr(self, "short_price"):
            raise ValueError("price data needed to calculate returns")

        # Align price arrays
        if len(self.long_price) == len(self.short_price):
            self.absolute_spread = self.long_price - self.short_price
        else:
            if len(self.long_price) > len(self.short_price):
                self.long_price = self.long_price[:len(self.short_price)]
            else:
                self.short_price = self.short_price[:len(self.long_price)]
            self.absolute_spread = self.long_price - self.short_price

        self.returns = self.absolute_spread / self.long_price

        if not hasattr(self, 'index'):
            self.index = pd.Index(np.arange(0, len(
                min([len(self.long_price), len(self.short_price)])
            )))
        
        # Initialize PnL tracking
        self._initialize_pnl_tracking()

    def _initialize_pnl_tracking(self):
        """Initialize PnL tracking arrays"""
        n_periods = len(self.index)
        
        # Initialize PnL components
        self.daily_pnl = np.zeros(n_periods)
        self.cumulative_pnl = np.zeros(n_periods)
        self.roll_costs = np.zeros(n_periods)
        self.transaction_costs = np.zeros(n_periods)
        self.gross_pnl = np.zeros(n_periods)
        self.net_pnl = np.zeros(n_periods)
    
    def calculate_spread_pnl(self, 
                           entry_date: Optional[int] = None,
                           exit_date: Optional[int] = None,
                           include_roll_costs: bool = True,
                           include_fees: bool = True) -> Dict[str, Any]:
        """
        Calculate comprehensive PnL from spread position
        
        Parameters:
        - entry_date: Index position for entry (None = start)
        - exit_date: Index position for exit (None = end)  
        - include_roll_costs: Whether to include roll yield/costs
        - include_fees: Whether to include transaction fees
        
        Returns:
        - Dictionary with PnL breakdown and metrics
        """
        start_idx = entry_date if entry_date is not None else 0
        end_idx = exit_date if exit_date is not None else len(self.index) - 1
        
        if start_idx >= end_idx or end_idx >= len(self.index):
            raise ValueError(f"Invalid date range: start={start_idx}, end={end_idx}")
        
        # Calculate raw spread PnL
        entry_spread = self.absolute_spread[start_idx]
        exit_spread = self.absolute_spread[end_idx]
        
        # Raw spread change (long spread position)
        spread_change = exit_spread - entry_spread
        gross_pnl = spread_change * self.position_size * self.contract_multiplier
        
        # Calculate daily PnL series
        daily_spread_changes = np.diff(self.absolute_spread[start_idx:end_idx+1])
        daily_gross_pnl = daily_spread_changes * self.position_size * self.contract_multiplier
        
        # Transaction costs
        total_fees = 0.0
        if include_fees and self.fees_per_contract > 0:
            # Entry and exit fees for both legs
            total_fees = 4 * self.fees_per_contract * abs(self.position_size)
        
        # Roll costs calculation
        total_roll_costs = 0.0
        roll_events = 0
        
        if include_roll_costs and self.roll_dates is not None:
            # Handle both DataFrame (new format) and array (legacy format) roll_dates
            if isinstance(self.roll_dates, pd.DataFrame):
                # New DataFrame format with comprehensive roll information
                if hasattr(self, 'index') and isinstance(self.index, pd.DatetimeIndex):
                    # Filter roll events within the date range
                    entry_date = self.index[start_idx] if start_idx < len(self.index) else None
                    exit_date = self.index[end_idx] if end_idx < len(self.index) else None
                    
                    if entry_date and exit_date:
                        roll_events_in_period = self.roll_dates[
                            (self.roll_dates.index >= entry_date) &
                            (self.roll_dates.index <= exit_date)
                        ]
                        
                        for roll_date, roll_info in roll_events_in_period.iterrows():
                            # Calculate roll cost based on days to expiry and confidence
                            base_roll_cost = 0.01  # Base cost per contract (adjustable)
                            
                            # Adjust cost based on confidence (lower confidence = higher cost)
                            confidence_factor = 1.0 / max(roll_info['confidence'], 0.1)
                            
                            # Adjust cost based on timing (earlier rolls may have higher costs)
                            timing_factor = max(1.0, (15 - roll_info['days_to_expiry']) / 10)
                            
                            roll_cost = (base_roll_cost * confidence_factor * timing_factor * 
                                       self.position_size * self.contract_multiplier)
                            total_roll_costs += roll_cost
                            roll_events += 1
                else:
                    # Fallback: use simple roll count if no proper date index
                    roll_events = len(self.roll_dates)
                    total_roll_costs = roll_events * 0.01 * self.position_size * self.contract_multiplier
                    
            else:
                # Legacy array format
                roll_indices = self.roll_dates[(self.roll_dates >= start_idx) & 
                                             (self.roll_dates <= end_idx)]
                
                for roll_idx in roll_indices:
                    if hasattr(self, 'roll_yield') and self.roll_yield is not None and roll_idx < len(self.roll_yield):
                        # Roll cost = position_size * roll_yield * contract_multiplier
                        roll_cost = (self.position_size * self.roll_yield[int(roll_idx)] * 
                                   self.contract_multiplier)
                        total_roll_costs += roll_cost
                        roll_events += 1
                    else:
                        # Default roll cost if no roll_yield available
                        total_roll_costs += 0.01 * self.position_size * self.contract_multiplier
                        roll_events += 1
        
        # Net PnL calculation
        net_pnl = gross_pnl - total_fees - total_roll_costs
        
        # Performance metrics
        holding_period = end_idx - start_idx
        if entry_spread != 0:
            total_return = net_pnl / (abs(entry_spread) * self.position_size * self.contract_multiplier)
        else:
            total_return = 0.0
        
        # Annualized return (assuming daily data)
        if holding_period > 0:
            annualized_return = total_return * (252 / holding_period)
        else:
            annualized_return = 0.0
        
        # Calculate volatility of daily PnL
        pnl_volatility = np.std(daily_gross_pnl) if len(daily_gross_pnl) > 1 else 0.0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if pnl_volatility > 0:
            sharpe_ratio = np.mean(daily_gross_pnl) / pnl_volatility * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative_pnl = np.cumsum(daily_gross_pnl)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            'gross_pnl': gross_pnl,
            'transaction_costs': total_fees,
            'roll_costs': total_roll_costs,
            'net_pnl': net_pnl,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'holding_period_days': holding_period,
            'entry_spread': entry_spread,
            'exit_spread': exit_spread,
            'spread_change': spread_change,
            'roll_events': roll_events,
            'daily_pnl': daily_gross_pnl,
            'cumulative_pnl': cumulative_pnl,
            'pnl_volatility': pnl_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'position_size': self.position_size,
            'contract_multiplier': self.contract_multiplier
        }
    
    def calculate_rolling_pnl(self, 
                            window: int = 20,
                            step: int = 1) -> pd.DataFrame:
        """
        Calculate rolling PnL metrics over specified windows
        
        Parameters:
        - window: Rolling window size in periods
        - step: Step size between calculations
        
        Returns:
        - DataFrame with rolling PnL metrics
        """
        results = []
        
        for start_idx in range(0, len(self.index) - window, step):
            end_idx = start_idx + window - 1
            
            try:
                pnl_metrics = self.calculate_spread_pnl(start_idx, end_idx)
                
                result = {
                    'start_date': self.index[start_idx],
                    'end_date': self.index[end_idx],
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'net_pnl': pnl_metrics['net_pnl'],
                    'total_return': pnl_metrics['total_return'],
                    'annualized_return': pnl_metrics['annualized_return'],
                    'sharpe_ratio': pnl_metrics['sharpe_ratio'],
                    'max_drawdown': pnl_metrics['max_drawdown'],
                    'pnl_volatility': pnl_metrics['pnl_volatility'],
                    'roll_events': pnl_metrics['roll_events']
                }
                results.append(result)
                
            except Exception as e:
                # Skip problematic windows
                continue
        
        return pd.DataFrame(results)
    
    def set_roll_schedule(self, 
                         roll_dates: np.array,
                         roll_yields: np.array):
        """
        Set roll schedule and yields for PnL calculation
        
        Parameters:
        - roll_dates: Array of indices where rolls occur
        - roll_yields: Array of roll yields/costs at each roll date
        """
        if len(roll_dates) != len(roll_yields):
            raise ValueError("roll_dates and roll_yields must have same length")
        
        self.roll_dates = np.array(roll_dates, dtype=int)
        self.roll_yield = np.array(roll_yields, dtype=float)
    
    def get_pnl_summary(self) -> Dict[str, float]:
        """Get comprehensive PnL summary for entire period"""
        return self.calculate_spread_pnl()
    
    def load_roll_dates_from_symbol(self, symbol: str):
        """
        Load roll dates DataFrame from DataClient for the given symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to load roll dates for (e.g., 'CL', 'NG')
        """
        try:
            from ..data_client import DataClient
            client = DataClient()
            self.roll_dates = client.read_roll_dates(symbol)
        except KeyError:
            # Roll dates not available
            self.roll_dates = None
        except Exception as e:
            print(f"Warning: Could not load roll dates for {symbol}: {e}")
            self.roll_dates = None
    
    def get_roll_adjusted_pnl(self, 
                            entry_date: Optional[int] = None,
                            exit_date: Optional[int] = None,
                            roll_cost_factor: float = 1.0) -> Dict[str, Any]:
        """
        Calculate PnL with enhanced roll cost modeling using roll_dates DataFrame.
        
        Parameters:
        -----------
        entry_date : int, optional
            Index position for entry (None = start)
        exit_date : int, optional
            Index position for exit (None = end)
        roll_cost_factor : float
            Multiplier for roll costs (default 1.0)
            
        Returns:
        --------
        Dict[str, Any]
            Enhanced PnL breakdown with detailed roll cost analysis
        """
        # Calculate base PnL with enhanced roll costs
        base_pnl = self.calculate_spread_pnl(
            entry_date=entry_date, 
            exit_date=exit_date,
            include_roll_costs=True,
            include_fees=True
        )
        
        # Add enhanced roll analysis if DataFrame available
        if isinstance(self.roll_dates, pd.DataFrame):
            start_idx = entry_date if entry_date is not None else 0
            end_idx = exit_date if exit_date is not None else len(self.index) - 1
            
            if hasattr(self, 'index') and isinstance(self.index, pd.DatetimeIndex):
                entry_date_ts = self.index[start_idx] if start_idx < len(self.index) else None
                exit_date_ts = self.index[end_idx] if end_idx < len(self.index) else None
                
                if entry_date_ts and exit_date_ts:
                    roll_events = self.roll_dates[
                        (self.roll_dates.index >= entry_date_ts) &
                        (self.roll_dates.index <= exit_date_ts)
                    ]
                    
                    # Enhanced roll analysis
                    roll_analysis = {
                        'total_roll_events': len(roll_events),
                        'avg_days_to_expiry': roll_events['days_to_expiry'].mean() if len(roll_events) > 0 else 0,
                        'avg_confidence': roll_events['confidence'].mean() if len(roll_events) > 0 else 0,
                        'low_confidence_rolls': len(roll_events[roll_events['confidence'] < 0.7]),
                        'early_rolls': len(roll_events[roll_events['days_to_expiry'] > 10]),
                        'roll_patterns': roll_events.groupby(['from_contract_expiration_code', 'to_contract_expiration_code']).size().to_dict() if len(roll_events) > 0 else {},
                        'roll_dates_list': roll_events.index.tolist()
                    }
                    
                    # Adjust total roll costs by factor
                    adjusted_roll_costs = base_pnl['roll_costs'] * roll_cost_factor
                    adjusted_net_pnl = base_pnl['gross_pnl'] - base_pnl['transaction_costs'] - adjusted_roll_costs
                    
                    base_pnl.update({
                        'adjusted_roll_costs': adjusted_roll_costs,
                        'adjusted_net_pnl': adjusted_net_pnl,
                        'roll_cost_factor': roll_cost_factor,
                        'roll_analysis': roll_analysis
                    })
        
        return base_pnl
    
    def analyze_roll_impact(self) -> Optional[Dict[str, Any]]:
        """
        Analyze the impact of roll events on spread returns.
        
        Returns:
        --------
        Dict[str, Any] or None
            Analysis of how roll events affected spread performance
        """
        if not isinstance(self.roll_dates, pd.DataFrame) or len(self.roll_dates) == 0:
            return None
        
        if not hasattr(self, 'index') or not isinstance(self.index, pd.DatetimeIndex):
            return None
        
        # Calculate returns around roll dates
        roll_impact_analysis = {
            'total_rolls': len(self.roll_dates),
            'roll_events': []
        }
        
        for roll_date, roll_info in self.roll_dates.iterrows():
            try:
                # Find the closest index to roll date
                roll_idx = self.index.get_loc(roll_date, method='nearest')
                
                # Calculate returns before and after roll (±5 days window)
                pre_roll_window = slice(max(0, roll_idx - 5), roll_idx)
                post_roll_window = slice(roll_idx, min(len(self.index), roll_idx + 6))
                
                pre_roll_return = np.mean(self.returns[pre_roll_window]) if len(self.returns[pre_roll_window]) > 0 else 0
                post_roll_return = np.mean(self.returns[post_roll_window]) if len(self.returns[post_roll_window]) > 0 else 0
                
                roll_event_analysis = {
                    'roll_date': roll_date,
                    'from_contract': roll_info['from_contract_expiration_code'],
                    'to_contract': roll_info['to_contract_expiration_code'],
                    'days_to_expiry': roll_info['days_to_expiry'],
                    'confidence': roll_info['confidence'],
                    'pre_roll_return': pre_roll_return,
                    'post_roll_return': post_roll_return,
                    'return_impact': post_roll_return - pre_roll_return,
                    'spread_level_at_roll': self.absolute_spread[roll_idx] if roll_idx < len(self.absolute_spread) else None
                }
                
                roll_impact_analysis['roll_events'].append(roll_event_analysis)
                
            except (KeyError, IndexError):
                continue
        
        # Aggregate statistics
        if roll_impact_analysis['roll_events']:
            impacts = [event['return_impact'] for event in roll_impact_analysis['roll_events']]
            roll_impact_analysis.update({
                'avg_return_impact': np.mean(impacts),
                'total_return_impact': np.sum(impacts),
                'positive_impact_rolls': len([i for i in impacts if i > 0]),
                'negative_impact_rolls': len([i for i in impacts if i < 0]),
                'max_positive_impact': max(impacts) if impacts else 0,
                'max_negative_impact': min(impacts) if impacts else 0
            })
        
        return roll_impact_analysis


@dataclass
class SeqData:
    """Simple sequential data container."""

    timestamps: Optional[pd.DatetimeIndex] = None
    seq_labels: Optional[Union[pd.DataFrame, List[Any]]] = None
    seq_prices: Optional[SpreadFeature] = None
    seq_spreads: Optional[SpreadFeature] = None
    seq_oi: Optional[SpreadFeature] = None
    seq_volume: Optional[SpreadFeature] = None
    seq_dte: Optional[np.ndarray] = None
    roll_dates: Optional[pd.DataFrame] = None


@dataclass
class FuturesCurve(SpreadFeature):
    """
    Single snapshot of a futures curve at a specific point in time
    """
    symbol: Optional[str] = None
    ref_date: datetime = None
    curve_month_labels: List[str] = field(default_factory=list)
    prices: np.ndarray = None
    volumes: Optional[np.ndarray] = None
    open_interest: Optional[np.ndarray] = None
    sequential: bool = False
    days_to_expiry: Optional[np.ndarray] = None
    seq_prices: Optional[np.ndarray] = None
    seq_labels: Optional[List[str]] = None
    seq_volumes: Optional[np.ndarray] = None
    seq_dte: Optional[np.ndarray] = None
    fm_label: Optional[str] = None

    def __post_init__(self):
        """Validate and process curve data after initialization"""
        if self.prices is None:
            self.prices = np.array([])
        else:
            self.prices = np.asarray(self.prices, dtype=float)

        if self.curve_month_labels is None or len(self.curve_month_labels) == 0:
            self.curve_month_labels = [f'M{i}' for i in range(len(self.prices))]
        elif len(self.curve_month_labels) != len(self.prices):
            labels = list(self.curve_month_labels)
            if len(labels) < len(self.prices):
                labels.extend([f'M{i}' for i in range(len(labels), len(self.prices))])
            self.curve_month_labels = labels[:len(self.prices)]

        if self.volumes is not None:
            self.volumes = np.asarray(self.volumes, dtype=float)
            if len(self.volumes) != len(self.prices):
                raise ValueError("Volumes must have same length as prices")

        if self.open_interest is not None:
            self.open_interest = np.asarray(self.open_interest, dtype=float)
            if len(self.open_interest) != len(self.prices):
                raise ValueError("Open interest must have same length as prices")

        if not self.sequential:
            self.sequence_curve()
            self.sequential = True

        if self.seq_dte is not None:
            seq_dte_array = np.asarray(self.seq_dte, dtype=float)
            if seq_dte_array.ndim > 1:
                seq_dte_array = np.squeeze(seq_dte_array)
            self.seq_dte = seq_dte_array
            self.days_to_expiry = seq_dte_array
        elif self.days_to_expiry is not None:
            self.days_to_expiry = np.asarray(self.days_to_expiry, dtype=float)

        if self.days_to_expiry is None or len(self.days_to_expiry) == 0:
            self.days_to_expiry = self._calculate_days_to_expiry()
        else:
            self.days_to_expiry = np.asarray(self.days_to_expiry, dtype=float)

        self.direction = "horizontal"

    def _calculate_days_to_expiry(self) -> np.ndarray:
        """Calculate days to expiry for each contract."""
        if self.ref_date is None or self.curve_month_labels is None or len(self.curve_month_labels) == 0:
            return np.array([])

        dte_list: List[float] = []
        month_occurrence: Dict[str, int] = {}

        for month_code in self.curve_month_labels:
            base_code = normalize_contract_code(month_code)
            if base_code is None or base_code not in MONTH_CODE_MAP:
                dte_list.append(np.nan)
                continue

            month_num = MONTH_CODE_MAP[base_code]
            inferred_year = infer_contract_year(month_code, self.ref_date.year)

            if inferred_year is not None:
                year = inferred_year
            else:
                occurrence = month_occurrence.get(base_code, 0)
                month_occurrence[base_code] = occurrence + 1

                year = self.ref_date.year
                if month_num < self.ref_date.month or (
                        month_num == self.ref_date.month and self.ref_date.day > 15):
                    year += 1
                year += occurrence

            expiry_date = datetime(year, month_num, 15)

            while expiry_date < self.ref_date:
                year += 1
                expiry_date = datetime(year, month_num, 15)

            days_to_expiry = max(0, (expiry_date - self.ref_date).days)
            dte_list.append(days_to_expiry)

        self.days_to_expiry = np.asarray(dte_list, dtype=float)
        return self.days_to_expiry

    def sequence_curve(self, roll_on: str = 'volume') -> np.ndarray:
        """
        Get prices in sequential order based on roll criteria

        Parameters:
        - roll_on: 'volume', 'oi', or 'calendar' (default order)
        """
        # If seq_prices is provided, use it directly
        if self.seq_prices is not None:
            return np.array([p for p in self.seq_prices if not np.isnan(p)])

        if roll_on == 'calendar':
            # Return prices in calendar month order
            return np.array([p for p in self.prices if not np.isnan(p)])

        elif roll_on == 'volume' and self.volumes is not None and len(self.volumes) > 0:
            # Sort by volume (highest first)
            paired = [(p, v, i) for i, (p, v) in enumerate(zip(self.prices, self.volumes))
                      if not np.isnan(p) and not np.isnan(v)]
            paired.sort(key=lambda x: x[1], reverse=True)
            return np.array([p for p, _, _ in paired])

        elif roll_on == 'oi' and self.open_interest is not None and len(self.open_interest) > 0:
            # Sort by open interest (highest first)
            paired = [(p, oi, i) for i, (p, oi) in enumerate(zip(self.prices, self.open_interest))
                      if not np.isnan(p) and not np.isnan(oi)]
            paired.sort(key=lambda x: x[1], reverse=True)
            return np.array([p for p, _, _ in paired])

        else:
            # Fallback to calendar order
            return self.sequence_curve('calendar')

    def get_granular_slope(self, method='linear_regression', interpolate=True) -> Dict[str, float]:
        """
        Calculate granular slope using various methods

        Enhanced from original to include path-based features
        """
        valid_prices = [(i, p, self.curve_month_labels[i]) for i, p in enumerate(self.prices)
                        if not np.isnan(p)]

        if len(valid_prices) < 2:
            return {'slope': 0.0, 'valid_points': len(valid_prices)}

        positions = np.array([v[0] for v in valid_prices])
        prices = np.array([v[1] for v in valid_prices])

        result = {'valid_points': len(valid_prices)}

        if method == 'linear_regression':
            slope, intercept, r_value, p_value, std_err = stats.linregress(positions, prices)
            result.update({
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err
            })

        elif method == 'polynomial':
            degree = min(3, len(prices) - 1)
            coeffs = np.polyfit(positions, prices, degree)
            poly = np.poly1d(coeffs)

            # Calculate curvature (second derivative)
            if degree >= 2:
                second_deriv = np.polyder(poly, 2)
                result['curvature'] = float(second_deriv(positions.mean()))

            result['slope'] = float(np.polyder(poly)(positions.mean()))
            result['polynomial_coeffs'] = coeffs.tolist()

        # Add shape complexity using entropy
        if len(prices) >= 3:
            price_changes = np.diff(prices)
            if price_changes.std() > 0:
                normalized_changes = (price_changes - price_changes.mean()) / price_changes.std()
                # Calculate approximate entropy
                result['shape_entropy'] = -np.sum(np.abs(normalized_changes) *
                                                  np.log(np.abs(normalized_changes) + 1e-10))

        return result

    def calculate_term_structure_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive term structure metrics
        """
        metrics = {}
        valid_prices = [p for p in self.prices if not np.isnan(p)]

        if len(valid_prices) < 2:
            return metrics

        # Basic shape metrics
        metrics['mean_level'] = np.mean(valid_prices)
        metrics['price_dispersion'] = np.std(valid_prices)
        metrics['skewness'] = skew(valid_prices)
        metrics['kurtosis'] = kurtosis(valid_prices)

        # Contango/Backwardation metrics
        price_diffs = np.diff(valid_prices)
        metrics['contango_ratio'] = np.sum(price_diffs > 0) / len(price_diffs) if len(price_diffs) > 0 else 0
        metrics['avg_calendar_spread'] = np.mean(price_diffs) if len(price_diffs) > 0 else 0

        # Term structure convexity
        if len(valid_prices) >= 3:
            second_diffs = np.diff(price_diffs)
            metrics['convexity'] = np.mean(second_diffs)
            metrics['max_convexity'] = np.max(np.abs(second_diffs))

        # Volume/OI concentration (Herfindahl index)
        if self.volumes is not None and len(self.volumes) > 0:
            valid_vols = [v for v in self.volumes if not np.isnan(v)]
            if valid_vols and sum(valid_vols) > 0:
                vol_shares = np.array(valid_vols) / sum(valid_vols)
                metrics['volume_concentration'] = np.sum(vol_shares ** 2)

        if self.open_interest is not None and len(self.open_interest) > 0:
            valid_oi = [o for o in self.open_interest if not np.isnan(o)]
            if valid_oi and sum(valid_oi) > 0:
                oi_shares = np.array(valid_oi) / sum(valid_oi)
                metrics['oi_concentration'] = np.sum(oi_shares ** 2)

        return metrics

    def __getitem__(self, key):
        """
        Access curve data by contract expiry label or sequential M{0+} codes
        
        Parameters:
        - key: str (expiry label like 'F', 'G', 'H' or sequential like 'M0', 'M1') 
               or int (sequential index with np.nan values removed)
        
        Returns:
        - float: Price for the specified contract
        
        Examples:
        - curve['F'] -> Front month price (if F is in curve_month_labels)
        - curve['M0'] -> First contract price from seq_prices
        - curve[0] -> First contract price (integer index)
        """
        if isinstance(key, str):
            if key in self.curve_month_labels:
                idx = self.curve_month_labels.index(key)
                return self.prices[idx] if idx < len(self.prices) and not np.isnan(self.prices[idx]) else None

            base_code = normalize_contract_code(key)
            if base_code and base_code in self.curve_month_labels:
                matching_indices = [i for i, label in enumerate(self.curve_month_labels)
                                    if normalize_contract_code(label) == base_code]
                if len(matching_indices) == 1:
                    idx = matching_indices[0]
                    return self.prices[idx] if not np.isnan(self.prices[idx]) else None

            if key.startswith('M') and len(key) >= 2:
                # Sequential M{0-11} codes
                try:
                    seq_idx = int(key[1:])
                    if self.seq_prices is not None:
                        # Filter out NaN values
                        valid_prices = [p for p in self.seq_prices if not np.isnan(p)]
                        if 0 <= seq_idx < len(valid_prices):
                            return valid_prices[seq_idx]
                    else:
                        # Fallback to calendar order with NaN filtering
                        valid_prices = [p for p in self.prices if not np.isnan(p)]
                        if 0 <= seq_idx < len(valid_prices):
                            return valid_prices[seq_idx]
                except (ValueError, IndexError):
                    pass
                    
        elif isinstance(key, int):
            # Integer index with NaN filtering
            if self.seq_prices is not None:
                valid_prices = [p for p in self.seq_prices if not np.isnan(p)]
            else:
                valid_prices = [p for p in self.prices if not np.isnan(p)]
                
            if 0 <= key < len(valid_prices):
                return valid_prices[key]
            elif key < 0 and abs(key) <= len(valid_prices):
                return valid_prices[key]  # Negative indexing
                
        raise KeyError(f"Invalid key: {key}. Use expiry labels (F, G, H...), sequential codes (M0, M1...), or integer indices.")


@dataclass
class Contract(SpreadFeature):
    symbol: str = ""
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    label: str = ""
    volume: np.ndarray = field(default_factory=lambda: np.array([]))
    dte: np.ndarray = field(default_factory=lambda: np.array([]))
    oi : np.ndarray = field(default_factory=lambda : np.array([]))
    index: pd.Index = None
    continuous: bool = False
    is_front_month: bool = False
    early_roll_days: int = 45
    direction: str = "vertical"
    tracker: ExpiryTracker = None
    expiration_date: datetime = None

    def __post_init__(self):
        super().__post_init__()
        # Initialize tracker if not provided
        if self.tracker is None and self.symbol and self.label:
            year_hint = infer_contract_year(self.label, datetime.today().year) or datetime.today().year

            days_hint = None
            if isinstance(self.dte, np.ndarray) and self.dte.size > 0:
                valid_dte = self.dte[~np.isnan(self.dte)] if self.dte.dtype != object else self.dte
                try:
                    valid_dte = valid_dte[~np.isnan(valid_dte)]
                except TypeError:
                    pass
                if len(valid_dte) > 0:
                    days_hint = int(float(valid_dte[0]))
            elif isinstance(self.dte, (int, float)):
                if not np.isnan(self.dte):
                    days_hint = int(float(self.dte))

            self.tracker = ExpiryTracker(
                self.symbol,
                month_code=self.label,
                year=year_hint,
                days_to_expiry=days_hint
            )
            self.expiration_date = self.tracker.expiry_date if self.tracker else None
        if len(self.data.shape) > 1:
            if self.data.shape[1] > 3:
                self.open, self.high, self.low, self.close = [self.data[:, i] for i in range(4)]
        else:
            self.close = self.data

        self.returns = (self.close - np.roll(self.close, 1)) / self.close

    @property
    def _is_rolled(self):
        if isinstance(self.dte, np.ndarray) and len(self.dte) > 0:
            return np.any(self.dte < self.early_roll_days)
        elif isinstance(self.dte, (int, float)):
            return self.dte < self.early_roll_days
        return False

    @property
    def _is_expired(self):
        if isinstance(self.dte, np.ndarray) and len(self.dte) > 0:
            return np.any(self.dte < 0)
        elif isinstance(self.dte, (int, float)):
            return self.dte < 0
        return False

    def roll_over(self):
        if self.expiration_date:
            self.expiration_date = self.expiration_date + timedelta(days=365)
        return

    def __sub__(self, other):
        if isinstance(other, Contract) and len(other) == len(self):
            abs_price = self.close - other.close

            return SpreadReturns(long_price=self.close,
                                 short_price=other.close,
                                 long_returns=self.returns,
                                 short_returns=other.returns,
                                 long_volume=self.volume,
                                 short_volume=other.volume,
                                 long_ct_oi=self.oi,
                                 short_ct_oi=other.oi,
                                 long_label=self.label,
                                 short_label=other.label,
                                 index = self.index)

    def __add__(self, other):
        if isinstance(other, Contract):
            data = {
                'abs_spread': (self.data + other.data),
                'dte': self.dte if self.dte < other.dte else other.dte
            }
            return data

    def __len__(self):
        return len(self.index)

    def __mul__(self, other):
        if isinstance(other, Contract):
            raise ValueError("Unable to multiply two Contract instances")
        elif isinstance(other, int):
            return other * self.data
    
    def get_prices(self, dates: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        Get price data (from .data attribute) for specified dates.
        
        Parameters:
        -----------
        dates : pd.DatetimeIndex, optional
            Specific dates to extract. If None, returns all data.
            
        Returns:
        --------
        np.ndarray
            Price data for the specified dates
        """
        if dates is None:
            return self.data if len(self.data.shape) == 1 else self.close
        
        if self.index is None or len(self.index) == 0:
            return self.data if len(self.data.shape) == 1 else self.close
        
        try:
            # Use pandas get_indexer for efficient date lookup
            import pandas as pd
            
            # Convert dates to pandas Index if not already
            if not isinstance(dates, pd.Index):
                dates = pd.Index(dates)
            
            # Use get_indexer for efficient lookup - returns -1 for missing dates
            date_indices = self.index.get_indexer(dates)
            
            # Handle missing dates by finding nearest
            missing_mask = date_indices == -1
            if missing_mask.any():
                missing_dates = dates[missing_mask]
                for i, missing_date in enumerate(missing_dates):
                    original_idx = np.where(missing_mask)[0][i]
                    nearest_idx = np.argmin(np.abs(self.index - missing_date))
                    date_indices[original_idx] = nearest_idx
            
            prices = self.data if len(self.data.shape) == 1 else self.close
            return prices[date_indices]
            
        except Exception:
            # Fallback: return all data
            return self.data if len(self.data.shape) == 1 else self.close


@dataclass
class SpreadData:
    """
    Data loading and management class with volume-based sequentialization
    All data stored as numpy arrays except index/timestamps

    IMPORTANT: All calculations should be performed on sequentialized data (seq_data),
    NOT on raw self.curve data which is unordered. Use seq_data.seq_prices, seq_data.seq_spreads, etc.
    for analysis. Raw self.curve should only be used when explicitly ordered by the user.
    """
    symbol: str = None
    curve: np.ndarray = None
    days_to_expiration = None
    dte: Optional[np.ndarray] = None
    seq_dte: Optional[np.ndarray] = None

    # Sequential DataFrames (loaded directly from HDF5)
    seq_prices: Optional[pd.DataFrame] = None
    seq_volume: Optional[pd.DataFrame] = None
    seq_oi: Optional[pd.DataFrame] = None
    seq_labels: Optional[pd.DataFrame] = None
    seq_spreads: Optional[pd.DataFrame] = None
    seq_dte: Optional[pd.DataFrame] = None

    # Volume and OI curves
    volume_curve: np.ndarray = None
    oi_curve: np.ndarray = None
    
    # Spot price data (separate real from synthetic) - both as Series with datetime index
    spot_prices: Optional[pd.Series] = None  # Real spot prices from market data (intersected with futures index)
    synthetic_spot_prices: Optional[pd.Series] = None  # Calculated synthetic spot prices with datetime index
    
    # Convenience yield data
    convenience_yield: Optional[np.ndarray] = None  # Front month convenience yield

    # Metadata
    commodity: str = ""
    index: pd.DatetimeIndex = None
    timestamps: pd.DatetimeIndex = None

    # Additional analysis data
    levy_areas: np.ndarray = None
    signatures: np.ndarray = None

    # Roll dates DataFrame for enhanced analysis
    roll_dates: Optional[pd.DataFrame] = None

    # Cache for computed slopes
    _slopes_cache: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if self.symbol is None:
            return

        self.commodity = self.symbol  # For compatibility
        self.contracts = {}
        self._slopes_cache = {}

        # Set up timestamps/index - ensure we have an index
        if self.index is not None:
            self.timestamps = self.index
        elif self.timestamps is not None:
            self.index = self.timestamps
        else:
            # Try to infer index from data
            if hasattr(self, 'seq_prices') and isinstance(self.seq_prices, pd.DataFrame):
                self.index = self.seq_prices.index
                self.timestamps = self.index
            elif hasattr(self, 'curve') and isinstance(self.curve, pd.DataFrame):
                self.index = self.curve.index
                self.timestamps = self.index

        # Convert DataFrames to numpy arrays
        self._convert_to_arrays()

        # Initialize data client mappings
        self.CODE_TO_MONTH = MONTH_CODE_MAP
        self.NUMBER_TO_CODE = {v: k for k, v in MONTH_CODE_MAP.items()}

        # Initialize derived features and SeqData
        self._initialize_features()

    def _convert_to_arrays(self):
        """Keep curve data as DataFrames with labels, convert other data to numpy arrays"""
        # Keep curve data as DataFrame to preserve column labels (contract names)
        if isinstance(self.curve, pd.DataFrame):
            # Store DataFrame version (curve already is a DataFrame)
            self.curve_df = self.curve
            # Don't convert curve to numpy - keep as DataFrame for constant maturity functionality

        # Convert volume and oi curves to numpy arrays as before
        if isinstance(self.volume_curve, pd.DataFrame):
            self.volume_curve_df = self.volume_curve.copy()
        if hasattr(self.volume_curve, 'values'):
            self.volume_curve = self.volume_curve.values

        if isinstance(self.oi_curve, pd.DataFrame):
            self.oi_curve_df = self.oi_curve.copy()
        if hasattr(self.oi_curve, 'values'):
            self.oi_curve = self.oi_curve.values
        if hasattr(self.convenience_yield, 'values'):
            self.convenience_yield = self.convenience_yield.values
    
    def calculate_synthetic_spot(self) -> Optional[pd.Series]:
        """
        Calculate synthetic spot prices using geometric weighted mean of all contracts.
        
        Uses volume-weighted geometric mean where weights are inversely proportional 
        to days-to-expiry to emphasize near-term contracts.
        
        Returns:
        --------
        pd.Series or None
            Synthetic spot prices time series with datetime index, or None if insufficient data
        """
        if self.curve is None or len(self.curve) == 0:
            return None
            
        if self.seq_prices is None or len(self.seq_prices) == 0:
            return None

        # Use sequential prices DataFrame directly
        prices = self.seq_prices.values
        
        # Get volume weights if available
        volume_weights = None
        if hasattr(self, 'seq_volume') and self.seq_volume is not None:
            volume_weights = self.seq_volume.values
        
        # Get days-to-expiry weights if available
        dte_weights = None
        if hasattr(self, 'seq_dte') and self.seq_dte is not None:
            dte_weights = self.seq_dte.values
        
        n_times, n_contracts = prices.shape
        synthetic_spot = np.full(n_times, np.nan)
        
        for t in range(n_times):
            price_row = prices[t, :]
            
            # Skip if all prices are NaN
            valid_mask = ~np.isnan(price_row)
            if not np.any(valid_mask):
                continue
                
            valid_prices = price_row[valid_mask]
            
            # Calculate weights
            weights = np.ones(len(valid_prices))
            
            # Apply volume weighting if available
            if volume_weights is not None and t < volume_weights.shape[0]:
                vol_row = volume_weights[t, :]  # Get full row first
                vol_row = vol_row[valid_mask]    # Then apply mask
                vol_valid = ~np.isnan(vol_row) & (vol_row > 0)
                if np.any(vol_valid):
                    weights[vol_valid] *= vol_row[vol_valid]
            
            # Apply inverse days-to-expiry weighting if available
            if dte_weights is not None and t < dte_weights.shape[0]:
                if isinstance(dte_weights, np.ndarray) and len(dte_weights.shape) == 2:
                    dte_row = dte_weights[t, :]  # Get full row first
                    dte_row = dte_row[valid_mask] # Then apply mask
                else:
                    # Handle case where dte_weights is 1D
                    dte_row = dte_weights[valid_mask] if len(dte_weights) >= n_contracts else np.full(len(valid_prices), 30)
                
                dte_valid = ~np.isnan(dte_row) & (dte_row > 0)
                if np.any(dte_valid):
                    # Inverse weighting - closer contracts get higher weight
                    # Add small epsilon to avoid division by zero
                    inverse_dte = 1.0 / (dte_row + 1.0)
                    weights[dte_valid] *= inverse_dte[dte_valid]
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                
                # Calculate geometric weighted mean
                # log(geometric_mean) = weighted_sum(log(prices))
                log_prices = np.log(valid_prices)
                log_geometric_mean = np.sum(weights * log_prices)
                synthetic_spot[t] = np.exp(log_geometric_mean)
        
        # Convert to pandas Series with datetime index
        if self.seq_data.timestamps is not None:
            return pd.Series(synthetic_spot, index=self.seq_data.timestamps, name='synthetic_spot')
        elif self.index is not None:
            # Ensure index length matches synthetic_spot length
            index_to_use = self.index[:len(synthetic_spot)] if len(self.index) >= len(synthetic_spot) else self.index
            if len(index_to_use) != len(synthetic_spot):
                # Create a compatible index
                index_to_use = pd.date_range(start=self.index[0], periods=len(synthetic_spot), freq='D')
            return pd.Series(synthetic_spot, index=index_to_use, name='synthetic_spot')
        else:
            # Fallback: create a simple date range
            return pd.Series(synthetic_spot, 
                           index=pd.date_range(start='2020-01-01', periods=len(synthetic_spot), freq='D'),
                           name='synthetic_spot')
    
    @property
    def spot(self) -> Optional[pd.Series]:
        """
        Property to access spot prices as pandas Series.

        Returns real spot prices if available, otherwise synthetic spot prices.
        This provides a clean interface for accessing spot data.

        Returns:
        --------
        Optional[pd.Series]
            Spot prices with datetime index, or None if no spot data available
        """
        return self.get_spot_prices(prefer_real=True)

    def get_spot_prices(self, prefer_real: bool = True) -> Optional[pd.Series]:
        """
        Get spot prices, preferring real or synthetic.

        Parameters:
        -----------
        prefer_real : bool, default True
            If True, returns real spot prices if available, otherwise synthetic.
            If False, returns synthetic spot prices if available, otherwise real.

        Returns:
        --------
        Optional[pd.Series]
            Spot prices with datetime index, or None if no spot data available
        """
        if prefer_real:
            # Return real spot prices if available (already Series with intersected index)
            return self.spot_prices if self.spot_prices is not None else self.synthetic_spot_prices
        else:
            # Return synthetic spot prices if available, otherwise real
            return self.synthetic_spot_prices if self.synthetic_spot_prices is not None else self.spot_prices
    
    def calculate_convenience_yield(self, 
                                  contract_months: Optional[Union[str, List[str]]] = None,
                                  prefer_real_spot: bool = True,
                                  annualized: bool = True) -> Optional[np.ndarray]:
        """
        Calculate convenience yield as the yield from spot - futures price.
        
        Convenience yield = (Futures Price - Spot Price) / Spot Price * (365 / Days to Expiry)
        
        A positive convenience yield indicates backwardation (spot > futures),
        suggesting scarcity or high storage costs.
        A negative convenience yield indicates contango (futures > spot),
        suggesting abundance or low storage costs.
        
        Parameters:
        -----------
        contract_months : str, list of str, or None
            Contract month codes to calculate convenience yield for.
            If None, uses the front month contract (M0).
            Examples: 'F', ['F', 'G', 'H'], 'M0', ['M0', 'M1']
        prefer_real_spot : bool, default True
            Whether to prefer real spot prices over synthetic ones
        annualized : bool, default True
            If True, annualizes the yield using days to expiry
            
        Returns:
        --------
        np.ndarray or None
            Convenience yield time series. Shape depends on contract_months:
            - Single contract: 1D array of yields over time
            - Multiple contracts: 2D array (time x contracts)
            Returns None if insufficient data available
            
        Examples:
        ---------
        >>> spread_data = SpreadData('CL_F')
        
        # Front month convenience yield
        >>> cy_front = spread_data.calculate_convenience_yield()
        
        # Specific contract months
        >>> cy_specific = spread_data.calculate_convenience_yield(['F', 'G', 'H'])
        
        # Sequential contracts (M0=front month, M1=second month, etc.)
        >>> cy_seq = spread_data.calculate_convenience_yield(['M0', 'M1', 'M2'])
        """
        # Get spot prices (as Series for proper index handling)
        spot_data = self.get_spot_prices(prefer_real=prefer_real_spot)
        if spot_data is None:
            return None
        
        # Extract values and index for processing
        if isinstance(spot_data, pd.Series):
            spot_prices = spot_data.values
            spot_index = spot_data.index
        else:
            spot_prices = spot_data
            spot_index = self.index if self.index is not None else None
            
        # Handle contract selection
        if contract_months is None:
            contract_months = ['M0']  # Default to front month
        elif isinstance(contract_months, str):
            contract_months = [contract_months]
            
        convenience_yields = []
        
        for contract_month in contract_months:
            # Get futures prices for this contract
            futures_prices = None
            days_to_expiry = None
            
            if contract_month.startswith('M') and contract_month[1:].isdigit():
                # Sequential contract like M0, M1, M2
                contract_idx = int(contract_month[1:])
                if (self.seq_data and hasattr(self.seq_data, 'seq_prices') and self.seq_data.seq_prices is not None):
                    seq_prices = self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices
                    if contract_idx < seq_prices.shape[1]:
                        futures_prices = seq_prices[:, contract_idx]
                        
                        # Get corresponding days to expiry
                        if self.seq_dte is not None:
                            if isinstance(self.seq_dte, np.ndarray) and len(self.seq_dte.shape) == 2:
                                if contract_idx < self.seq_dte.shape[1]:
                                    days_to_expiry = self.seq_dte[:, contract_idx]
            
            elif len(contract_month) == 1 and contract_month in MONTH_CODE_ORDER:
                # Specific contract month like F, G, H
                if hasattr(self, 'contracts') and contract_month in self.contracts:
                    contract = self.contracts[contract_month]
                    futures_prices = contract.data
                    days_to_expiry = contract.dte
                elif (self.curve is not None and hasattr(self.curve, 'shape') and 
                      len(self.curve.shape) == 2):
                    # Try to find in main curve data
                    curve_df = pd.DataFrame(self.curve, index=self.index, 
                                          columns=getattr(self, 'curve_columns', None))
                    if hasattr(curve_df, 'columns') and contract_month in curve_df.columns:
                        futures_prices = curve_df[contract_month].values
                        
                        # Get corresponding dte
                        if (self.dte is not None and hasattr(self.dte, 'shape') and 
                            len(self.dte.shape) == 2):
                            dte_df = pd.DataFrame(self.dte, index=self.index,
                                                columns=getattr(self, 'curve_columns', None))
                            if hasattr(dte_df, 'columns') and contract_month in dte_df.columns:
                                days_to_expiry = dte_df[contract_month].values
            
            if futures_prices is None:
                warnings.warn(f"Could not find futures prices for contract {contract_month}")
                convenience_yields.append(np.full(len(spot_prices), np.nan))
                continue
                
            # Ensure arrays have same length and are 1D
            min_length = min(len(spot_prices), len(futures_prices))
            spot_truncated = np.asarray(spot_prices[:min_length]).flatten()
            futures_truncated = np.asarray(futures_prices[:min_length]).flatten()
            
            # Calculate basic convenience yield: (Futures - Spot) / Spot
            valid_mask = (spot_truncated != 0) & ~np.isnan(spot_truncated) & ~np.isnan(futures_truncated)
            convenience_yield = np.full(min_length, np.nan)
            
            if np.any(valid_mask):
                convenience_yield[valid_mask] = ((futures_truncated[valid_mask] - spot_truncated[valid_mask]) / 
                                               spot_truncated[valid_mask])
                
                # Annualize if requested and dte data is available
                if annualized and days_to_expiry is not None:
                    dte_truncated = np.asarray(days_to_expiry[:min_length]).flatten()
                    dte_valid = (dte_truncated > 0) & ~np.isnan(dte_truncated)
                    # Ensure 1D boolean indexing
                    combined_valid = valid_mask & dte_valid
                    if np.any(combined_valid):
                        annualization_factor = 365.25 / dte_truncated[combined_valid]
                        convenience_yield[combined_valid] *= annualization_factor
                elif annualized:
                    warnings.warn(f"Cannot annualize convenience yield for {contract_month}: no days-to-expiry data")
            
            convenience_yields.append(convenience_yield)
        
        # Return format depends on number of contracts - return as pandas Series/DataFrame
        if len(convenience_yields) == 1:
            # Single contract - return as Series with datetime index
            cy_data = convenience_yields[0]
            if spot_index is not None and len(spot_index) >= len(cy_data):
                return pd.Series(cy_data, index=spot_index[:len(cy_data)], name=f'convenience_yield_{contract_months[0]}')
            else:
                return pd.Series(cy_data, name=f'convenience_yield_{contract_months[0]}')
        else:
            # Multiple contracts - return as DataFrame with datetime index
            max_length = max(len(cy) for cy in convenience_yields)
            result = np.full((max_length, len(convenience_yields)), np.nan)
            for i, cy in enumerate(convenience_yields):
                result[:len(cy), i] = cy
            
            # Create DataFrame with proper index and column names
            columns = [f'convenience_yield_{contract}' for contract in contract_months]
            if spot_index is not None and len(spot_index) >= max_length:
                return pd.DataFrame(result, index=spot_index[:max_length], columns=columns)
            else:
                return pd.DataFrame(result, columns=columns)

    def _calculate_slopes(self) -> Dict[str, np.ndarray]:
        """Calculate slopes for all curves simultaneously and cache results"""
        if 'curve_slopes' not in self._slopes_cache:
            if self.curve is not None and len(self.curve.shape) == 2:
                # Calculate slopes across contracts for each time point
                slopes = []
                for i, row in enumerate(self.curve):
                    valid_prices = row[~np.isnan(row)]
                    if len(valid_prices) >= 2:
                        slope = np.polyfit(range(len(valid_prices)), valid_prices, 1)[0]
                        slopes.append(slope)
                    else:
                        slopes.append(np.nan)

                self._slopes_cache['curve_slopes'] = np.array(slopes)
            else:
                self._slopes_cache['curve_slopes'] = np.array([])

        return self._slopes_cache

    def _get_seq_dataframe(self, attr_name: str) -> Optional[pd.DataFrame]:
        """Return sequential data as a DataFrame if available."""
        value = getattr(self, attr_name, None)
        if isinstance(value, pd.DataFrame):
            return value
        return None

    def _get_seq_feature(self, attr_name: str):
        """Return a sequential feature from either the attribute or seq_data namespace."""
        seq_data = getattr(self, 'seq_data', None)
        if seq_data is not None and hasattr(seq_data, attr_name):
            return getattr(seq_data, attr_name)
        return getattr(self, attr_name, None)

    # Removed _create_seq_feature and _feature_to_dataframe methods
    # They were unnecessary complexity - we now store DataFrames directly

    def _initialize_features(self):
        """Initialize sequential data directly without unnecessary SpreadFeature wrappers."""
        # For backward compatibility, keep seq_data attribute but simplify internal structure
        # Store sequential data directly as DataFrames/arrays for faster access

        # No need to create seq_data if no curve data
        if self.curve is None or len(self.curve) == 0:
            return

        # Sequential DataFrames are already loaded in _extract_sequential_data
        # Just ensure they exist and are properly formatted

        # Handle seq_dte specially as it may need array conversion
        if hasattr(self, 'seq_dte') and self.seq_dte is not None and not isinstance(self.seq_dte, np.ndarray):
            if isinstance(self.seq_dte, pd.DataFrame):
                self.seq_dte = self.seq_dte.values
            else:
                self.seq_dte = np.asarray(self.seq_dte, dtype=float)
                if self.seq_dte.ndim == 1:
                    self.seq_dte = self.seq_dte[:, np.newaxis]

        # Create a simple namespace object for backward compatibility
        # This allows access like self.seq_data.seq_prices but stores DataFrames directly
        self.seq_data = type('SeqData', (), {
            'timestamps': self.index,
            'seq_labels': getattr(self, 'seq_labels', None),
            'seq_prices': getattr(self, 'seq_prices', None),
            'seq_spreads': getattr(self, 'seq_spreads', None),
            'seq_oi': getattr(self, 'seq_oi', None),
            'seq_volume': getattr(self, 'seq_volume', None),
            'seq_dte': getattr(self, 'seq_dte', None),
            'roll_dates': getattr(self, 'roll_dates', None)
        })()

        # Calculate derived features if we have price data
        if hasattr(self, 'seq_prices') and self.seq_prices is not None:
            try:
                self.calculate_maturity_bucket_spreads()
            except Exception as exc:
                warnings.warn(f"Could not calculate maturity bucket spreads: {exc}")

            try:
                self.synthetic_spot_prices = self.calculate_synthetic_spot()
            except Exception as exc:
                warnings.warn(f"Could not calculate synthetic spot prices: {exc}")
                self.synthetic_spot_prices = None

            try:
                self.convenience_yield = self.calculate_convenience_yield()
            except Exception as exc:
                warnings.warn(f"Could not calculate convenience yield: {exc}")
                self.convenience_yield = None

    def get_sequential_prices(self, dropna: bool = False) -> pd.DataFrame:
        """
        Get sequential prices DataFrame directly (FAST - no curve recreation).

        Returns:
        --------
        pd.DataFrame
            Sequential prices with DatetimeIndex and M0, M1, M2... columns

        Notes:
        ------
        This method returns the sequential prices DataFrame directly without
        recreating FuturesCurve objects, making it much more efficient than
        get_seq_curves() for simple DataFrame operations.
        """
        if hasattr(self, 'seq_prices') and self.seq_prices is not None:
            df = self.seq_prices
        else:
            raise ValueError("No sequential price data available")

        return df.dropna(how='all') if dropna else df


    def get_sequential_spreads(self, dropna: bool = False) -> Optional[pd.DataFrame]:
        """
        Get sequential spreads DataFrame directly (FAST - no curve recreation).

        Returns:
        --------
        pd.DataFrame
            Sequential spreads with DatetimeIndex and M0, M1, M2... columns
        """
        if hasattr(self, 'seq_spreads') and self.seq_spreads is not None:
            df = self.seq_spreads
        else:
            return None

        return df.dropna(how='all') if dropna else df




    def get_sequential_volume(self, dropna: bool = False) -> Optional[pd.DataFrame]:
        if hasattr(self, 'seq_volume') and self.seq_volume is not None:
            df = self.seq_volume
        else:
            return None
        return df.dropna(how='all') if dropna else df

    def get_sequential_oi(self, dropna: bool = False) -> Optional[pd.DataFrame]:
        if hasattr(self, 'seq_oi') and self.seq_oi is not None:
            df = self.seq_oi
        else:
            return None

        return df.dropna(how='all') if dropna else df

    def get_contract_expiries(self) -> pd.Series:
        """Return a Series mapping contract identifiers to expiry dates."""
        expiries: Dict[str, pd.Timestamp] = {}

        contracts = getattr(self, 'contracts', {}) or {}
        for label, contract in contracts.items():
            expiry = getattr(contract, 'expiration_date', None)
            if expiry is not None:
                expiries[label] = pd.Timestamp(expiry)

        if not expiries:
            dte_df: Optional[pd.DataFrame] = None
            if isinstance(self.seq_dte, pd.DataFrame):
                dte_df = self.seq_dte
            elif self._get_seq_feature('seq_dte') is not None:
                seq_dte_array = np.asarray(self.seq_data.seq_dte, dtype=float)
                if seq_dte_array.ndim == 1:
                    seq_dte_array = seq_dte_array[:, np.newaxis]
                try:
                    price_df = self.get_sequential_prices()
                except ValueError:
                    price_df = None
                if price_df is not None and seq_dte_array.size > 0:
                    n_rows = min(seq_dte_array.shape[0], len(price_df.index))
                    n_cols = min(seq_dte_array.shape[1], price_df.shape[1])
                    dte_df = pd.DataFrame(
                        seq_dte_array[:n_rows, :n_cols],
                        index=price_df.index[:n_rows],
                        columns=price_df.columns[:n_cols]
                    )

            if dte_df is not None:
                for column in dte_df.columns:
                    series = dte_df[column].dropna()
                    if series.empty:
                        continue
                    first_date = series.index[0]
                    dte_days = series.iloc[0]
                    if pd.notna(dte_days):
                        expiries[column] = pd.Timestamp(first_date) + pd.to_timedelta(int(dte_days), unit='D')

        if not expiries:
            return pd.Series(dtype="datetime64[ns]")

        expiries_series = pd.Series(expiries)
        expiries_series = pd.to_datetime(expiries_series)
        return expiries_series

    def get_constant_maturity_inputs(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return price and expiry inputs compatible with utility interpolation code."""
        if hasattr(self, 'curve_df') and isinstance(self.curve_df, pd.DataFrame):
            prices = self.curve_df.copy()
        elif isinstance(self.curve, pd.DataFrame):
            prices = self.curve.copy()
        elif isinstance(self.curve, np.ndarray) and hasattr(self, 'curve_columns'):
            prices = pd.DataFrame(self.curve, index=self.index, columns=self.curve_columns)
        else:
            prices = self.get_sequential_prices()

        expiries = self.get_contract_expiries()
        if not expiries.empty:
            common = [col for col in prices.columns if col in expiries.index]
            if common:
                prices = prices.loc[:, common]
                expiries = expiries.loc[common]
            else:
                seq_prices = self.get_sequential_prices(dropna=False)
                seq_common = [col for col in seq_prices.columns if col in expiries.index]
                if seq_common:
                    prices = seq_prices.loc[:, seq_common]
                    expiries = expiries.loc[seq_common]

        return prices, expiries

    def create_futures_curve(self, date: Optional[datetime] = None) -> FuturesCurve:
        """
        Create a FuturesCurve object for a specific date using seq_data if available
        """
        if date is None:
            date = datetime.now()

        # Use sequential DataFrames if available for better performance
        if self.seq_prices is not None:
            return self._create_curve_from_seq_data(date)

        # Fall back to original curve-based method
        if not hasattr(self, 'curve') or self.curve is None:
            raise ValueError("No curve data available")

        # Get data for specific date
        if date in set(self.index.tolist()):
            price_row = self.curve[self.index == date]
        else:
            # Get nearest date
            time_diff = np.abs(self.curve.index - date)
            nearest_idx = time_diff.argmin()
            price_row = self.curve[nearest_idx]

        # Extract month labels and prices
        month_labels = []
        prices = []
        volumes = []
        ois = []

        for month_code in MONTH_CODE_ORDER:
            if month_code in price_row.index:
                price = price_row[month_code]
                if pd.notna(price):
                    month_labels.append(month_code)
                    prices.append(float(price))

                    # Add volume if available
                    if hasattr(self, 'volume') and month_code in self.volume.columns:
                        if date in self.volume.index:
                            vol = _safe_get_value(self.volume)
                        else:
                            vol = self.volume.iloc[nearest_idx][month_code]
                        volumes.append(float(vol) if pd.notna(vol) else np.nan)
                    else:
                        volumes.append(np.nan)

                    # Add OI if available
                    if hasattr(self, 'oi') and month_code in self.oi.columns:
                        if date in self.oi.index:
                            oi = self.oi.loc[date, month_code]
                        else:
                            oi = self.oi.iloc[nearest_idx][month_code]
                        ois.append(float(oi) if pd.notna(oi) else np.nan)
                    else:
                        ois.append(np.nan)

        return FuturesCurve(
            ref_date=date,
            curve_month_labels=month_labels,
            prices=prices,
            volumes=volumes if any(not np.isnan(v) for v in volumes) else None,
            open_interest=ois if any(not np.isnan(o) for o in ois) else None
        )

    def _create_curve_from_seq_data(self, date: datetime) -> Optional[FuturesCurve]:
        """Create FuturesCurve using sequential DataFrames directly"""
        if self.seq_prices is None or len(self.seq_prices) == 0:
            return None

        # Find the index for the requested date
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()

        # Find closest date in the seq_prices index
        try:
            if date in self.seq_prices.index:
                date_idx = self.seq_prices.index.get_loc(date)
            else:
                # Find nearest date
                date_idx = self.seq_prices.index.get_indexer([date], method='nearest')[0]
                if date_idx == -1:
                    return None
        except (KeyError, IndexError):
            return None

        # Check if date_idx is within bounds
        if date_idx >= len(self.seq_prices):
            return None

        # Extract data for this date - keep NaN values to maintain consistency
        prices = []
        volumes = []
        ois = []
        seq_dtes = []
        labels = []

        # Extract price data for this date
        try:
            price_row = self.seq_prices.iloc[date_idx]
        except (IndexError, KeyError):
            return None

        # Keep all values including NaN for consistent indexing
        for i, p in enumerate(price_row):
            if not np.isnan(p):
                prices.append(float(p))
                # Use corresponding label from seq_labels or default
                if self.seq_labels is not None and date_idx < len(self.seq_labels):
                    try:
                        label_row = self.seq_labels.iloc[date_idx]
                        if i < len(label_row) and pd.notna(label_row.iloc[i]):
                            labels.append(str(label_row.iloc[i]))
                        else:
                            labels.append(f'M{i}')
                    except (IndexError, KeyError):
                        labels.append(f'M{i}')
                else:
                    labels.append(f'M{i}')

        # Extract volume data for valid contracts only
        if self.seq_volume is not None and date_idx < len(self.seq_volume):
            try:
                volume_row = self.seq_volume.iloc[date_idx]
                for i, p in enumerate(price_row):
                    if not np.isnan(p) and i < len(volume_row):
                        vol = volume_row.iloc[i] if hasattr(volume_row, 'iloc') else volume_row[i]
                        volumes.append(float(vol) if pd.notna(vol) else np.nan)
            except (IndexError, KeyError):
                volumes = [np.nan] * len(prices)

        # Extract OI data for valid contracts only
        if self.seq_oi is not None and date_idx < len(self.seq_oi):
            try:
                oi_row = self.seq_oi.iloc[date_idx]
                for i, p in enumerate(price_row):
                    if not np.isnan(p) and i < len(oi_row):
                        oi = oi_row.iloc[i] if hasattr(oi_row, 'iloc') else oi_row[i]
                        ois.append(float(oi) if pd.notna(oi) else np.nan)
            except (IndexError, KeyError):
                ois = [np.nan] * len(prices)

        # Extract DTE data for valid contracts
        if self.seq_dte is not None and date_idx < len(self.seq_dte):
            try:
                if isinstance(self.seq_dte, pd.DataFrame):
                    dte_row = self.seq_dte.iloc[date_idx]
                    for i, p in enumerate(price_row):
                        if not np.isnan(p) and i < len(dte_row):
                            dte = dte_row.iloc[i]
                            seq_dtes.append(float(dte) if pd.notna(dte) else np.nan)
                else:
                    # Handle numpy array case
                    dte_row = self.seq_dte[date_idx] if hasattr(self.seq_dte, '__getitem__') else []
                    for i, p in enumerate(price_row):
                        if not np.isnan(p) and i < len(dte_row):
                            seq_dtes.append(float(dte_row[i]) if not np.isnan(dte_row[i]) else np.nan)
            except (IndexError, KeyError):
                seq_dtes = [np.nan] * len(prices)

        # Check if we have enough valid contracts for a meaningful curve
        n_contracts = len(prices)
        min_contracts = 2  # Require at least 2 valid contracts
        if n_contracts < min_contracts:
            return None

        # Ensure all arrays have the same length as prices
        if len(volumes) < n_contracts:
            volumes.extend([np.nan] * (n_contracts - len(volumes)))
        if len(ois) < n_contracts:
            ois.extend([np.nan] * (n_contracts - len(ois)))
        if len(seq_dtes) < n_contracts:
            seq_dtes.extend([np.nan] * (n_contracts - len(seq_dtes)))
        if len(labels) < n_contracts:
            labels.extend([f'M{i}' for i in range(len(labels), n_contracts)])

        try:
            return FuturesCurve(
                symbol=self.symbol,
                ref_date=date,
                curve_month_labels=labels,
                prices=np.array(prices),
                volumes=np.array(volumes) if not all(np.isnan(volumes)) else None,
                open_interest=np.array(ois) if not all(np.isnan(ois)) else None,
                seq_prices=np.array(prices),
                seq_labels=labels,
                seq_volumes=np.array(volumes) if not all(np.isnan(volumes)) else None,
                seq_dte=np.array(seq_dtes) if not all(np.isnan(seq_dtes)) else None,
                days_to_expiry=np.array(seq_dtes) if not all(np.isnan(seq_dtes)) else None
            )
        except Exception as e:
            # If curve creation fails, return None
            return None
    
    def __getitem__(self, key):
        """
        Enhanced getitem method for flexible data access
        
        Supports:
        - Single character (F, G, H, etc): Returns Contract for that expiration month
        - Two character (M0, M1, M2, etc): Returns Contract from seq_data with continuous=True
        - datetime: Returns FuturesCurve for that date
        - hyphenated date string (YYYY-MM-DD): Returns FuturesCurve for that date
        - int: Returns FuturesCurve for that row index (supports negative indexing)
        - slice: Returns pd.Series of FuturesCurves over date range (supports datetime or date string bounds)
        """
        
        if isinstance(key, str):
            if len(key) == 1 and key in MONTH_CODE_ORDER:
                # Single character - return Contract object from contracts dict
                if hasattr(self, 'contracts') and key in self.contracts:
                    return self.contracts[key]
                else:
                    return self._get_contract_by_month(key)
            
            elif len(key) == 2 and key.startswith('M') and key[1:].isdigit():
                # Two character like M0, M1, M2 - return from seq_data
                contract_idx = int(key[1:])
                return self._get_seq_contract(contract_idx)
            
            elif self._is_date_string(key):
                # Hyphenated date string like '2023-01-15' - parse and return FuturesCurve
                parsed_date = self._parse_date_string(key)
                return self.create_futures_curve(parsed_date)
                
            else:
                raise KeyError(f"Invalid string key '{key}'. Use single month code (F,G,H...), Mx format (M0,M1,M2...), or date string (YYYY-MM-DD)")
                
        elif isinstance(key, datetime):
            # datetime - return FuturesCurve for that date
            return self.create_futures_curve(key)
            
        elif isinstance(key, int):
            # Integer index access
            if self.index is None or len(self.index) == 0:
                raise IndexError("No index data available")

            if key < 0:
                key = len(self.index) + key  # Handle negative indexing

            if key >= len(self.index) or key < 0:
                raise IndexError(f"Index {key} out of range")

            date = self.index[key]
            return self.create_futures_curve(date)
            
        elif isinstance(key, slice):
            # slice - return analysis over date range (supports datetime or date string bounds)
            return self._get_slice_analysis(key)
            
        else:
            raise TypeError(f"Invalid key type {type(key)}. Use str, int, datetime, or slice")
    
    def _is_date_string(self, key: str) -> bool:
        """Check if string is a valid hyphenated date format"""
        import re
        # Match patterns like YYYY-MM-DD, YYYY-M-D, or variations with time
        date_patterns = [
            r'^\d{4}-\d{1,2}-\d{1,2}$',              # YYYY-MM-DD
            r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}$',  # YYYY-MM-DD HH:MM
            r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}$'  # YYYY-MM-DD HH:MM:SS
        ]
        return any(re.match(pattern, key) for pattern in date_patterns)
    
    def _parse_date_string(self, date_str: str) -> datetime:
        """Parse hyphenated date string to datetime object"""
        from dateutil.parser import parse
        try:
            return parse(date_str)
        except Exception as e:
            raise ValueError(f"Unable to parse date string '{date_str}': {e}")
    
    def _convert_slice_bound_to_index(self, bound, bound_type: str) -> int:
        """Convert slice bound (string, datetime, int, or None) to index"""
        if bound is None:
            return 0 if bound_type == 'start' else len(self.index)
        
        if isinstance(bound, int):
            return bound
        
        elif isinstance(bound, str) and self._is_date_string(bound):
            # Parse date string to datetime and find index
            parsed_date = self._parse_date_string(bound)
            try:
                return self.index.get_indexer([parsed_date], method='nearest')[0]
            except (KeyError, IndexError):
                # If exact match not found, use nearest neighbor
                try:
                    return self.index.get_indexer([parsed_date], method='pad')[0]
                except (KeyError, IndexError):
                    if bound_type == 'start':
                        return 0
                    else:
                        return len(self.index)
        
        elif isinstance(bound, datetime):
            # Find index for datetime
            try:
                return self.index.get_indexer([bound], method='nearest')[0]
            except (KeyError, IndexError):
                try:
                    return self.index.get_indexer([bound], method='pad')[0]
                except (KeyError, IndexError):
                    if bound_type == 'start':
                        return 0
                    else:
                        return len(self.index)
        
        else:
            raise TypeError(f"Invalid slice bound type {type(bound)}. Use int, datetime, or date string")
    
    def _get_contract_by_month(self, month_code: str) -> Contract:
        """Get Contract object for specific expiration month"""
        if not hasattr(self, 'seq_data') or self.seq_data is None:
            raise ValueError("No sequential data available")
            
        if not hasattr(self.seq_data, 'seq_prices') or self.seq_data.seq_prices is None:
            raise ValueError("No price data available")
        
        # Find month code in available data
        month_idx = None
        if hasattr(self, 'seq_labels') and self.seq_labels is not None:
            # Look through sequential labels to find month position
            for i, labels_row in enumerate(self.seq_labels.values):
                if isinstance(labels_row, list) and month_code in labels_row:
                    month_idx = labels_row.index(month_code)
                    break
        
        if month_idx is None:
            # Fallback to calendar position
            if month_code in MONTH_CODE_ORDER:
                month_idx = MONTH_CODE_ORDER.index(month_code)
            else:
                raise KeyError(f"Month code '{month_code}' not found in data")
        
        # Extract time series for this contract
        if month_idx < self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices.shape[1]:
            contract_prices = self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices[:, month_idx]
            
            # Extract volumes and OI if available
            contract_volumes = None
            if self.seq_data.seq_volume is not None and self.seq_data.seq_volume.values if hasattr(self.seq_data.seq_volume, 'values') else self.seq_data.seq_volume is not None:
                if month_idx < self.seq_data.seq_volume.values if hasattr(self.seq_data.seq_volume, 'values') else self.seq_data.seq_volume.shape[1]:
                    contract_volumes = self.seq_data.seq_volume.values if hasattr(self.seq_data.seq_volume, 'values') else self.seq_data.seq_volume[:, month_idx]
            
            contract_oi = None
            if self.seq_data.seq_oi is not None and self.seq_data.seq_oi.values if hasattr(self.seq_data.seq_oi, 'values') else self.seq_data.seq_oi is not None:
                if month_idx < self.seq_data.seq_oi.values if hasattr(self.seq_data.seq_oi, 'values') else self.seq_data.seq_oi.shape[1]:
                    contract_oi = self.seq_data.seq_oi.values if hasattr(self.seq_data.seq_oi, 'values') else self.seq_data.seq_oi[:, month_idx]
            
            # Calculate days to expiry (simplified)
            dte_array = np.full(len(contract_prices), np.nan)
            
            return Contract(
                symbol=self.symbol,
                data=contract_prices,
                label=month_code,
                volume=contract_volumes if contract_volumes is not None else np.array([]),
                dte=dte_array,
                index=self.index,
                oi = contract_oi,
                direction="vertical"
            )
        else:
            raise KeyError(f"Contract position {month_idx} not available in data")
    
    def _get_seq_contract(self, contract_idx: int) -> Contract:
        """Get Contract object for sequential contract position (M0, M1, etc) with continuous=True"""
        if not hasattr(self, 'seq_data') or self.seq_data is None:
            raise ValueError("No sequential data available")
            
        if not hasattr(self.seq_data, 'seq_prices') or self.seq_data.seq_prices is None:
            raise ValueError("No price data available")
        
        if contract_idx >= self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices.shape[1]:
            raise IndexError(f"Contract index {contract_idx} exceeds available contracts ({self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices.shape[1]})")
        
        # Extract time series for this sequential position
        contract_prices = self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices[:, contract_idx]
        
        # Extract volume and oi if available
        contract_volume = np.array([])
        contract_oi = np.array([])
        contract_dte = np.array([])
        
        if (hasattr(self.seq_data, 'seq_volume') and self.seq_data.seq_volume is not None and 
            hasattr(self.seq_data.seq_volume, 'data') and self.seq_data.seq_volume.values if hasattr(self.seq_data.seq_volume, 'values') else self.seq_data.seq_volume is not None):
            if contract_idx < self.seq_data.seq_volume.values if hasattr(self.seq_data.seq_volume, 'values') else self.seq_data.seq_volume.shape[1]:
                contract_volume = self.seq_data.seq_volume.values if hasattr(self.seq_data.seq_volume, 'values') else self.seq_data.seq_volume[:, contract_idx]
                
        if (hasattr(self.seq_data, 'seq_oi') and self.seq_data.seq_oi is not None and 
            hasattr(self.seq_data.seq_oi, 'data') and self.seq_data.seq_oi.values if hasattr(self.seq_data.seq_oi, 'values') else self.seq_data.seq_oi is not None):
            if contract_idx < self.seq_data.seq_oi.values if hasattr(self.seq_data.seq_oi, 'values') else self.seq_data.seq_oi.shape[1]:
                contract_oi = self.seq_data.seq_oi.values if hasattr(self.seq_data.seq_oi, 'values') else self.seq_data.seq_oi[:, contract_idx]
                
        if (hasattr(self.seq_data, 'seq_dte') and self.seq_data.seq_dte is not None and 
            hasattr(self.seq_data.seq_dte, 'data') and self.seq_data.seq_dte is not None):
            if contract_idx < self.seq_data.seq_dte.shape[1]:
                contract_dte = self.seq_data.seq_dte[:, contract_idx]
        
        return Contract(
            symbol=self.symbol,
            data=contract_prices,
            label=f"M{contract_idx}",
            volume=contract_volume,
            oi=contract_oi,
            dte=contract_dte,
            index=self.index,
            continuous=True,  # Set continuous to True as requested
            is_front_month=(contract_idx == 0),
            direction="vertical",
            sequential=True
        )
    
    def _get_slice_analysis(self, key: slice) -> pd.Series:
        """Get pd.Series of FuturesCurve objects for date range slice"""
        if self.seq_prices is None:
            raise ValueError("No sequential data available")
        
        # Convert slice bounds to datetime indices if they are strings
        start_idx = self._convert_slice_bound_to_index(key.start, 'start')
        stop_idx = self._convert_slice_bound_to_index(key.stop, 'stop')
        step = key.step if key.step is not None else 1

        # Validate indices
        if start_idx < 0:
            start_idx = len(self.index) + start_idx
        if stop_idx < 0:
            stop_idx = len(self.index) + stop_idx
            
        start_idx = max(0, min(start_idx, len(self.index) - 1))
        stop_idx = max(0, min(stop_idx, len(self.index)))
        
        if start_idx >= stop_idx:
            raise ValueError(f"Invalid slice range: start={start_idx}, stop={stop_idx}")
        
        # Extract date range
        date_range = self.index[start_idx:stop_idx:step]

        # VECTORIZED CURVE CREATION (35-108x faster than original loop)
        curves, actual_dates = self._create_futures_curves_batch(date_range.tolist())

        # Use the actual dates pulled from seq_data for alignment
        # Fallback to the original date if no actual timestamp is available
        aligned_dates: List[pd.Timestamp] = []
        for requested_date, actual_date in zip(date_range, actual_dates):
            if pd.isna(actual_date):
                aligned_dates.append(pd.Timestamp(requested_date))
            else:
                aligned_dates.append(pd.Timestamp(actual_date))

        result_index = pd.DatetimeIndex(
            aligned_dates,
            name=getattr(self.index, 'name', None)
        )

        # Return pd.Series with datetime index and FuturesCurve values
        return pd.Series(curves, index=result_index, name='futures_curves')

    def _create_futures_curves_batch(self, dates_list: List[datetime]) -> Tuple[List[Optional[FuturesCurve]], List[pd.Timestamp]]:
        """
        VECTORIZED method to create multiple FuturesCurve objects efficiently

        This method is 35-108x faster than individual curve creation by using:
        - Vectorized date lookups with pandas get_indexer
        - Vectorized data extraction with numpy array slicing  
        - Vectorized NaN checking with boolean masks
        - Bulk processing instead of individual loops
        """
        
        if self.seq_prices is None:
            raise ValueError("No seq_prices available for batch curve creation")

        if len(dates_list) == 0:
            return [], []

        if self.index is None:
            raise ValueError("No timestamp data available for batch curve creation")

        # STEP 1: Vectorized date lookup (much faster than individual searches)
        dates_index = pd.DatetimeIndex(dates_list)
        timestamps = self.index
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)
        date_indices = timestamps.get_indexer(dates_index, method='nearest')
        date_indices = np.asarray(date_indices, dtype=int)

        # STEP 2: Vectorized data extraction with bounds validation
        if hasattr(self, 'seq_prices') and self.seq_prices is not None:
            seq_prices = self.seq_prices.values  # Get numpy array from DataFrame
        elif hasattr(self, 'seq_data') and hasattr(self.seq_data, 'seq_prices'):
            seq_prices = getattr(self.seq_data.seq_prices, 'data', None)
        else:
            seq_prices = None

        if seq_prices is None:
            raise ValueError("No sequential price data available")

        seq_prices_data = np.asarray(seq_prices)
        if seq_prices_data.ndim == 1:
            seq_prices_data = seq_prices_data[:, np.newaxis]

        n_rows, n_contracts = seq_prices_data.shape
        # Check bounds against actual price data length, not timestamps length
        valid_index_mask = (date_indices >= 0) & (date_indices < n_rows)

        price_rows = np.full((len(dates_list), n_contracts), np.nan, dtype=float)
        if np.any(valid_index_mask):
            price_rows_valid = seq_prices_data[date_indices[valid_index_mask]]
            price_rows[valid_index_mask] = price_rows_valid

        valid_masks = ~np.isnan(price_rows)

        if hasattr(self, 'seq_labels') and self.seq_labels is not None:
            seq_labels = self.seq_labels
        elif hasattr(self, 'seq_data') and hasattr(self.seq_data, 'seq_labels'):
            seq_labels = getattr(self.seq_data, 'seq_labels', None)
        else:
            seq_labels = None

        curves: List[Optional[FuturesCurve]] = []
        actual_dates: List[pd.Timestamp] = []

        # STEP 3: Batch process results (still need loop for object creation)
        for i, (date_idx, requested_date) in enumerate(zip(date_indices, dates_list)):
            if not valid_index_mask[i]:
                curves.append(None)
                actual_dates.append(pd.NaT)
                continue

            prices_row = price_rows[i]
            valid_mask = valid_masks[i]

            # Require minimum number of valid contracts for meaningful curve
            min_contracts = 2  # Require at least 2 valid contracts
            if not valid_mask.any() or valid_mask.sum() < min_contracts:
                actual_dates.append(pd.Timestamp(timestamps[date_idx]))
                curves.append(None)
                continue

            # Use boolean indexing to extract valid data
            valid_prices = prices_row[valid_mask]

            # Get labels efficiently while handling varying structures
            valid_labels: List[str] = []
            if seq_labels is not None and len(seq_labels) > date_idx:
                # Use .iloc for positional indexing if it's a DataFrame
                if hasattr(seq_labels, 'iloc'):
                    date_labels = seq_labels.iloc[date_idx]
                else:
                    date_labels = seq_labels[date_idx]
                if isinstance(date_labels, (list, tuple, np.ndarray)):
                    for label, include in zip(date_labels, valid_mask[:len(date_labels)]):
                        if include:
                            valid_labels.append(str(label))

            if len(valid_labels) < len(valid_prices):
                valid_labels.extend(
                    [f'M{j}' for j in range(len(valid_labels), len(valid_prices))]
                )

            actual_date = pd.Timestamp(timestamps[date_idx])

            # Create simplified curve data structure (can be upgraded to FuturesCurve later)
            curve_data = FuturesCurve(
                ref_date=actual_date,
                prices=valid_prices,
                seq_prices=valid_prices,
                curve_month_labels=valid_labels,
            )
            curves.append(curve_data)
            actual_dates.append(actual_date)

        return curves, actual_dates
    
    def get_seq_curves(self, 
                      date_range: Optional[Union[slice, pd.DatetimeIndex, List[datetime]]] = None,
                      step: int = 1) -> pd.Series:
        """
        Extract a pandas Series of FuturesCurves indexed by datetime for curve evolution analysis
        
        This method provides the primary interface for feeding curve data to CurveEvolution
        and path signature analysis for detecting drivers of curve evolution.
        
        Parameters:
        -----------
        date_range : slice, DatetimeIndex, or list of datetime, optional
            Range of dates to extract. If None, uses full available range
        step : int, default 1
            Step size for date sampling (e.g., step=5 for every 5th day)
            
        Returns:
        --------
        pd.Series
            DateTime-indexed Series with FuturesCurve objects as values
            
        Example:
        --------
        >>> spread_data = SpreadData('CL_F')
        >>> seq_curves = spread_data.get_seq_curves('2023-01-01':'2023-12-31')
        >>> # Feed to CurveEvolution for driver analysis
        >>> evolution = CurveEvolution(seq_curves)
        """
        
        if self.seq_prices is None:
            raise ValueError("No sequential data available for curve extraction")
            
        if self.index is None or len(self.index) == 0:
            raise ValueError("No index data available")
        
        # Handle different date_range types
        if date_range is None:
            # Use full range
            selected_dates = self.index[::step]
        elif isinstance(date_range, slice):
            # Convert slice to indices and apply step
            start_idx = self._convert_slice_bound_to_index(date_range.start, 'start')
            stop_idx = self._convert_slice_bound_to_index(date_range.stop, 'stop')
            selected_dates = self.index[start_idx:stop_idx:step]
        elif isinstance(date_range, (pd.DatetimeIndex, list)):
            # Use provided dates, applying step
            if isinstance(date_range, list):
                date_range = pd.DatetimeIndex(date_range)
            selected_dates = date_range[::step]
        else:
            raise TypeError(f"Invalid date_range type: {type(date_range)}")
        
        # BROADCAST: Generate FuturesCurve objects efficiently using vectorized operations
        curves_list = []
        valid_dates = []

        # Get all date indices at once (vectorized)
        date_indices = self.seq_prices.index.get_indexer(selected_dates, method='nearest')
        valid_mask = date_indices != -1

        # Filter to valid dates and indices
        valid_date_indices = date_indices[valid_mask]
        valid_selected_dates = selected_dates[valid_mask]

        if len(valid_date_indices) == 0:
            raise ValueError("No valid curves could be generated for the specified date range")

        # BROADCAST: Extract data for all dates at once
        prices_matrix = self.seq_prices.iloc[valid_date_indices].values  # (n_dates, n_contracts)

        # Get other data if available (broadcast)
        volumes_matrix = None
        if hasattr(self, 'seq_volume') and self.seq_volume is not None:
            volumes_matrix = self.seq_volume.iloc[valid_date_indices].values

        oi_matrix = None
        if hasattr(self, 'seq_oi') and self.seq_oi is not None:
            oi_matrix = self.seq_oi.iloc[valid_date_indices].values

        dte_matrix = None
        if hasattr(self, 'seq_dte') and self.seq_dte is not None:
            dte_matrix = self.seq_dte.iloc[valid_date_indices].values

        # Generate labels once (broadcast)
        seq_labels = [f'M{i}' for i in range(prices_matrix.shape[1])]

        # Create FuturesCurve objects for each date (still need loop but with pre-extracted data)
        for i, date in enumerate(valid_selected_dates):
            prices = prices_matrix[i]
            volumes = volumes_matrix[i] if volumes_matrix is not None else None
            ois = oi_matrix[i] if oi_matrix is not None else None
            seq_dtes = dte_matrix[i] if dte_matrix is not None else None

            # Create curve with pre-extracted data
            curve = FuturesCurve(
                symbol=self.symbol,
                ref_date=date,
                seq_prices=prices,
                seq_labels=seq_labels,
                seq_volumes=volumes,
                seq_dte=seq_dtes,
                sequential=True
            )

            curves_list.append(curve)
            valid_dates.append(date)
        
        # Return as pandas Series with datetime index
        return pd.Series(
            curves_list, 
            index=pd.DatetimeIndex(valid_dates),
            name='sequential_curves'
        )

    def _generate_seq_labels(self):
        """Generate sequential labels from curve columns"""
        labels_data = []
        index_data = _safe_get_index(self.curve)

        for idx in index_data:
            row_labels = []
            for month_code in MONTH_CODE_ORDER:
                if _safe_check_column(self.curve, month_code):
                    if pd.notna(_safe_get_value(self.curve, idx, month_code)):
                        row_labels.append(month_code)
            labels_data.append(row_labels)

        self.seq_labels = pd.DataFrame({'labels': labels_data}, index=index_data)

    def _calculate_seq_spreads(self):
        """Calculate sequential calendar spreads using pre-loaded seq_labels when available"""
        spreads_list = []
        index_data = _safe_get_index(self.curve)

        for idx in index_data:
            row_spreads = {}
            available_months = []

            # First try to use pre-loaded sequential labels to avoid redundant calculation
            seq_labels = None
            if not _is_empty(self.seq_labels) and idx in self.seq_labels.index:
                labels_row = self.seq_labels.loc[idx]
                if 'labels' in labels_row:
                    actual_labels = labels_row['labels']
                    if isinstance(actual_labels, list) and len(actual_labels) > 0:
                        seq_labels = actual_labels

            if seq_labels is not None:
                # Use pre-loaded sequence order
                for month_code in seq_labels:
                    if _safe_check_column(self.curve, month_code):
                        if pd.notna(_safe_get_value(self.curve, idx, month_code)):
                            available_months.append(month_code)
            else:
                # Fallback to standard calendar order
                for month_code in MONTH_CODE_ORDER:
                    if _safe_check_column(self.curve, month_code):
                        if pd.notna(_safe_get_value(self.curve, idx, month_code)):
                            available_months.append(month_code)

            # Calculate spreads (back - front)
            for i in range(len(available_months) - 1):
                front = available_months[i]
                back = available_months[i + 1]
                spread_name = f"{front}{back}"
                front_val = _safe_get_value(self.curve, idx, front)
                back_val = _safe_get_value(self.curve, idx, back)
                row_spreads[spread_name] = back_val - front_val

            spreads_list.append(row_spreads)

        self.seq_spreads = pd.DataFrame(spreads_list, index=index_data)

        # Calculate spread volumes/OI
        if not _is_empty(self.volume_curve):
            self._calculate_spread_volume()
        if not _is_empty(self.oi_curve):
            self._calculate_spread_oi()

    def _calculate_spread_volume(self):
        """Calculate volume for spreads (minimum of legs)"""
        volume_list = []
        vol_index = _safe_get_index(self.volume_curve)

        for idx in vol_index:
            row_volume = {}
            spread_cols = _safe_get_columns(self.seq_spreads)
            for spread_col in spread_cols:
                if len(str(spread_col)) == 2:
                    front, back = str(spread_col)[0], str(spread_col)[1]
                    if _safe_check_column(self.volume_curve, front) and _safe_check_column(self.volume_curve, back):
                        front_vol = _safe_get_value(self.volume_curve, idx, front)
                        back_vol = _safe_get_value(self.volume_curve, idx, back)
                        if pd.notna(front_vol) and pd.notna(back_vol):
                            row_volume[spread_col] = min(front_vol, back_vol)
            volume_list.append(row_volume)

        self.seq_volume = pd.DataFrame(volume_list, index=vol_index)

    def _calculate_spread_oi(self):
        """Calculate OI for spreads"""
        oi_list = []
        oi_index = _safe_get_index(self.oi_curve)

        for idx in oi_index:
            row_oi = {}
            spread_cols = _safe_get_columns(self.seq_spreads)
            for spread_col in spread_cols:
                if len(str(spread_col)) == 2:
                    front, back = str(spread_col)[0], str(spread_col)[1]
                    if _safe_check_column(self.oi_curve, front) and _safe_check_column(self.oi_curve, back):
                        front_oi = _safe_get_value(self.oi_curve, idx, front)
                        back_oi = _safe_get_value(self.oi_curve, idx, back)
                        if pd.notna(front_oi) and pd.notna(back_oi):
                            row_oi[spread_col] = min(front_oi, back_oi)
            oi_list.append(row_oi)

        self.seq_oi = pd.DataFrame(oi_list, index=oi_index)

    def calculate_maturity_bucket_spreads(self):
        """
        Calculate specific maturity bucket spreads using available month codes:
        - 6mo vs 3mo spread: ~6 month contract minus ~3 month contract
        - 10-12mo vs 6mo spread: ~10-12 month contracts minus ~6 month contract
        - 24mo vs 12mo spread: ~24 month contract minus ~12 month contract (when available)
        
        Uses actual available month codes from seq_labels to determine correct contracts
        rather than fixed indices, accommodating commodities with limited contract months.
        """
        if not hasattr(self, 'seq_data') or self.seq_data is None:
            raise ValueError("No sequential data available for maturity bucket calculations")
            
        if self.seq_data.seq_prices is None or self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices is None:
            raise ValueError("No sequential price data available")
        
        # Get available sequential labels if present
        if not hasattr(self, 'seq_labels') or self.seq_labels is None:
            raise ValueError("No seq_labels available for determining contract months")
            
        seq_prices_data = self.seq_prices.values
        n_dates, n_contracts = seq_prices_data.shape

        seq_dte_array = None
        if hasattr(self.seq_data, 'seq_dte') and self.seq_data.seq_dte is not None:
            try:
                seq_dte_array = np.asarray(self.seq_data.seq_dte, dtype=float)
                if seq_dte_array.ndim == 1:
                    seq_dte_array = np.expand_dims(seq_dte_array, axis=0)
                if seq_dte_array.shape[0] != n_dates and seq_dte_array.shape[1] == n_dates:
                    seq_dte_array = seq_dte_array.T
                if seq_dte_array.shape[0] != n_dates:
                    seq_dte_array = None
            except Exception:
                seq_dte_array = None

        # Initialize maturity bucket spreads array (6m-3m, 12m-6m, 24m-12m)
        maturity_spreads = np.full((n_dates, 3), np.nan)

        for i in range(n_dates):
            prices = seq_prices_data[i]
            dte_row = seq_dte_array[i] if seq_dte_array is not None and i < seq_dte_array.shape[0] else None

            # Get available month codes for this date
            available_months = []
            if hasattr(self.seq_labels, 'iloc') and i < len(self.seq_labels):
                # DataFrame format
                labels_row = self.seq_labels.iloc[i]
                if hasattr(labels_row, 'values') and len(labels_row.values) > 0:
                    if isinstance(labels_row.values[0], list):
                        available_months = labels_row.values[0]
                    else:
                        available_months = labels_row.values.tolist()
            elif isinstance(self.seq_labels, list) and i < len(self.seq_labels):
                # List format
                available_months = self.seq_labels[i]

            if not available_months:
                continue

            # Filter out NaN contracts and build valid month-price pairs
            valid_contracts = []
            month_occurrences: Dict[str, int] = {}
            for j, month_code in enumerate(available_months[:n_contracts]):
                if j >= len(prices) or np.isnan(prices[j]):
                    continue

                base_code = normalize_contract_code(month_code)
                month_num = MONTH_CODE_MAP.get(base_code, None)

                dte_value = None
                if dte_row is not None and j < len(dte_row):
                    dte_value = dte_row[j]

                occurrence = 0
                if base_code:
                    occurrence = month_occurrences.get(base_code, 0)
                    month_occurrences[base_code] = occurrence + 1

                approx_months = self._estimate_months_to_expiry(
                    month_code,
                    month_num,
                    dte_value=dte_value,
                    occurrence=occurrence
                )

                if approx_months is None:
                    continue

                valid_contracts.append({
                    'month_code': month_code,
                    'price': prices[j],
                    'seq_index': j,
                    'month_num': month_num,
                    'approx_months_out': approx_months,
                    'dte': dte_value
                })

            if len(valid_contracts) < 2:
                continue

            # Sort by approximate months to expiry
            valid_contracts.sort(key=lambda x: x['approx_months_out'])

            # Find contracts closest to target maturities
            three_month_contract = self._find_closest_maturity_contract(valid_contracts, target_months=3)
            six_month_contract = self._find_closest_maturity_contract(valid_contracts, target_months=6)
            twelve_month_contract = self._find_closest_maturity_contract(valid_contracts, target_months=12)
            two_year_contract = self._find_closest_maturity_contract(valid_contracts, target_months=24)
            long_contracts = self._find_long_maturity_contracts(valid_contracts, min_months=9)

            # Calculate 6mo vs 3mo spread
            if three_month_contract and six_month_contract:
                maturity_spreads[i, 0] = six_month_contract['price'] - three_month_contract['price']

            # Calculate 10-12mo vs 6mo spread
            if long_contracts and six_month_contract:
                avg_long_price = np.mean([c['price'] for c in long_contracts])
                maturity_spreads[i, 1] = avg_long_price - six_month_contract['price']

            # Calculate ~24m vs 12m spread when sufficiently far contracts exist
            if two_year_contract and twelve_month_contract:
                if (two_year_contract['approx_months_out'] >= 18 and
                        two_year_contract['approx_months_out'] > twelve_month_contract['approx_months_out']):
                    maturity_spreads[i, 2] = (two_year_contract['price'] -
                                              twelve_month_contract['price'])

        # Create SpreadFeature objects for the calculated spreads
        bucket_6m_3m = SpreadFeature(
            data=maturity_spreads[:, 0],
            direction="vertical",
            index=self.index,
            labels=["6m_vs_3m"]
        )
        
        bucket_12m_6m = SpreadFeature(
            data=maturity_spreads[:, 1],
            direction="vertical",
            index=self.index,
            labels=["12m_vs_6m"]
        )

        bucket_24m_12m = SpreadFeature(
            data=maturity_spreads[:, 2],
            direction="vertical",
            index=self.index,
            labels=["24m_vs_12m"]
        )

        # Store as attributes
        self.maturity_6m_3m_spread = bucket_6m_3m
        self.maturity_12m_6m_spread = bucket_12m_6m
        self.maturity_24m_12m_spread = bucket_24m_12m

        return {
            '6m_vs_3m': bucket_6m_3m,
            '12m_vs_6m': bucket_12m_6m,
            '24m_vs_12m': bucket_24m_12m
        }

    def _estimate_months_to_expiry(self,
                                   month_code: str,
                                   month_num: Optional[int],
                                   dte_value: Optional[float] = None,
                                   occurrence: int = 0) -> Optional[float]:
        """
        Estimate approximate months to expiry based on month code.
        This is a rough approximation for sorting contracts by maturity.
        """
        months_from_dte = dte_to_months(dte_value)
        if months_from_dte is not None:
            return months_from_dte

        base_code = normalize_contract_code(month_code)
        if month_num is None and base_code in MONTH_CODE_MAP:
            month_num = MONTH_CODE_MAP[base_code]

        if month_num is None:
            return None

        if not hasattr(self, 'index') or len(self.index) == 0:
            return float(month_num + occurrence * 12)

        current_month = self.index[0].month if hasattr(self.index[0], 'month') else 1
        months_forward = month_num - current_month
        if months_forward <= 0:
            months_forward += 12

        months_forward += occurrence * 12
        return float(months_forward)
    
    def _find_closest_maturity_contract(self, contracts: list, target_months: int) -> dict:
        """Find the contract closest to the target maturity in months."""
        if not contracts:
            return None
            
        best_contract = None
        min_distance = float('inf')
        
        for contract in contracts:
            distance = abs(contract['approx_months_out'] - target_months)
            if distance < min_distance:
                min_distance = distance
                best_contract = contract
                
        return best_contract
    
    def _find_long_maturity_contracts(self, contracts: list, min_months: int) -> list:
        """Find contracts with maturity >= min_months for averaging."""
        return [c for c in contracts if c['approx_months_out'] >= min_months]

    def get_maturity_bucket_spreads(self, as_dataframe: bool = False):
        """
        Get maturity bucket spreads as SpreadFeature objects or DataFrame.
        
        Parameters:
        -----------
        as_dataframe : bool, default False
            If True, returns spreads as a pandas DataFrame with columns ['6m_vs_3m', '12m_vs_6m']
            If False, returns dictionary with SpreadFeature objects
            
        Returns:
        --------
        dict or pd.DataFrame
            Maturity bucket spreads indexed by datetime
        """
        if (not hasattr(self, 'maturity_6m_3m_spread') or
                not hasattr(self, 'maturity_12m_6m_spread') or
                not hasattr(self, 'maturity_24m_12m_spread')):
            # Calculate if not already done
            self.calculate_maturity_bucket_spreads()

        if as_dataframe:
            return pd.DataFrame({
                '6m_vs_3m': self.maturity_6m_3m_spread.data,
                '12m_vs_6m': self.maturity_12m_6m_spread.data,
                '24m_vs_12m': self.maturity_24m_12m_spread.data
            }, index=self.index)
        else:
            return {
                '6m_vs_3m': self.maturity_6m_3m_spread,
                '12m_vs_6m': self.maturity_12m_6m_spread,
                '24m_vs_12m': self.maturity_24m_12m_spread
            }
    
    def get_maturity_bucket_contracts_info(self, date_index: int = 0) -> dict:
        """
        Get information about which specific contracts were used for maturity bucket spreads
        on a given date. Useful for validation and debugging.
        
        Parameters:
        -----------
        date_index : int, default 0
            Index of the date to analyze (0 = first date)
            
        Returns:
        --------
        dict
            Information about selected contracts including month codes, 
            prices, and estimated months to expiry
        """
        if not hasattr(self, 'seq_labels') or self.seq_labels is None:
            return {'error': 'No seq_labels available'}
        
        if not hasattr(self, 'seq_data') or self.seq_data.seq_prices is None:
            return {'error': 'No sequential price data available'}
            
        if date_index >= len(self.index):
            return {'error': f'Date index {date_index} out of range (max: {len(self.index)-1})'}
        
        # Prepare DTE information if available
        seq_dte_array = None
        if hasattr(self.seq_data, 'seq_dte') and self.seq_data.seq_dte is not None:
            try:
                seq_dte_array = np.asarray(self.seq_data.seq_dte, dtype=float)
                if seq_dte_array.ndim == 1:
                    seq_dte_array = np.expand_dims(seq_dte_array, axis=0)
                if seq_dte_array.shape[0] != self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices.shape[0] and \
                        seq_dte_array.shape[1] == self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices.shape[0]:
                    seq_dte_array = seq_dte_array.T
                if seq_dte_array.shape[0] != self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices.shape[0]:
                    seq_dte_array = None
            except Exception:
                seq_dte_array = None

        # Get data for specified date
        prices = self.seq_data.seq_prices.values if hasattr(self.seq_data.seq_prices, 'values') else self.seq_data.seq_prices[date_index]
        dte_row = None
        if seq_dte_array is not None and date_index < seq_dte_array.shape[0]:
            dte_row = seq_dte_array[date_index]

        # Get available month codes for this date
        available_months = []
        if hasattr(self.seq_labels, 'iloc'):
            labels_row = self.seq_labels.iloc[date_index]
            if hasattr(labels_row, 'values') and len(labels_row.values) > 0:
                if isinstance(labels_row.values[0], list):
                    available_months = labels_row.values[0]
                else:
                    available_months = labels_row.values.tolist()
        elif isinstance(self.seq_labels, list) and date_index < len(self.seq_labels):
            available_months = self.seq_labels[date_index]

        # Build valid contracts info
        valid_contracts = []
        month_occurrences: Dict[str, int] = {}
        for j, month_code in enumerate(available_months):
            if j >= len(prices) or np.isnan(prices[j]):
                continue

            base_code = normalize_contract_code(month_code)
            month_num = MONTH_CODE_MAP.get(base_code, None)

            dte_value = None
            if dte_row is not None and j < len(dte_row):
                dte_value = dte_row[j]

            occurrence = 0
            if base_code:
                occurrence = month_occurrences.get(base_code, 0)
                month_occurrences[base_code] = occurrence + 1

            approx_months = self._estimate_months_to_expiry(
                month_code,
                month_num,
                dte_value=dte_value,
                occurrence=occurrence
            )

            if approx_months is None:
                continue

            valid_contracts.append({
                'month_code': month_code,
                'price': float(prices[j]),
                'seq_index': j,
                'month_num': month_num,
                'approx_months_out': approx_months,
                'dte': dte_value
            })

        if len(valid_contracts) < 2:
            return {'error': 'Insufficient valid contracts for analysis'}

        # Sort by approximate months to expiry
        valid_contracts.sort(key=lambda x: x['approx_months_out'])

        # Find selected contracts
        three_month_contract = self._find_closest_maturity_contract(valid_contracts, target_months=3)
        six_month_contract = self._find_closest_maturity_contract(valid_contracts, target_months=6)
        twelve_month_contract = self._find_closest_maturity_contract(valid_contracts, target_months=12)
        two_year_contract = self._find_closest_maturity_contract(valid_contracts, target_months=24)
        long_contracts = self._find_long_maturity_contracts(valid_contracts, min_months=9)

        result = {
            'analysis_date': self.index[date_index],
            'available_contracts': valid_contracts,
            'selected_contracts': {
                '3_month_target': three_month_contract,
                '6_month_target': six_month_contract,
                '12_month_target': twelve_month_contract,
                '24_month_target': two_year_contract,
                'long_contracts': long_contracts
            },
            'calculated_spreads': {}
        }

        # Calculate the spreads that would be computed
        if three_month_contract and six_month_contract:
            result['calculated_spreads']['6m_vs_3m'] = (
                six_month_contract['price'] - three_month_contract['price']
            )

        if long_contracts and six_month_contract:
            avg_long_price = np.mean([c['price'] for c in long_contracts])
            result['calculated_spreads']['12m_vs_6m'] = avg_long_price - six_month_contract['price']

        if two_year_contract and twelve_month_contract:
            if (two_year_contract['approx_months_out'] >= 18 and
                    two_year_contract['approx_months_out'] > twelve_month_contract['approx_months_out']):
                result['calculated_spreads']['24m_vs_12m'] = (
                    two_year_contract['price'] - twelve_month_contract['price']
                )

        return result

    def load_from_client(self):
        """
        Load all available curve data types for this symbol (legacy method).

        For more control over data loading, use the class method load_from_client_filtered().
        """
        # Use the class method to get a new instance with all data
        filtered_data = self.load_from_client_filtered(
            symbol=self.symbol,
            start_date=None,
            end_date=None,
            curve_types=None
        )

        # Copy the data from the filtered instance to this instance
        for attr_name in ['curve', 'volume_curve', 'oi_curve', 'seq_prices', 'seq_volume',
                         'seq_oi', 'seq_labels', 'seq_spreads', 'seq_dte', 'dte',
                         'days_to_expiration', 'spot_prices', 'synthetic_spot_prices', 'roll_dates',
                         'index', 'timestamps']:
            if hasattr(filtered_data, attr_name):
                setattr(self, attr_name, getattr(filtered_data, attr_name))

        # Ensure index is set (fallback to seq_prices index if available)
        if self.index is None and hasattr(self, 'seq_prices') and isinstance(self.seq_prices, pd.DataFrame):
            self.index = self.seq_prices.index
        elif self.index is None and hasattr(self, 'curve') and isinstance(self.curve, pd.DataFrame):
            self.index = self.curve.index

        # Ensure timestamps matches index
        if self.index is not None and self.timestamps is None:
            self.timestamps = self.index

        return self

    @classmethod
    def load_from_client_filtered(
        cls,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        curve_types: Optional[Union[str, List[str]]] = None
    ) -> 'SpreadData':
        """
        Class method to create SpreadData instance with filtered data from DataClient.

        Parameters:
        -----------
        symbol : str
            Symbol to load (e.g., 'CL_F', 'NG_F')
        start_date : str, optional
            Start date for data filtering (e.g., '2020-01-01')
        end_date : str, optional
            End date for data filtering (e.g., '2023-12-31')
        curve_types : str, list of str, or None
            Specific curve types to load. If None, loads all available types.
            Available types: 'curve', 'volume_curve', 'oi_curve', 'seq_curve',
            'seq_volume', 'seq_oi', 'seq_labels', 'seq_spreads', 'dte', 'seq_dte', 'spot'

        Returns:
        --------
        SpreadData
            New SpreadData instance with filtered data

        Examples:
        ---------
        # Load recent NG data with only sequential types
        ng_data = SpreadData.load_from_client_filtered(
            'NG_F',
            start_date='2020-01-01',
            curve_types=['seq_curve', 'seq_spreads', 'seq_labels', 'seq_dte']
        )

        # Load CL data for specific year
        cl_data = SpreadData.load_from_client_filtered(
            'CL_F',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        """
        cli = DataClient()

        # Default curve types if none specified
        if curve_types is None:
            all_curve_types = [
                'curve', 'volume_curve', 'oi_curve',
                'seq_curve', 'seq_volume', 'seq_oi', 'seq_labels', 'seq_spreads',
                'dte', 'seq_dte', 'spot'
            ]
        else:
            # Ensure curve_types is a list
            if isinstance(curve_types, str):
                all_curve_types = [curve_types]
            else:
                all_curve_types = list(curve_types)

        # Add roll_dates to curve types if not already included
        if 'roll_dates' not in all_curve_types:
            all_curve_types.append('roll_dates')

        # Query all data in single HDF5 transaction (including roll_dates)
        curve_data = cli.query_curve_data(
            symbol,
            curve_types=all_curve_types,
            start_date=start_date,
            end_date=end_date
        )

        # Create new SpreadData instance
        spread_data = cls(symbol=symbol)

        # Extract roll_dates from unified query result
        if isinstance(curve_data, dict) and 'roll_dates' in curve_data:
            spread_data.roll_dates = curve_data['roll_dates']
        else:
            spread_data.roll_dates = None

        # Extract data using existing methods
        spread_data._extract_curve_data(curve_data)
        spread_data._extract_scalar_data(curve_data)
        spread_data._extract_sequential_data(curve_data)

        return spread_data

    def _extract_scalar_data(self, curve_data):
        """
        Efficiently extract scalar arrays (dte, seq_dte, spot) from curve_data.

        Maps HDF5 keys to SpreadData attributes:
        - 'spot' key → self.spot_prices (pandas Series)
        - 'dte' key → self.dte (numpy array)
        - 'seq_dte' key → self.seq_dte (DataFrame/array)
        """
        # Extract dte data
        if 'dte' in curve_data and curve_data['dte'] is not None:
            dte_data = curve_data['dte']
            self.days_to_expiration = self.dte = dte_data.values if hasattr(dte_data, 'values') else dte_data
        else:
            self.dte = self.days_to_expiration = None

        # Extract seq_dte data
        if 'seq_dte' in curve_data and curve_data['seq_dte'] is not None:
            seq_dte_data = curve_data['seq_dte']
            self.seq_dte = seq_dte_data.copy() if isinstance(seq_dte_data, pd.DataFrame) else seq_dte_data
        else:
            self.seq_dte = None

        # Extract real spot prices as Series with index intersection (keep synthetic separate)
        # Maps HDF5 '/spot' key to self.spot_prices attribute as pandas Series
        if 'spot' in curve_data and curve_data['spot'] is not None:
            spot_data = curve_data['spot']
            # Convert to Series if not already
            if isinstance(spot_data, pd.Series):
                spot_series = spot_data
            elif hasattr(spot_data, 'values') and hasattr(spot_data, 'index'):
                # DataFrame case - take first column if multiple
                if isinstance(spot_data, pd.DataFrame):
                    spot_series = spot_data.iloc[:, 0] if len(spot_data.columns) > 0 else pd.Series(dtype=float)
                else:
                    spot_series = pd.Series(spot_data.values, index=spot_data.index)
            else:
                # Raw array case (numpy array or list) - convert to Series with proper index
                if hasattr(self, 'index') and self.index is not None:
                    # Ensure array length matches index length
                    spot_array = np.asarray(spot_data)
                    if len(spot_array) == len(self.index):
                        spot_series = pd.Series(spot_array, index=self.index, name='spot')
                    elif len(spot_array) < len(self.index):
                        # Truncate index to match array length
                        spot_series = pd.Series(spot_array, index=self.index[:len(spot_array)], name='spot')
                    else:
                        # Truncate array to match index length
                        spot_series = pd.Series(spot_array[:len(self.index)], index=self.index, name='spot')
                else:
                    # No index available - create default datetime index
                    spot_array = np.asarray(spot_data)
                    default_index = pd.date_range(start='2020-01-01', periods=len(spot_array), freq='D')
                    spot_series = pd.Series(spot_array, index=default_index, name='spot')
            
            # Intersect with futures index if available
            if hasattr(self, 'index') and self.index is not None and len(spot_series) > 0:
                # Only keep spot prices that intersect with futures index dates
                intersecting_dates = spot_series.index.intersection(self.index)
                if len(intersecting_dates) > 0:
                    self.spot_prices = spot_series.loc[intersecting_dates]
                else:
                    self.spot_prices = None
            else:
                self.spot_prices = spot_series
        else:
            self.spot_prices = None

    def _extract_curve_data(self, curve_data):
        """Efficiently extract main curve data and create Contract objects in batch"""
        if 'curve' not in curve_data or not isinstance(curve_data['curve'], pd.DataFrame):
            return
        
        curve_df = curve_data['curve']
        self.curve = curve_df

        # Use seq_curve index if available (more accurate than curve index)
        if 'seq_curve' in curve_data and isinstance(curve_data['seq_curve'], pd.DataFrame):
            self.index = curve_data['seq_curve'].index
        else:
            self.index = curve_df.index
        
        # Batch create Contract objects with pre-extracted dte data
        contract_columns = curve_df.columns
        contract_data = curve_df.values  # Single conversion to numpy
        
        # Pre-extract dte data for all contracts if available
        dte_matrix = None
        if self.dte is not None:
            if hasattr(self.dte, 'values') and hasattr(self.dte, 'columns'):
                # DataFrame case: extract matching columns
                dte_columns = self.dte.columns
                dte_matrix = np.full((len(self.dte), len(contract_columns)), np.nan)
                for i, col in enumerate(contract_columns):
                    if col in dte_columns:
                        dte_matrix[:, i] = self.dte[col].values
            elif isinstance(self.dte, np.ndarray):
                # Array case: use direct indexing
                dte_matrix = self.dte if len(self.dte.shape) > 1 else self.dte.reshape(-1, 1)
        
        # Batch create all Contract objects
        self.contracts = {}
        for i, col in enumerate(contract_columns):
            contract_dte = dte_matrix[:, i] if dte_matrix is not None and i < dte_matrix.shape[1] else np.array([])
            self.contracts[col] = Contract(
                symbol=self.symbol,
                data=contract_data[:, i],
                label=col,
                index=self.index,
                dte=contract_dte
            )

    def _extract_sequential_data(self, curve_data):
        """Efficiently extract sequential data types"""
        seq_mapping = {
            'seq_curve': 'seq_prices',  # Map seq_curve to seq_prices for consistency
            'seq_prices': 'seq_prices',
            'seq_spreads': 'seq_spreads',
            'seq_oi': 'seq_oi',
            'seq_volume': 'seq_volume',
            'seq_labels': 'seq_labels',
            'seq_dte': 'seq_dte'
        }
        
        for curve_key, attr_name in seq_mapping.items():
            if curve_key in curve_data and isinstance(curve_data[curve_key], pd.DataFrame):
                setattr(self, attr_name, curve_data[curve_key])
        
        # Handle other DataFrame types (skip 'spot' and 'roll_dates' - handled separately)
        for k, val in curve_data.items():
            if (isinstance(val, pd.DataFrame) and
                k not in {'curve', 'seq_curve', 'seq_prices', 'seq_spreads', 'seq_oi', 'seq_volume', 'seq_labels', 'seq_dte', 'spot', 'roll_dates'}):
                setattr(self, k, val.values if hasattr(val, 'values') else val)

    def to_dataframe(
        self,
        columns: Optional[List[str]] = None,
        include_labels: bool = True
    ) -> pd.DataFrame:
        """Return a concatenated DataFrame of sequential features."""

        columns = columns or ['prices', 'spreads', 'volume', 'oi']
        if isinstance(columns, str):
            columns = [columns]

        feature_map = {
            'prices': self._get_seq_feature('seq_prices'),
            'spreads': self._get_seq_feature('seq_spreads'),
            'volume': self._get_seq_feature('seq_volume'),
            'oi': self._get_seq_feature('seq_oi')
        }

        frames: List[pd.DataFrame] = []
        for key in columns:
            feature = feature_map.get(key)
            if feature is None:
                continue
            df = self._feature_to_dataframe(feature)
            if df is None:
                continue
            frames.append(df.add_prefix(f'{key}_'))

        if include_labels and isinstance(self.seq_labels, pd.DataFrame):
            frames.append(self.seq_labels.add_prefix('label_'))

        if not frames:
            return pd.DataFrame(index=self.index)

        return pd.concat(frames, axis=1)

    def plot_spread(self, 
                   contracts: Optional[List[str]] = None,
                   plot_type: str = 'time_series',
                   date_range: Optional[Union[slice, str]] = None,
                   height: int = 600,
                   title: Optional[str] = None) -> 'go.Figure':
        """
        Plot spread analysis with selectable contracts
        
        Parameters:
        -----------
        contracts : List[str], optional
            List of contracts to plot. Can be:
            - Month codes: ['F', 'G', 'H', 'J', 'K']
            - Sequential codes: ['M0', 'M1', 'M2', 'M3']
            - Mix of both: ['F', 'M1', 'H']
            If None, defaults to ['M0', 'M1', 'M2', 'M3'] (first 4 contracts)
        plot_type : str, default 'time_series'
            Type of plot: 'time_series', 'curve_snapshots', 'spreads'
        date_range : slice or str, optional
            Date range to plot (e.g., '2023-01-01':'2023-12-31')
        height : int, default 600
            Plot height in pixels
        title : str, optional
            Custom plot title
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        
        # Import plotly here to avoid circular imports
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly")
        
        # Default contracts if none specified
        if contracts is None:
            contracts = ['M0', 'M1', 'M2', 'M3']
        
        # Handle date range
        if date_range is None:
            plot_dates = self.index
        else:
            # Convert string to slice if needed
            if isinstance(date_range, str):
                if ':' in date_range:
                    start, end = date_range.split(':')
                    date_range = slice(start, end)
                else:
                    # Single date
                    plot_dates = [pd.to_datetime(date_range)]
                    date_range = None  # Mark as handled
            
            # Handle different date_range types
            if isinstance(date_range, pd.DatetimeIndex):
                # Handle pandas DatetimeIndex (e.g., from pd.date_range)
                plot_dates = date_range
                
            elif isinstance(date_range, slice):
                # Efficient date range extraction without creating FuturesCurve objects
                start_idx = self._convert_slice_bound_to_index(date_range.start, 'start')
                stop_idx = self._convert_slice_bound_to_index(date_range.stop, 'stop')
                step = date_range.step if date_range.step is not None else 1
                
                # Validate and clamp indices
                start_idx = max(0, min(start_idx, len(self.index) - 1))
                stop_idx = max(0, min(stop_idx, len(self.index)))
                
                if start_idx >= stop_idx:
                    plot_dates = self.index  # Fallback to full range
                else:
                    plot_dates = self.index[start_idx:stop_idx:step]
                    
            elif date_range is not None:
                # Other types - fallback to full range
                plot_dates = self.index
        
        if plot_type == 'time_series':
            return self._plot_time_series(contracts, plot_dates, height, title)
        elif plot_type == 'curve_snapshots':
            return self._plot_curve_snapshots(contracts, plot_dates, height, title)
        elif plot_type == 'spreads':
            return self._plot_spread_analysis(contracts, plot_dates, height, title)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use 'time_series', 'curve_snapshots', or 'spreads'")
    
    def _plot_time_series(self, contracts: List[str], dates: pd.DatetimeIndex, height: int, title: Optional[str]) -> 'go.Figure':
        """Plot individual contract time series"""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for contract in contracts:
            try:
                # Get price data for this contract using the fixed helper method
                prices = self._get_contract_prices(contract, dates)
                
                if len(prices) > 0:
                    valid_mask = ~np.isnan(prices)
                    if np.sum(valid_mask) > 0:
                        # Ensure dates and prices are aligned
                        if len(dates) == len(prices):
                            plot_dates = dates[valid_mask]
                            plot_prices = prices[valid_mask]
                        else:
                            # If lengths don't match, use minimum length
                            min_len = min(len(dates), len(prices))
                            plot_dates = dates[:min_len][valid_mask[:min_len]]
                            plot_prices = prices[:min_len][valid_mask[:min_len]]
                        
                        fig.add_trace(go.Scatter(
                            x=plot_dates,
                            y=plot_prices,
                            mode='lines',
                            name=f'{contract} Contract',
                            hovertemplate=f'{contract}: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
                        ))
                        
            except Exception as e:
                print(f"Warning: Could not plot contract {contract}: {e}")
                continue
        
        symbol_name = self.symbol if hasattr(self, 'symbol') and self.symbol else "Futures"
        default_title = f'{symbol_name} Contract Time Series'
        
        fig.update_layout(
            title=title or default_title,
            height=height,
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def _plot_curve_snapshots(self, contracts: List[str], dates: pd.DatetimeIndex, height: int, title: Optional[str]) -> 'go.Figure':
        """Plot curve snapshots at different time periods"""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Sample a few dates for curve snapshots
        snapshot_dates = dates[::max(1, len(dates)//5)]  # Take ~5 snapshots
        
        for i, snapshot_date in enumerate(snapshot_dates):
            try:
                curve_data = self[snapshot_date]
                if hasattr(curve_data, 'prices'):
                    prices = []
                    labels = []
                    
                    for contract in contracts:
                        try:
                            if hasattr(curve_data, '__getitem__'):
                                price = curve_data[contract]
                                if price is not None and not np.isnan(price):
                                    prices.append(price)
                                    labels.append(contract)
                        except (KeyError, IndexError):
                            continue
                    
                    if len(prices) > 1:
                        fig.add_trace(go.Scatter(
                            x=labels,
                            y=prices,
                            mode='lines+markers',
                            name=f'{snapshot_date.strftime("%Y-%m-%d")}',
                            hovertemplate='Contract: %{x}<br>Price: %{y:.2f}<br>Date: ' + snapshot_date.strftime("%Y-%m-%d") + '<extra></extra>'
                        ))
                        
            except Exception as e:
                continue
        
        symbol_name = self.symbol if hasattr(self, 'symbol') and self.symbol else "Futures"
        default_title = f'{symbol_name} Curve Snapshots'
        
        fig.update_layout(
            title=title or default_title,
            height=height,
            xaxis_title="Contract",
            yaxis_title="Price",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def _plot_spread_analysis(self, contracts: List[str], dates: pd.DatetimeIndex, height: int, title: Optional[str]) -> 'go.Figure':
        """Plot spread analysis between selected contracts"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if len(contracts) < 2:
            raise ValueError("Need at least 2 contracts for spread analysis")
        
        # Create subplots for different spreads
        n_spreads = len(contracts) - 1
        fig = make_subplots(
            rows=n_spreads, cols=1,
            subplot_titles=[f'{contracts[i+1]} - {contracts[i]} Spread' for i in range(n_spreads)],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        for i in range(n_spreads):
            front_contract = contracts[i]
            back_contract = contracts[i+1]
            
            try:
                # Calculate spread time series
                front_prices = self._get_contract_prices(front_contract, dates)
                back_prices = self._get_contract_prices(back_contract, dates)
                
                if len(front_prices) > 0 and len(back_prices) > 0:
                    # Align prices
                    min_len = min(len(front_prices), len(back_prices))
                    spread = back_prices[:min_len] - front_prices[:min_len]
                    spread_dates = dates[:min_len]
                    
                    # Remove NaN values
                    valid_mask = ~np.isnan(spread)
                    
                    if np.sum(valid_mask) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=spread_dates[valid_mask],
                                y=spread[valid_mask],
                                mode='lines',
                                name=f'{back_contract}-{front_contract}',
                                hovertemplate=f'Spread: %{{y:.2f}}<br>Date: %{{x}}<br>State: %{{text}}<extra></extra>',
                                text=['Contango' if s > 0 else 'Backwardation' for s in spread[valid_mask]]
                            ),
                            row=i+1, col=1
                        )
                        
                        # Add zero line
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=i+1, col=1)
                        
            except Exception as e:
                print(f"Warning: Could not calculate spread {back_contract}-{front_contract}: {e}")
                continue
        
        symbol_name = self.symbol if hasattr(self, 'symbol') and self.symbol else "Futures"
        default_title = f'{symbol_name} Spread Analysis'
        
        fig.update_layout(
            title=title or default_title,
            height=height,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update y-axis labels
        for i in range(n_spreads):
            fig.update_yaxes(title_text="Spread", row=i+1, col=1)
        
        fig.update_xaxes(title_text="Date", row=n_spreads, col=1)
        
        return fig
    
    def _get_contract_prices(self, contract: str, dates: pd.DatetimeIndex) -> np.ndarray:
        """Helper method to extract price data for a specific contract using Contract objects"""
        
        try:
            # Use SpreadData's __getitem__ method to get the Contract object
            contract_obj = self[contract]
            
            if contract_obj is not None and hasattr(contract_obj, 'get_prices'):
                # Use the Contract's get_prices method with date filtering
                return contract_obj.get_prices(dates)
                
        except Exception as e:
            print(f"Warning: Error getting prices for contract {contract}: {e}")
        
        return np.array([])
    
    def get_roll_events(self, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       roll_pattern: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get roll events for this symbol with optional filtering.
        
        Parameters:
        -----------
        start_date : str, optional
            Start date filter in 'YYYY-MM-DD' format
        end_date : str, optional
            End date filter in 'YYYY-MM-DD' format  
        roll_pattern : str, optional
            Filter by contract transition pattern (e.g., 'H->J')
            
        Returns:
        --------
        pd.DataFrame or None
            Filtered roll events DataFrame, None if no roll_dates available
        """
        if self.roll_dates is None:
            return None
        
        filtered_df = self.roll_dates.copy()
        
        # Apply date filters
        if start_date:
            filtered_df = filtered_df[filtered_df.index >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df.index <= end_date]
        
        # Apply roll pattern filter
        if roll_pattern and '->' in roll_pattern:
            from_contract, to_contract = roll_pattern.split('->')
            from_contract = from_contract.strip()
            to_contract = to_contract.strip()
            
            filtered_df = filtered_df[
                (filtered_df['from_contract_expiration_code'] == from_contract) &
                (filtered_df['to_contract_expiration_code'] == to_contract)
            ]
        
        return filtered_df if len(filtered_df) > 0 else None
    
    def get_next_roll_date(self, current_date: Union[str, pd.Timestamp]) -> Optional[pd.Timestamp]:
        """
        Get the next roll date after the specified date.
        
        Parameters:
        -----------
        current_date : str or pd.Timestamp
            Reference date to find next roll after
            
        Returns:
        --------
        pd.Timestamp or None
            Next roll date, None if no roll_dates available or no future rolls
        """
        if self.roll_dates is None:
            return None
        
        current_date = pd.to_datetime(current_date)
        future_rolls = self.roll_dates[self.roll_dates.index > current_date]
        
        return future_rolls.index.min() if len(future_rolls) > 0 else None
    
    def get_roll_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive roll statistics for this symbol.
        
        Returns:
        --------
        Dict[str, Any] or None
            Roll statistics including timing, patterns, confidence metrics
        """
        if self.roll_dates is None:
            return None
        
        return {
            'total_rolls': len(self.roll_dates),
            'date_range': {
                'start': self.roll_dates.index.min(),
                'end': self.roll_dates.index.max()
            },
            'avg_days_to_expiry': self.roll_dates['days_to_expiry'].mean(),
            'avg_interval_days': self.roll_dates['interval_days'].mean(),
            'avg_confidence': self.roll_dates['confidence'].mean(),
            'low_confidence_rolls': len(self.roll_dates[self.roll_dates['confidence'] < 0.7]),
            'roll_patterns': self.roll_dates.groupby(['from_contract_expiration_code', 'to_contract_expiration_code']).size().to_dict(),
            'reasons': self.roll_dates['reason'].value_counts().to_dict()
        }

    @classmethod
    def example(cls) -> 'SpreadData':
        """
        Create a simple example SpreadData instance with preloaded test data.

        Generates 100 days of synthetic futures curve data with 6 contracts.
        Perfect for testing operations without requiring actual market data.

        Returns:
        --------
        SpreadData
            Instance with synthetic test data ready for analysis
        """
        # Generate 100 business days starting from Jan 1, 2023
        dates = pd.bdate_range(start='2023-01-01', periods=100, freq='B')
        n_contracts = 6

        # Generate realistic contango curve
        base_price = 50.0
        prices = np.zeros((100, n_contracts))
        volumes = np.zeros((100, n_contracts))
        open_interest = np.zeros((100, n_contracts))
        days_to_expiry = np.zeros((100, n_contracts))

        for i in range(100):
            # Simple random walk for front month
            if i == 0:
                front_price = base_price
            else:
                front_price = prices[i-1, 0] * (1 + np.random.normal(0, 0.01))

            # Generate contango curve (higher prices for longer contracts)
            for j in range(n_contracts):
                contango_premium = j * 0.5  # 50 cents per month
                prices[i, j] = front_price + contango_premium + np.random.normal(0, 0.05)

                # Volume decreases with contract distance
                volumes[i, j] = max(10, 1000 * np.exp(-j * 0.3) * (1 + np.random.normal(0, 0.2)))

                # Open interest pattern
                open_interest[i, j] = max(100, 5000 * np.exp(-j * 0.2) * (1 + np.random.normal(0, 0.15)))

                # Days to expiry (decreasing)
                days_to_expiry[i, j] = 30 + j * 30 - (i % 30)

        # Create contract labels
        contract_labels = [f'M{i}' for i in range(n_contracts)]
        month_labels = ['H', 'J', 'K', 'M', 'N', 'Q']  # Mar, Apr, May, Jun, Jul, Aug

        # Create DataFrames
        seq_prices = pd.DataFrame(prices, index=dates, columns=contract_labels)
        seq_volume = pd.DataFrame(volumes, index=dates, columns=contract_labels)
        seq_oi = pd.DataFrame(open_interest, index=dates, columns=contract_labels)
        seq_labels = pd.DataFrame([month_labels] * 100, index=dates, columns=contract_labels)
        seq_dte = pd.DataFrame(days_to_expiry, index=dates, columns=contract_labels)
        curve_df = pd.DataFrame(prices, index=dates, columns=month_labels)

        # Calculate spreads vs front month
        spreads = prices - prices[:, [0]]
        seq_spreads = pd.DataFrame(spreads, index=dates, columns=contract_labels)

        # Create SpreadData instance with all required data
        instance = cls(
            symbol="EXAMPLE",
            seq_prices=seq_prices,
            seq_volume=seq_volume,
            seq_oi=seq_oi,
            seq_labels=seq_labels,
            seq_dte=seq_dte,
            seq_spreads=seq_spreads,
            index=dates,
            curve=curve_df,
            volume_curve=volumes,
            oi_curve=open_interest
        )

        return instance
    
    def has_roll_data(self) -> bool:
        """
        Check if roll dates data is available for this symbol.
        
        Returns:
        --------
        bool
            True if roll_dates DataFrame is available and not empty
        """
        return self.roll_dates is not None and len(self.roll_dates) > 0
    
    def get_roll_events_in_range(self, 
                                start_date: Union[str, pd.Timestamp],
                                end_date: Union[str, pd.Timestamp]) -> Optional[pd.DataFrame]:
        """
        Get all roll events within a specific date range.
        
        Parameters:
        -----------
        start_date : str or pd.Timestamp
            Start date of range
        end_date : str or pd.Timestamp
            End date of range
            
        Returns:
        --------
        pd.DataFrame or None
            Roll events in the specified range, None if no data available
        """
        if self.roll_dates is None:
            return None
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        mask = (self.roll_dates.index >= start_date) & (self.roll_dates.index <= end_date)
        filtered_df = self.roll_dates[mask]

        return filtered_df if len(filtered_df) > 0 else None


@dataclass
class CrossSpreadLeg:
    """Definition of a single leg within a multi-asset spread."""

    data: 'SpreadData'
    ratio: float = 1.0
    contract_ratios: Optional[Dict[str, float]] = None
    label: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.data, SpreadData):
            raise TypeError("CrossSpreadLeg requires a SpreadData instance")

        self.ratio = float(self.ratio)
        if self.contract_ratios is not None:
            normalized: Dict[str, float] = {}
            for contract, weight in self.contract_ratios.items():
                normalized[str(contract)] = float(weight)
            self.contract_ratios = normalized

    @property
    def display_label(self) -> str:
        return self.label or getattr(self.data, 'symbol', None) or 'LEG'

    def weight_for_contract(self, contract: str) -> float:
        if self.contract_ratios and contract in self.contract_ratios:
            return self.contract_ratios[contract]
        return self.ratio

    def weights_for_contracts(self, contracts: Sequence[str]) -> Dict[str, float]:
        return {contract: self.weight_for_contract(contract) for contract in contracts}




@dataclass
class CrossProductSpreadData:
    """Cross-asset spread container supporting multi-leg weighted combinations."""

    base: SpreadData
    hedge: Optional[SpreadData] = None
    align: str = "intersection"
    base_ratio: float = 1.0
    hedge_ratio: float = -1.0
    base_contract_ratios: Optional[Dict[str, float]] = None
    hedge_contract_ratios: Optional[Dict[str, float]] = None
    additional_legs: Optional[Sequence[Union[CrossSpreadLeg, Tuple['SpreadData', float], Dict[str, Any]]]] = None

    seq_spreads: pd.DataFrame = field(init=False)
    spread_dataframe: pd.DataFrame = field(init=False)
    index: pd.DatetimeIndex = field(init=False)
    columns: List[str] = field(init=False)
    base_symbol: Optional[str] = field(init=False)
    hedge_symbol: Optional[str] = field(init=False)
    pair_label: str = field(init=False)
    legs: List[CrossSpreadLeg] = field(init=False, repr=False)
    leg_source_labels: List[str] = field(init=False, repr=False)
    leg_labels: List[str] = field(init=False)
    leg_label_map: Dict[str, str] = field(init=False, repr=False)
    leg_contract_weights: Dict[str, Dict[str, float]] = field(init=False)
    leg_contributions: Dict[str, pd.DataFrame] = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.base, SpreadData):
            raise TypeError("CrossProductSpreadData requires a SpreadData instance for the base leg")

        legs: List[CrossSpreadLeg] = [
            CrossSpreadLeg(
                data=self.base,
                ratio=self.base_ratio,
                contract_ratios=self.base_contract_ratios,
                label=getattr(self.base, 'symbol', None),
            )
        ]

        if self.hedge is not None:
            if not isinstance(self.hedge, SpreadData):
                raise TypeError("CrossProductSpreadData requires SpreadData inputs for all legs")
            legs.append(
                CrossSpreadLeg(
                    data=self.hedge,
                    ratio=self.hedge_ratio,
                    contract_ratios=self.hedge_contract_ratios,
                    label=getattr(self.hedge, 'symbol', None),
                )
            )

        if self.additional_legs:
            for definition in self.additional_legs:
                legs.append(self._coerce_leg(definition))

        if len(legs) < 2:
            raise ValueError("CrossProductSpreadData requires at least two legs")

        if self.align not in {"intersection", "union"}:
            raise ValueError("align must be either 'intersection' or 'union'")

        price_frames: List[pd.DataFrame] = []
        for leg in legs:
            frame = leg.data.get_sequential_prices(dropna=False)
            if not isinstance(frame, pd.DataFrame):
                raise ValueError("Sequential prices must be a pandas DataFrame")
            price_frames.append(frame)

        if self.align == "intersection":
            common_index = price_frames[0].index
            for frame in price_frames[1:]:
                common_index = common_index.intersection(frame.index)
        else:
            common_index = price_frames[0].index
            for frame in price_frames[1:]:
                common_index = common_index.union(frame.index)

        common_index = common_index.sort_values()
        if len(common_index) == 0:
            raise ValueError("No overlapping dates across the provided SpreadData legs")

        common_columns = price_frames[0].columns
        for frame in price_frames[1:]:
            common_columns = common_columns.intersection(frame.columns)

        if len(common_columns) == 0:
            raise ValueError("No overlapping contract identifiers across the provided SpreadData legs")

        column_set = set(common_columns)
        column_order = [col for col in price_frames[0].columns if col in column_set]
        if not column_order:
            column_order = list(common_columns)

        aligned_frames = [
            frame.reindex(index=common_index, columns=column_order).astype(float)
            for frame in price_frames
        ]

        self.legs = legs
        raw_labels = [leg.display_label for leg in self.legs]
        resolved_labels = self._resolve_leg_labels(raw_labels)
        self.leg_source_labels = raw_labels
        self.leg_labels = resolved_labels
        self.leg_label_map = dict(zip(resolved_labels, raw_labels))

        weight_series = [
            pd.Series(leg.weights_for_contracts(column_order), index=column_order, dtype=float)
            for leg in self.legs
        ]

        weighted_frames = [
            frame.multiply(weights, axis=1)
            for frame, weights in zip(aligned_frames, weight_series)
        ]

        spread_df = weighted_frames[0].copy()
        for weighted in weighted_frames[1:]:
            spread_df = spread_df + weighted

        spread_df = spread_df.sort_index()

        self.spread_dataframe = spread_df
        self.index = spread_df.index
        self.columns = list(spread_df.columns)
        self.base_symbol = getattr(self.base, 'symbol', None)
        self.hedge_symbol = getattr(self.hedge, 'symbol', None) if self.hedge is not None else None

        self.leg_contract_weights = {
            label: weights.to_dict()
            for label, weights in zip(self.leg_labels, weight_series)
        }
        self.leg_contributions = {
            label: frame.copy()
            for label, frame in zip(self.leg_labels, weighted_frames)
        }

        self.pair_label = " + ".join(
            self._describe_leg(leg, label)
            for leg, label in zip(self.legs, self.leg_labels)
        )

        self.seq_spreads = SpreadFeature(
            data=spread_df.values,
            frame=spread_df,
            labels=self.columns,
            index=self.index,
            sequential=True,
            direction="horizontal",
        )

    @staticmethod
    def _coerce_leg(definition: Union[CrossSpreadLeg, Tuple['SpreadData', float], Dict[str, Any]]) -> CrossSpreadLeg:
        if isinstance(definition, CrossSpreadLeg):
            return definition

        if isinstance(definition, tuple):
            if len(definition) < 2:
                raise ValueError("Tuple leg definitions must provide at least (data, ratio)")

            data = definition[0]
            ratio = definition[1]
            contract_ratios: Optional[Dict[str, float]] = None
            label: Optional[str] = None

            if len(definition) >= 3:
                third = definition[2]
                if isinstance(third, dict):
                    contract_ratios = third
                    if len(definition) >= 4:
                        label = definition[3]
                else:
                    label = third
                    if len(definition) >= 4:
                        contract_ratios = definition[3]

            if len(definition) > 4:
                raise ValueError("Tuple leg definitions support at most four elements")

            return CrossSpreadLeg(data=data, ratio=ratio, contract_ratios=contract_ratios, label=label)

        if isinstance(definition, dict):
            leg_dict = dict(definition)
            if 'data' not in leg_dict:
                raise ValueError("Dictionary leg definitions require a 'data' key")

            data = leg_dict.pop('data')
            ratio = leg_dict.pop('ratio', leg_dict.pop('weight', 1.0))
            contract_ratios = leg_dict.pop('contract_ratios', leg_dict.pop('contracts', None))
            label = leg_dict.pop('label', None)

            if leg_dict:
                raise ValueError(f"Unsupported keys for leg definition: {sorted(leg_dict.keys())}")

            return CrossSpreadLeg(data=data, ratio=ratio, contract_ratios=contract_ratios, label=label)

        raise TypeError("Leg definitions must be CrossSpreadLeg instances, tuples, or dictionaries")

    @staticmethod
    def _resolve_leg_labels(labels: Sequence[str]) -> List[str]:
        counts: Dict[str, int] = {}
        resolved: List[str] = []
        for label in labels:
            count = counts.get(label, 0)
            counts[label] = count + 1
            if count == 0:
                resolved.append(label)
            else:
                resolved.append(f"{label}_{count + 1}")
        return resolved

    @staticmethod
    def _format_ratio(value: float) -> str:
        if float(value).is_integer():
            return str(int(round(value)))
        return f"{value:.4g}"

    def _describe_leg(self, leg: CrossSpreadLeg, label: str) -> str:
        original_label = self.leg_label_map.get(label, label)
        display_label = label if original_label == label else f"{label}({original_label})"
        descriptor = f"{display_label}*{self._format_ratio(leg.ratio)}"
        if leg.contract_ratios:
            overrides = ", ".join(
                f"{contract}:{self._format_ratio(weight)}"
                for contract, weight in sorted(leg.contract_ratios.items())
            )
            descriptor += f" [{overrides}]"
        return descriptor

    def get_leg_weights(self) -> Dict[str, Dict[str, float]]:
        """Return the weight applied to each contract for every leg."""
        return {label: weights.copy() for label, weights in self.leg_contract_weights.items()}

    def get_leg_contribution(self, label: str) -> pd.DataFrame:
        """Return the weighted price contribution for a specific leg."""
        if label in self.leg_contributions:
            return self.leg_contributions[label].copy()

        for resolved, original in self.leg_label_map.items():
            if original == label and resolved in self.leg_contributions:
                return self.leg_contributions[resolved].copy()

        raise KeyError(f"Leg {label} not available in cross-product spreads")

    def to_dataframe(self) -> pd.DataFrame:
        """Return the cross-asset spread matrix."""
        return self.spread_dataframe.copy()

    def get_contract_spread(self, contract: str) -> pd.Series:
        """Return the spread series for a given sequential contract code."""
        if contract not in self.spread_dataframe.columns:
            raise KeyError(f"Contract {contract} not available in cross-product spreads")
        return self.spread_dataframe[contract].copy()

    def get_front_spread(self) -> Optional[pd.Series]:
        """Convenience accessor for the front-month spread if available."""
        for key in ("M0", "front", "F"):
            if key in self.spread_dataframe.columns:
                return self.spread_dataframe[key].copy()
        if len(self.spread_dataframe.columns) > 0:
            return self.spread_dataframe.iloc[:, 0].copy()
        return None

    def describe(self) -> Dict[str, Any]:
        """Basic statistics describing the cross-product spreads and leg structure."""
        stats = {
            'pair': self.pair_label,
            'base_symbol': self.base_symbol,
            'hedge_symbol': self.hedge_symbol,
            'contracts': self.columns,
            'date_range': {
                'start': self.index.min() if len(self.index) else None,
                'end': self.index.max() if len(self.index) else None,
            },
            'legs': [
                {
                    'label': label,
                    'source_label': self.leg_label_map.get(label, label),
                    'ratio': leg.ratio,
                    'contract_weights': self.leg_contract_weights[label].copy(),
                }
                for leg, label in zip(self.legs, self.leg_labels)
            ],
        }

        if not self.spread_dataframe.empty:
            stats['mean_spread'] = self.spread_dataframe.mean().to_dict()
            stats['std_spread'] = self.spread_dataframe.std().to_dict()

        return stats






