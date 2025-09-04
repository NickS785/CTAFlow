
import os
import re
import calendar
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple, List
from config import *

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
        Fixed sequencing that maintains proper calendar order.
        Prevents rolled contracts from appearing in wrong positions.
        """
        curve = self.curve
        if curve is None:
            raise RuntimeError("Curve not built. Call build_curve() first.")
        if self.dte is None:
            self.compute_dte_matrix(curve)

        max_cols = curve.shape[1]
        mcols = [f"M{i}" for i in range(max_cols)]
        seq_prices = pd.DataFrame(index=curve.index, columns=mcols, dtype="float")
        seq_labels = pd.DataFrame(index=curve.index, columns=mcols, dtype="object")
        seq_dte = pd.DataFrame(index=curve.index, columns=mcols, dtype="float")

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
            from data.data_client import DataClient
            client = DataClient(market_path=self.hdf_path)
            
            # Ensure symbol has _F suffix for consistency
            symbol_key = f"{self.symbol}_F" if not self.symbol.endswith('_F') else self.symbol
            
            # Write curve data using DataClient
            client.write_market(self.curve, f"market/{symbol_key}/curve")
            
            if self.front is not None:
                client.write_market(self.front.to_frame(), f"market/{symbol_key}/front_month")
            
            if self.dte is not None:
                client.write_market(self.dte, f"market/{symbol_key}/days_to_expiry")
                
            if self.seq_prices is not None:
                client.write_market(self.seq_prices, f"market/{symbol_key}/curve_seq")
                
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
                if self.front is not None:
                    store.put(f"market/{symbol_key}/front_month", self.front.to_frame(), format="table", data_columns=True)
                if self.dte is not None:
                    store.put(f"market/{symbol_key}/days_to_expiry", self.dte, format="table", data_columns=True)
                if self.seq_prices is not None:
                    store.put(f"market/{symbol_key}/curve_seq", self.seq_prices, format="table", data_columns=True)
                if self.seq_labels is not None:
                    store.put(f"market/{symbol_key}/curve_seq_labels", self.seq_labels, format="table", data_columns=True)
                if self.seq_dte is not None:
                    store.put(f"market/{symbol_key}/days_to_expiry_seq", self.seq_dte, format="table", data_columns=True)
                if self.seq_prices is not None:
                    store.put(f"market/{symbol_key}/spreads_seq", self.spreads_vs_front(self.seq_prices), format="table", data_columns=True)

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
        roll_issues = self.diagnose_rolls( fm)
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

        return {
            "curve": f"market/{symbol_key}/curve",
            "front": f"market/{symbol_key}/front_month",
            "dte": f"market/{symbol_key}/days_to_expiry",
            "seq_curve": f"market/{symbol_key}/curve_seq",
            "seq_labels": f"market/{symbol_key}/curve_seq_labels",
            "seq_dte": f"market/{symbol_key}/days_to_expiry_seq",
            "seq_spreads": f"market/{symbol_key}/spreads_seq",
        }
