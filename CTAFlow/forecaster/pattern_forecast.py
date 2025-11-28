"""Helpers for building ML-ready datasets (X, y) for pattern prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["PatternDataset", "PatternMLBuilder"]

from pandas import Series


@dataclass
class PatternDataset:
    """Container for ML-ready data derived from a pattern specification.

    Attributes
    ----------
    X      : pd.DataFrame
        Feature matrix with one row per training example.
    y      : pd.Series
        Target series aligned to X.index.
    meta   : Dict[str, Any]
        Pattern metadata or any additional info needed downstream.
    info   : Dict[str, Any]
        Diagnostics such as number of samples, filter masks, etc.
    """

    X: pd.DataFrame
    y: pd.Series
    meta: Dict[str, Any]
    info: Dict[str, Any]


class PatternMLBuilder:
    """Factory for building (X, y) datasets for different pattern types.

    This module is intentionally model-agnostic: it only defines how to
    compute the dependent variable (y) and feature matrix (X) based on
    the pattern type and price/feature history.

    Supported pattern families
    --------------------------
    - weekday_mean:
        y = daily return for the target weekday
        X = any features aligned to those dates.

    - momentum_oc (open-close momentum):
        - X: opening-period log return over an early window after session open,
             e.g. r_open_1p5h = log(price(t_open + 1.5h) / price_open).
        - y: closing-period log return from the early window to session close,
             e.g. log(price_close / price(t_open + 1.5h)).

    - momentum_sc (session-close momentum):
        - X: log move from session open to some point before close,
             e.g. log(price(t_close - 1.5h) / price_open).
        - y: log move from that point to session close,
             e.g. log(price_close / price(t_close - 1.5h)).

    - momentum_cc (close-close momentum):
        - X: short-term momentum (returns over the last st_momentum_days).
        - y: session return (open to close).

    For momentum patterns, additional volatility-significance predictors
    can be appended to X (vol_features_df).
    """

    def __init__(
        self,
        *,
        session_col: str = "session_id",
        price_col: str = "Close",
        open_col: str = "Open",
    ) -> None:
        self.session_col = session_col
        self.price_col = price_col
        self.open_col = open_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_dataset(
            self,
            price_df: pd.DataFrame,
            pattern_meta: Mapping[str, Any],
            *,
            features_df: Optional[pd.DataFrame] = None,
            vol_features_df: Optional[pd.DataFrame] = None,
            vol_persistence: bool = False,
    ) -> PatternDataset:
        """Dispatch to the appropriate builder based on pattern type.

        Parameters
        ----------
        price_df : pd.DataFrame
            OHLCV price data
        pattern_meta : Mapping[str, Any]
            Pattern metadata
        features_df : Optional[pd.DataFrame]
            Additional features to include in X
        vol_features_df : Optional[pd.DataFrame]
            Volatility-specific features
        vol_persistence : bool
            If True, add opening period realized volatility as a feature to X
        """
        ptype = (
                pattern_meta.get("pattern_type")
                or pattern_meta.get("type")
                or pattern_meta.get("name")
        )
        if ptype is None:
            raise ValueError("pattern_meta must contain a 'pattern_type' or 'type' key.")

        ptype = str(ptype).lower()

        if "weekday" in ptype:
            X, y, info = self._build_weekday_mean_dataset(price_df, pattern_meta, features_df)
        elif "weekend_hedging" in ptype:
            X, y, info = self._build_weekend_hedging_dataset(
                price_df, pattern_meta, features_df, vol_features_df
            )
        elif "time_predictive_nextday" in ptype:
            X, y, info = self._build_time_predictive_dataset(
                price_df,
                pattern_meta,
                features_df=features_df,
                horizon_days=1,
            )
        elif "time_predictive_nextweek" in ptype:
            X, y, info = self._build_time_predictive_dataset(
                price_df,
                pattern_meta,
                features_df=features_df,
                horizon_days=5,
            )
        elif "momentum_oc" in ptype or ptype == "momentum_oc":
            X, y, info = self._build_momentum_oc_dataset(
                price_df, pattern_meta, features_df, vol_features_df, vol_persistence
            )
        elif "momentum_sc" in ptype or ptype == "momentum_sc":
            X, y, info = self._build_momentum_sc_dataset(
                price_df, pattern_meta, features_df, vol_features_df, vol_persistence
            )
        elif "momentum_cc" in ptype or ptype == "momentum_cc":
            X, y, info = self._build_momentum_cc_dataset(
                price_df, pattern_meta, features_df, vol_features_df, vol_persistence
            )
        else:
            raise NotImplementedError(f"Pattern type '{ptype}' is not supported yet.")

        return PatternDataset(X=X, y=y, meta=dict(pattern_meta), info=info)


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_dt_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index, errors="raise")
            except Exception as exc:
                raise TypeError("Expected a DatetimeIndex or coercible index.") from exc
        return df.sort_index()

    def _session_key(self, df: pd.DataFrame) -> pd.Series:
        """Infer a per-session key: session_id if available else calendar date."""
        if self.session_col in df.columns:
            return df[self.session_col]
        return df.index.normalize()

    def _gate_column(self, pattern_meta: Mapping[str, Any], features_df: Optional[pd.DataFrame]) -> str:
        if features_df is None:
            raise ValueError("features_df is required for time-based predictive patterns")

        explicit = (
            pattern_meta.get("pattern_gate_col")
            or pattern_meta.get("gate_col")
            or (pattern_meta.get("pattern_payload") or {}).get("pattern_gate_col")
        )
        if explicit and explicit in features_df.columns:
            return str(explicit)

        gate_candidates = [col for col in features_df.columns if str(col).endswith("_gate")]
        if not gate_candidates:
            raise KeyError("Unable to locate a gate column for the provided pattern")
        if len(gate_candidates) > 1:
            # Prefer the first deterministic ordering while still being explicit
            gate_candidates = sorted(gate_candidates)
        return gate_candidates[0]

    def _month_filter(
        self, df: pd.DataFrame, months_active: Optional[Sequence[int]]
    ) -> pd.Series:
        if not months_active:
            return pd.Series(True, index=df.index)
        months_active = set(int(m) for m in months_active)
        return df.index.month.isin(months_active)

    def _calculate_realized_volatility(
        self,
        df: pd.DataFrame,
        window_minutes: int,
    ) -> pd.Series:
        """Calculate realized volatility over a rolling window.

        Parameters
        ----------
        df : pd.DataFrame
            Price data with DatetimeIndex
        window_minutes : int
            Window size in minutes for RV calculation

        Returns
        -------
        pd.Series
            Realized volatility (std of log returns over window)
        """
        df = self._ensure_dt_index(df)
        price = pd.to_numeric(df[self.price_col], errors="coerce")

        # Calculate log returns
        log_returns = np.log(price / price.shift(1))

        # Calculate rolling standard deviation
        window_td = pd.Timedelta(minutes=window_minutes)
        rv = log_returns.rolling(window=window_td).std()

        return rv

    def _weekday_filter(
        self, df: pd.DataFrame, weekday_spec: Optional[str]
    ) -> Series | bool:
        """Filter by weekday string, e.g. 'Monday' or 'Friday->Monday'.

        For weekday_mean patterns we only use the first token before '->'.
        """
        if not weekday_spec:
            return pd.Series(True, index=df.index)
        # 'Friday->Monday' -> 'Friday'
        day = weekday_spec.split("->")[0].strip()
        target = day.lower()
        names = df.index.day_name().str.lower()
        return names == target

    def _close_to_close_returns(self, df: pd.DataFrame, horizon_days: int) -> pd.Series:
        df = self._ensure_dt_index(df)
        session_key = self._session_key(df)
        closes = pd.to_numeric(df[self.price_col], errors="coerce")
        session_closes = closes.groupby(session_key).last()
        future = session_closes.shift(-horizon_days)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.log(future / session_closes)
        cleaned = pd.Series(raw).replace([np.inf, -np.inf], np.nan)
        return session_key.map(cleaned)
    # ------------------------------------------------------------------
    # 4) Weekend hedging: Friday → Monday
    #     X: Friday session features (at least Fri session return, weekend gap)
    #     y: Monday session log return (open → close)
    # ------------------------------------------------------------------
    def _build_weekend_hedging_dataset(
        self,
        price_df: pd.DataFrame,
        pattern_meta: Mapping[str, Any],
        features_df: Optional[pd.DataFrame],
        vol_features_df: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        df = self._ensure_dt_index(price_df)
        if self.price_col not in df.columns:
            raise KeyError(self.price_col)
        if self.open_col not in df.columns:
            raise KeyError(self.open_col)

        sess_key = self._session_key(df)
        price = pd.to_numeric(df[self.price_col], errors="coerce")
        open_px = pd.to_numeric(df[self.open_col], errors="coerce")

        # Build per-session summary: one row per trading session
        sess_rows: list[Dict[str, Any]] = []
        for sid, g in df.groupby(sess_key):
            g = g.sort_index()
            if g.empty:
                continue
            day = g.index[0].normalize()
            weekday_name = g.index[0].day_name()
            o = float(open_px.loc[g.index[0]])
            c = float(price.loc[g.index[-1]])
            if not np.isfinite(o) or not np.isfinite(c):
                continue
            sess_rows.append(
                {
                    "session_id": sid,
                    "session_day": day,
                    "weekday": weekday_name,
                    "open": o,
                    "close": c,
                    "first_ts": g.index[0],
                    "last_ts": g.index[-1],
                }
            )

        if not sess_rows:
            return pd.DataFrame(), pd.Series(dtype=float), {
                "pattern_type": "weekend_hedging",
                "n_samples": 0,
            }

        sess_df = pd.DataFrame(sess_rows).set_index("session_day").sort_index()

        # Filter by months_active (applied to the target Monday session)
        months_active = pattern_meta.get("months_active") or pattern_meta.get("months")
        if months_active:
            months_set = set(int(m) for m in months_active)
        else:
            months_set = None

        # We are interested in Friday -> next Monday pairs
        is_fri = sess_df["weekday"].str.lower() == "friday"
        is_mon = sess_df["weekday"].str.lower() == "monday"

        fri_days = sess_df.index[is_fri]
        mon_days = sess_df.index[is_mon]

        samples: Dict[pd.Timestamp, Dict[str, Any]] = {}

        for fri_day in fri_days:
            # Find the next Monday session after this Friday
            future_mons = mon_days[mon_days > fri_day]
            if len(future_mons) == 0:
                continue
            mon_day = future_mons[0]

            if months_set is not None and mon_day.month not in months_set:
                continue

            row_f = sess_df.loc[fri_day]
            row_m = sess_df.loc[mon_day]

            px_f_open = float(row_f["open"])
            px_f_close = float(row_f["close"])
            px_m_open = float(row_m["open"])
            px_m_close = float(row_m["close"])

            if not all(
                np.isfinite(x)
                for x in (px_f_open, px_f_close, px_m_open, px_m_close)
            ):
                continue

            # Friday session return
            r_fri = np.log(px_f_close / px_f_open)
            # Monday session return (target y)
            r_mon = np.log(px_m_close / px_m_open)
            # Weekend gap (optional predictor): close_F → open_M
            r_gap = np.log(px_m_open / px_f_close)

            samples[mon_day] = {
                "fri_day": fri_day,
                "mon_day": mon_day,
                "x_fri_ret": r_fri,
                "x_weekend_gap": r_gap,
                "y_mon_ret": r_mon,
                "t_fri_close": row_f["last_ts"],
                "t_mon_open": row_m["first_ts"],
                "t_mon_close": row_m["last_ts"],
            }

        if not samples:
            return pd.DataFrame(), pd.Series(dtype=float), {
                "pattern_type": "weekend_hedging",
                "n_samples": 0,
            }

        # Index samples by target day (Monday)
        index = pd.Index(sorted(samples.keys()), name="session_day")
        list_idx = index.tolist()
        base_X = pd.DataFrame(
            {
                "x_fri_ret": [samples[d]["x_fri_ret"] for d in list_idx],
                "x_weekend_gap": [samples[d]["x_weekend_gap"] for d in list_idx],
            },
            index=index,
        )
        y = pd.Series(
            [samples[d]["y_mon_ret"] for d in list_idx],
            index=index,
            name="y_mon_ret",
        )

        # Attach intraday features: default = snapshot at Friday close
        if features_df is not None and not features_df.empty:
            fdf = self._ensure_dt_index(features_df)
            feat_rows = []
            for mon_day in list_idx:
                t_fri_close = samples[mon_day]["t_fri_close"]
                if t_fri_close in fdf.index:
                    feat_rows.append(fdf.loc[t_fri_close])
                else:
                    subset = fdf.loc[:t_fri_close]
                    feat_rows.append(
                        subset.iloc[-1] if not subset.empty else pd.Series(dtype=float)
                    )
            feat_X = pd.DataFrame(feat_rows, index=index)
            base_X = pd.concat([base_X, feat_X], axis=1)

        # Attach volatility-significance features (e.g. daily vol regime)
        if vol_features_df is not None and not vol_features_df.empty:
            vdf = self._ensure_dt_index(vol_features_df)
            # Assume vol features are daily or intraday; take last per day
            v_daily = vdf.groupby(vdf.index.normalize()).last()
            # We use Friday's date as the vol snapshot (predictor)
            fri_dates = pd.Index([samples[d]["fri_day"] for d in index])
            vX = v_daily.reindex(fri_dates)
            vX.index = index  # align to Monday index
            base_X = pd.concat([base_X, vX], axis=1)

        info = {
            "pattern_type": "weekend_hedging",
            "n_samples": int(len(y)),
            "months_active": months_active,
            "weekday": pattern_meta.get("weekday", "Friday->Monday"),
        }
        return base_X, y, info

    def _build_time_predictive_dataset(
        self,
        price_df: pd.DataFrame,
        pattern_meta: Mapping[str, Any],
        *,
        features_df: Optional[pd.DataFrame],
        horizon_days: int,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        df = self._ensure_dt_index(price_df)
        gate_col = self._gate_column(pattern_meta, features_df)

        gates = features_df[gate_col].reindex(df.index).fillna(0).astype(bool)
        targets = self._close_to_close_returns(df, horizon_days)
        valid_mask = gates & targets.notna()

        if not valid_mask.any():
            return pd.DataFrame(), pd.Series(dtype=float), {
                "pattern_type": pattern_meta.get("pattern_type", "time_predictive"),
                "n_samples": 0,
            }

        X = features_df.reindex(df.index)
        X = X.loc[valid_mask]
        y = targets.loc[valid_mask]

        info = {
            "pattern_type": pattern_meta.get("pattern_type", "time_predictive"),
            "gate_col": gate_col,
            "n_samples": int(valid_mask.sum()),
        }
        return X, y, info


    # ------------------------------------------------------------------
    # 1) Weekday mean patterns
    # ------------------------------------------------------------------
    def _build_weekday_mean_dataset(
        self,
        price_df: pd.DataFrame,
        pattern_meta: Mapping[str, Any],
        features_df: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """y = daily returns on target weekday; X = features aligned to those days."""
        df = self._ensure_dt_index(price_df)
        if self.price_col not in df.columns:
            raise KeyError(self.price_col)

        price = pd.to_numeric(df[self.price_col], errors="coerce")

        # Daily close and daily log returns
        daily_close = price.groupby(df.index.normalize()).last().dropna()
        daily_ret = np.log(daily_close).diff()

        # Filters from metadata
        months_active = pattern_meta.get("months_active") or pattern_meta.get("months")
        daily_index = daily_close.index
        daily_df = pd.DataFrame(index=daily_index)
        daily_df["close"] = daily_close
        daily_df["ret"] = daily_ret

        # month filter
        m_mask = self._month_filter(daily_df, months_active)
        # weekday filter
        weekday_spec = pattern_meta.get("weekday")
        w_mask = self._weekday_filter(daily_df, weekday_spec)

        mask = m_mask & w_mask & daily_df["ret"].notna()
        y = daily_df.loc[mask, "ret"].copy()

        # Align features: we expect features_df indexed intraday or daily
        if features_df is not None and not features_df.empty:
            fdf = self._ensure_dt_index(features_df)
            # take end-of-day snapshot of features (last row per day)
            f_daily = fdf.groupby(fdf.index.normalize()).last()
            X = f_daily.reindex(y.index).copy()
        else:
            X = pd.DataFrame(index=y.index)

        info = {
            "pattern_type": "weekday_mean",
            "weekday": weekday_spec,
            "months_active": months_active,
            "n_samples": int(len(y)),
        }
        return X, y, info

    # ------------------------------------------------------------------
    # 2) Momentum OC: X = early window from open, y = rest of session to close
    # ------------------------------------------------------------------
    def _build_momentum_oc_dataset(
        self,
        price_df: pd.DataFrame,
        pattern_meta: Mapping[str, Any],
        features_df: Optional[pd.DataFrame],
        vol_features_df: Optional[pd.DataFrame],
        vol_persistence: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        df = self._ensure_dt_index(price_df)
        if self.price_col not in df.columns:
            raise KeyError(self.price_col)
        if self.open_col not in df.columns:
            raise KeyError(self.open_col)

        sess_key = self._session_key(df)
        price = pd.to_numeric(df[self.price_col], errors="coerce")
        open_px = pd.to_numeric(df[self.open_col], errors="coerce")

        # params: early window length (minutes) derived from period_length_min
        early_min = int(pattern_meta.get("period_length_min", 90))

        months_active = pattern_meta.get("months_active") or pattern_meta.get("months")
        weekday_spec = pattern_meta.get("weekday")

        samples: Dict[pd.Timestamp, Dict[str, Any]] = {}

        for sid, g in df.groupby(sess_key):
            g = g.sort_index()
            if g.empty:
                continue

            day = g.index[0].normalize()
            # filter by months & weekday at session level
            if months_active and g.index[0].month not in set(int(m) for m in months_active):
                continue
            if weekday_spec:
                if g.index[0].day_name().lower() != weekday_spec.split("->")[0].strip().lower():
                    continue

            open_time = g.index[0]
            close_time = g.index[-1]
            # target early decision time ~ open + early_min
            target_ts = open_time + pd.Timedelta(minutes=early_min)
            # find first bar at or after target_ts
            decision_idx = g.index[g.index >= target_ts]
            if len(decision_idx) == 0:
                continue
            t_decision = decision_idx[0]

            px_open = float(open_px.loc[g.index[0]])
            px_dec = float(price.loc[t_decision])
            px_close = float(price.loc[g.index[-1]])

            if not np.isfinite(px_open) or not np.isfinite(px_dec) or not np.isfinite(px_close):
                continue

            # X: opening window log return
            x_early = np.log(px_dec / px_open)
            # y: rest-of-session log return
            y_close = np.log(px_close / px_dec)

            samples[day] = {
                "x_early_ret": x_early,
                "y_close_ret": y_close,
                "t_decision": t_decision,
            }

        if not samples:
            return pd.DataFrame(), pd.Series(dtype=float), {
                "pattern_type": "momentum_oc",
                "n_samples": 0,
            }

        index = pd.Index(sorted(samples.keys()), name="session_day")
        base_X = pd.DataFrame(
            {"x_early_ret": [samples[d]["x_early_ret"] for d in index]}, index=index
        )
        y = pd.Series(
            [samples[d]["y_close_ret"] for d in index],
            index=index,
            name="y_close_ret",
        )

        # Join features at t_decision from features_df / vol_features_df
        if features_df is not None and not features_df.empty:
            fdf = self._ensure_dt_index(features_df)
            feat_rows = []
            for d in index:
                t_dec = samples[d]["t_decision"]
                # pick the exact index if it exists; fallback to last before t_dec
                if t_dec in fdf.index:
                    feat_rows.append(fdf.loc[t_dec])
                else:
                    subset = fdf.loc[:t_dec]
                    feat_rows.append(subset.iloc[-1] if not subset.empty else pd.Series(dtype=float))
            feat_X = pd.DataFrame(feat_rows, index=index)
            base_X = pd.concat([base_X, feat_X], axis=1)

        # Add opening period realized volatility if vol_persistence=True
        if vol_persistence:
            rv = self._calculate_realized_volatility(df, window_minutes=early_min)
            rv_rows = []
            for d in index:
                t_dec = samples[d]["t_decision"]
                if t_dec in rv.index:
                    rv_rows.append({"opening_rv": rv.loc[t_dec]})
                else:
                    subset = rv.loc[:t_dec]
                    rv_val = subset.iloc[-1] if not subset.empty else np.nan
                    rv_rows.append({"opening_rv": rv_val})
            rv_X = pd.DataFrame(rv_rows, index=index)
            base_X = pd.concat([base_X, rv_X], axis=1)

        if vol_features_df is not None and not vol_features_df.empty:
            vdf = self._ensure_dt_index(vol_features_df)
            # assume vol_features_df already at daily frequency (or take last per day)
            v_daily = vdf.groupby(vdf.index.normalize()).last()
            vX = v_daily.reindex(index)
            base_X = pd.concat([base_X, vX], axis=1)

        info = {
            "pattern_type": "momentum_oc",
            "n_samples": int(len(y)),
            "period_length_min": early_min,
            "months_active": months_active,
            "weekday": weekday_spec,
        }
        return base_X, y, info

    # ------------------------------------------------------------------
    # 3) Momentum SC: X = open → t_before_close, y = t_before_close → close
    # ------------------------------------------------------------------
    def _build_momentum_sc_dataset(
        self,
        price_df: pd.DataFrame,
        pattern_meta: Mapping[str, Any],
        features_df: Optional[pd.DataFrame],
        vol_features_df: Optional[pd.DataFrame],
        vol_persistence: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        df = self._ensure_dt_index(price_df)
        if self.price_col not in df.columns:
            raise KeyError(self.price_col)
        if self.open_col not in df.columns:
            raise KeyError(self.open_col)

        sess_key = self._session_key(df)
        price = pd.to_numeric(df[self.price_col], errors="coerce")
        open_px = pd.to_numeric(df[self.open_col], errors="coerce")

        late_min = int(pattern_meta.get("period_length_min", 90))
        months_active = pattern_meta.get("months_active") or pattern_meta.get("months")
        weekday_spec = pattern_meta.get("weekday")

        samples: Dict[pd.Timestamp, Dict[str, Any]] = {}

        for sid, g in df.groupby(sess_key):
            g = g.sort_index()
            if g.empty:
                continue

            day = g.index[0].normalize()
            if months_active and g.index[0].month not in set(int(m) for m in months_active):
                continue
            if weekday_spec:
                if g.index[0].day_name().lower() != weekday_spec.split("->")[0].strip().lower():
                    continue

            open_time = g.index[0]
            close_time = g.index[-1]
            target_ts = close_time - pd.Timedelta(minutes=late_min)
            # find last bar at or before target_ts
            pre_idx = g.index[g.index <= target_ts]
            if len(pre_idx) == 0:
                continue
            t_preclose = pre_idx[-1]

            px_open = float(open_px.loc[g.index[0]])
            px_pre = float(price.loc[t_preclose])
            px_close = float(price.loc[g.index[-1]])

            if not np.isfinite(px_open) or not np.isfinite(px_pre) or not np.isfinite(px_close):
                continue

            # X: log move from open to t_preclose
            x_sc = np.log(px_pre / px_open)
            # y: log move from t_preclose to close
            y_sc = np.log(px_close / px_pre)

            samples[day] = {
                "x_sc_ret": x_sc,
                "y_sc_ret": y_sc,
                "t_decision": t_preclose,
            }

        if not samples:
            return pd.DataFrame(), pd.Series(dtype=float), {
                "pattern_type": "momentum_sc",
                "n_samples": 0,
            }

        index = pd.Index(sorted(samples.keys()), name="session_day")
        base_X = pd.DataFrame(
            {"x_sc_ret": [samples[d]["x_sc_ret"] for d in index]}, index=index
        )
        y = pd.Series(
            [samples[d]["y_sc_ret"] for d in index],
            index=index,
            name="y_sc_ret",
        )

        # attach intraday features at t_preclose
        if features_df is not None and not features_df.empty:
            fdf = self._ensure_dt_index(features_df)
            feat_rows = []
            for d in index:
                t_dec = samples[d]["t_decision"]
                if t_dec in fdf.index:
                    feat_rows.append(fdf.loc[t_dec])
                else:
                    subset = fdf.loc[:t_dec]
                    feat_rows.append(subset.iloc[-1] if not subset.empty else pd.Series(dtype=float))
            feat_X = pd.DataFrame(feat_rows, index=index)
            base_X = pd.concat([base_X, feat_X], axis=1)

        # Add opening period realized volatility if vol_persistence=True
        if vol_persistence:
            rv = self._calculate_realized_volatility(df, window_minutes=late_min)
            rv_rows = []
            for d in index:
                t_dec = samples[d]["t_decision"]
                if t_dec in rv.index:
                    rv_rows.append({"opening_rv": rv.loc[t_dec]})
                else:
                    subset = rv.loc[:t_dec]
                    rv_val = subset.iloc[-1] if not subset.empty else np.nan
                    rv_rows.append({"opening_rv": rv_val})
            rv_X = pd.DataFrame(rv_rows, index=index)
            base_X = pd.concat([base_X, rv_X], axis=1)

        if vol_features_df is not None and not vol_features_df.empty:
            vdf = self._ensure_dt_index(vol_features_df)
            v_daily = vdf.groupby(vdf.index.normalize()).last()
            vX = v_daily.reindex(index)
            base_X = pd.concat([base_X, vX], axis=1)

        info = {
            "pattern_type": "momentum_sc",
            "n_samples": int(len(y)),
            "period_length_min": late_min,
            "months_active": months_active,
            "weekday": weekday_spec,
        }
        return base_X, y, info

    # ------------------------------------------------------------------
    # 4) Momentum CC: X = short-term trend (st_momentum_days), y = session O->C
    # ------------------------------------------------------------------
    def _build_momentum_cc_dataset(
        self,
        price_df: pd.DataFrame,
        pattern_meta: Mapping[str, Any],
        features_df: Optional[pd.DataFrame],
        vol_features_df: Optional[pd.DataFrame],
        vol_persistence: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        df = self._ensure_dt_index(price_df)
        if self.price_col not in df.columns:
            raise KeyError(self.price_col)
        if self.open_col not in df.columns:
            raise KeyError(self.open_col)

        sess_key = self._session_key(df)
        price = pd.to_numeric(df[self.price_col], errors="coerce")
        open_px = pd.to_numeric(df[self.open_col], errors="coerce")

        # Get st_momentum_days from pattern metadata
        st_momentum_days = int(pattern_meta.get("st_momentum_days", 3))
        months_active = pattern_meta.get("months_active") or pattern_meta.get("months")
        weekday_spec = pattern_meta.get("weekday")

        # Calculate per-session close prices
        sess_close = price.groupby(sess_key).last()
        sess_open = open_px.groupby(sess_key).first()

        # Calculate short-term momentum (returns over last st_momentum_days)
        st_momentum = np.log(sess_close / sess_close.shift(st_momentum_days))

        # Calculate session return (open to close)
        sess_return = np.log(sess_close / sess_open)

        samples: Dict[pd.Timestamp, Dict[str, Any]] = {}

        for sid, g in df.groupby(sess_key):
            g = g.sort_index()
            if g.empty:
                continue

            day = g.index[0].normalize()
            # filter by months & weekday at session level
            if months_active and g.index[0].month not in set(int(m) for m in months_active):
                continue
            if weekday_spec:
                if g.index[0].day_name().lower() != weekday_spec.split("->")[0].strip().lower():
                    continue

            # Get st_momentum and session_return for this session
            if sid not in st_momentum.index or sid not in sess_return.index:
                continue

            x_st_mom = float(st_momentum.loc[sid])
            y_sess_ret = float(sess_return.loc[sid])

            if not np.isfinite(x_st_mom) or not np.isfinite(y_sess_ret):
                continue

            samples[day] = {
                "x_st_momentum": x_st_mom,
                "y_sess_ret": y_sess_ret,
                "t_decision": g.index[0],  # Session open time
            }

        if not samples:
            return pd.DataFrame(), pd.Series(dtype=float), {
                "pattern_type": "momentum_cc",
                "n_samples": 0,
            }

        index = pd.Index(sorted(samples.keys()), name="session_day")
        base_X = pd.DataFrame(
            {"x_st_momentum": [samples[d]["x_st_momentum"] for d in index]}, index=index
        )
        y = pd.Series(
            [samples[d]["y_sess_ret"] for d in index],
            index=index,
            name="y_sess_ret",
        )

        # Attach intraday features at session open
        if features_df is not None and not features_df.empty:
            fdf = self._ensure_dt_index(features_df)
            feat_rows = []
            for d in index:
                t_dec = samples[d]["t_decision"]
                if t_dec in fdf.index:
                    feat_rows.append(fdf.loc[t_dec])
                else:
                    subset = fdf.loc[:t_dec]
                    feat_rows.append(subset.iloc[-1] if not subset.empty else pd.Series(dtype=float))
            feat_X = pd.DataFrame(feat_rows, index=index)
            base_X = pd.concat([base_X, feat_X], axis=1)

        # Add opening period realized volatility if vol_persistence=True
        # For CC patterns, use the st_momentum_days period as the window
        if vol_persistence:
            # Calculate RV over the full session (session open to close)
            sess_key = self._session_key(df)
            rv_rows = []
            for d in index:
                # Get all bars for this session
                sid = [k for k, g in df.groupby(sess_key) if g.index[0].normalize() == d]
                if sid:
                    sess_data = df[df[self.session_col if self.session_col in df.columns else "session_id"] == sid[0]]
                    if len(sess_data) > 1:
                        sess_rv = self._calculate_realized_volatility(sess_data, window_minutes=int(st_momentum_days * 390))
                        rv_val = sess_rv.iloc[-1] if not sess_rv.empty and len(sess_rv) > 0 else np.nan
                    else:
                        rv_val = np.nan
                else:
                    rv_val = np.nan
                rv_rows.append({"opening_rv": rv_val})
            rv_X = pd.DataFrame(rv_rows, index=index)
            base_X = pd.concat([base_X, rv_X], axis=1)

        # Attach volatility features for vol_persistence
        if vol_features_df is not None and not vol_features_df.empty:
            vdf = self._ensure_dt_index(vol_features_df)
            v_daily = vdf.groupby(vdf.index.normalize()).last()
            vX = v_daily.reindex(index)
            base_X = pd.concat([base_X, vX], axis=1)

        info = {
            "pattern_type": "momentum_cc",
            "n_samples": int(len(y)),
            "st_momentum_days": st_momentum_days,
            "months_active": months_active,
            "weekday": weekday_spec,
        }
        return base_X, y, info
