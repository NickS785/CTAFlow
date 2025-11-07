# -*- coding: utf-8 -*-
"""
HARX intraday volatility forecaster & cross-sectional selector (robust)
- Sessionized RV from intraday bars (k-min sum of squared log-returns).
- HARX on log(RV): y_t = c + b1*log(RV_{t-1}) + b5*log(mean RV_{t-5}) + b22*log(mean RV_{t-22}) + X_t*γ + e_t
- Expanding-window OOS forecasts + selection score = zpred * max(R²_recent,0) / (1+QLIKE_recent).
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- Params ----------------
@dataclass
class HARXParams:
    train_min: int = 250
    oos_window: int = 60
    use_log: bool = True
    add_const: bool = True
    ridge_lambda: float = 1e-6   # L2 regularization for ill-conditioned designs
    max_cond: float = 1e8        # use ridge if cond(X) > max_cond

# ---------------- Screener ----------------
class HARXScreener:
    def __init__(
        self,
        session: str = "08:30-15:00",
        cross_midnight: bool = False,
        tz_shift_minutes: int = 0,
        rv_sample_min: int = 5,
        min_returns: int = 30,
        winsor_pct: float = 0.0,
        harx: HARXParams = HARXParams(),
        selection_lookback: int = 60,
    ):
        self.session = session
        self.cross_midnight = cross_midnight
        self.tz_shift_minutes = tz_shift_minutes
        self.rv_sample_min = rv_sample_min
        self.min_returns = min_returns
        self.winsor_pct = winsor_pct
        self.harx = harx
        self.selection_lookback = selection_lookback
        self._sess_start_td, self._sess_end_td = self._parse_session(session)

    # ---------- Public API ----------
    def session_rv(self, intraday_df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_cols(intraday_df.copy())
        if "DateTime" not in df.columns:
            if {"Date", "Time"}.issubset(df.columns):
                df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
            else:
                raise ValueError("Need DateTime or Date+Time")
        else:
            df["DateTime"] = pd.to_datetime(df["DateTime"])
        if self.tz_shift_minutes:
            df["DateTime"] = df["DateTime"] + pd.Timedelta(minutes=self.tz_shift_minutes)

        df["session_date"] = self._assign_session_date_index(df["DateTime"], self._sess_start_td, self.cross_midnight)

        rows = []
        for d, g in df.groupby("session_date", sort=True):
            px = g.set_index("DateTime")["Close"].resample(f"{self.rv_sample_min}T").last().dropna()
            if px.shape[0] < 2:
                continue
            ret = np.log(px).diff().dropna()
            if ret.shape[0] < self.min_returns:
                continue
            rv = float(np.sum(ret.values**2))
            rows.append({"Date": pd.to_datetime(d), "RV": rv, "N": int(ret.shape[0])})

        out = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        if out.empty:
            return out
        if 0.0 < self.winsor_pct < 0.5 and out.shape[0] > 20:
            lo = out["RV"].quantile(self.winsor_pct)
            hi = out["RV"].quantile(1.0 - self.winsor_pct)
            out["RV"] = out["RV"].clip(lower=lo, upper=hi)
        return out

    def forecast_symbol(
        self, symbol: str, intraday_df: pd.DataFrame, exog_daily: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, float, float, float]:
        rv_df = self.session_rv(intraday_df)
        if rv_df.empty or rv_df.shape[0] < self.harx.train_min + 10:
            return pd.DataFrame(), np.nan, np.nan, np.nan
        rv_df["Date"] = pd.to_datetime(rv_df["Date"]).dt.normalize()
        rv = rv_df.set_index("Date")["RV"]
        oos = self._rolling_oos(rv, exog_daily, self.harx)
        if oos.empty:
            return pd.DataFrame(), np.nan, np.nan, np.nan
        score, zpred, r2_recent = self._selection_score(oos, lookback=self.selection_lookback)
        oos["symbol"] = symbol
        return oos, score, zpred, r2_recent

    def rank(
        self,
        bundle: Dict[str, pd.DataFrame],
        exog_map: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
        top_k: int = 10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows, oos_all = [], []
        for sym, idf in bundle.items():
            exog = None if exog_map is None else exog_map.get(sym)
            oos, sc, z, r2 = self.forecast_symbol(sym, idf, exog)
            if oos.empty:
                continue
            oos_all.append(oos)
            rows.append(
                {
                    "symbol": sym,
                    "latest_date": oos["Date"].iloc[-1],
                    "predicted_RV": oos["yhat"].iloc[-1],
                    "actual_RV": oos["y"].iloc[-1],
                    "score": sc,
                    "zpred": z,
                    "r2_recent": r2,
                }
            )
        oos_all_df = (
            pd.concat(oos_all, ignore_index=True)
            if len(oos_all)
            else pd.DataFrame(columns=["Date", "y", "yhat", "r2_is", "rmse_is", "qlike", "symbol"])
        )
        top = (
            pd.DataFrame(rows)
            .sort_values(["score", "zpred", "r2_recent"], ascending=False)
            .head(top_k)
            .reset_index(drop=True)
            if rows
            else pd.DataFrame(columns=["symbol", "latest_date", "predicted_RV", "actual_RV", "score", "zpred", "r2_recent"])
        )
        return oos_all_df, top

    # ---------- Internals ----------
    @staticmethod
    def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("datetime", "timestamp", "date_time"):
                rename[c] = "DateTime"
            elif cl == "date":
                rename[c] = "Date"
            elif cl == "time":
                rename[c] = "Time"
            elif cl == "open":
                rename[c] = "Open"
            elif cl == "high":
                rename[c] = "High"
            elif cl == "low":
                rename[c] = "Low"
            elif cl in ("close", "last", "price"):
                rename[c] = "Close"
            elif cl in ("volume", "vol"):
                rename[c] = "Volume"
            elif cl in ("symbol", "ticker", "root"):
                rename[c] = "symbol"
        return df.rename(columns=rename)

    @staticmethod
    def _parse_session(session: str) -> Tuple[pd.Timedelta, pd.Timedelta]:
        s, e = session.split("-")
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        return pd.Timedelta(hours=sh, minutes=sm), pd.Timedelta(hours=eh, minutes=em)

    @staticmethod
    def _assign_session_date_index(ts: pd.Series, start_td: pd.Timedelta, cross_midnight: bool) -> pd.Series:
        minutes = ts.dt.hour * 60 + ts.dt.minute
        start_min = int(start_td / pd.Timedelta(minutes=1))
        if cross_midnight:
            session_date = np.where(minutes >= start_min, ts.dt.date, (ts - pd.Timedelta(days=1)).dt.date)
        else:
            session_date = np.where(minutes < start_min, (ts - pd.Timedelta(days=1)).dt.date, ts.dt.date)
        return pd.Series(session_date, index=ts.index)

    @staticmethod
    def _make_har_design(rv: pd.Series, use_log: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        eps = 1e-12
        rv_clip = rv.clip(lower=eps)
        y = np.log(rv_clip).rename("y") if use_log else rv_clip.rename("y")
        rv_ma5 = rv_clip.rolling(5).mean().shift(1)
        rv_ma22 = rv_clip.rolling(22).mean().shift(1)

        def _logsafe(s: pd.Series) -> pd.Series:
            return np.log(s.clip(lower=eps))

        X = pd.DataFrame(
            {
                "x1": (_logsafe(rv_clip).shift(1) if use_log else rv_clip.shift(1)),
                "x5": (_logsafe(rv_ma5) if use_log else rv_ma5),
                "x22": (_logsafe(rv_ma22) if use_log else rv_ma22),
            }
        )
        Z = pd.concat([y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        return Z[["x1", "x5", "x22"]], Z["y"]

    @staticmethod
    def _add_exog(X: pd.DataFrame, exog: Optional[pd.DataFrame]) -> pd.DataFrame:
        if exog is None or exog.empty:
            return X
        e = exog.copy()
        if "Date" not in e.columns:
            raise ValueError("Exogenous data must contain 'Date'")
        e["Date"] = pd.to_datetime(e["Date"]).dt.normalize()
        X_ = X.copy()
        X_.index = pd.to_datetime(X_.index).normalize()
        E = e.set_index(pd.to_datetime(e["Date"]).dt.normalize()).drop(columns=["Date"], errors="ignore")
        X2 = X_.join(E, how="left").fillna(method="ffill")
        X2 = X2.dropna(axis=1, how="all")  # drop all-NaN exog cols
        return X2

    @staticmethod
    def _sanitize_design(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        Z = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if Z.empty:
            return X.iloc[0:0], y.iloc[0:0]
        Xc = Z.iloc[:, :-1]
        yc = Z.iloc[:, -1]
        var = Xc.var(numeric_only=True)
        keep = var.index[var > 0.0].tolist()
        if not keep:
            return X.iloc[0:0], y.iloc[0:0]
        return Xc[keep], yc

    @staticmethod
    def _fit_ols(
        X: pd.DataFrame, y: pd.Series, add_const: bool = True, ridge_lambda: float = 1e-6, max_cond: float = 1e8
    ) -> Tuple[np.ndarray, float, float]:
        Xs, ys = HARXScreener._sanitize_design(X, y)
        if Xs.empty or ys.empty or len(Xs) != len(ys):
            return np.array([]), np.nan, np.nan

        Xn = Xs.copy()
        if add_const:
            Xn = pd.concat([pd.Series(1.0, index=Xs.index, name="const"), Xs], axis=1)
        Xv = Xn.values
        yv = ys.values

        try:
            cond = np.linalg.cond(Xv)
        except Exception:
            cond = np.inf

        def _ols():
            beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
            return beta

        def _ridge(lmbd: float):
            XtX = Xv.T @ Xv
            beta = np.linalg.solve(XtX + lmbd * np.eye(XtX.shape[0]), Xv.T @ yv)
            return beta

        try:
            beta = _ridge(ridge_lambda) if (not np.isfinite(cond) or cond > max_cond) else _ols()
        except Exception:
            beta = _ridge(ridge_lambda)

        yhat = Xv @ beta
        resid = yv - yhat
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((yv - yv.mean())**2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.mean(resid**2)))
        return beta, r2, rmse

    @staticmethod
    def _qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-12
        y_true = np.clip(y_true, eps, None)
        y_pred = np.clip(y_pred, eps, None)
        return float(np.mean(np.log(y_pred) + y_true / y_pred))

    def _rolling_oos(self, rv: pd.Series, exog: Optional[pd.DataFrame], hp: HARXParams) -> pd.DataFrame:
        rv = rv.dropna()
        if rv.shape[0] < hp.train_min + 5:
            return pd.DataFrame(columns=["Date", "y", "yhat", "r2_is", "rmse_is", "qlike"])
        rows = []
        for t in range(hp.train_min, rv.shape[0]):
            rv_train = rv.iloc[:t]
            X, y = self._make_har_design(rv_train, use_log=hp.use_log)
            X.index = rv_train.index[-len(X):]
            if exog is not None:
                X = self._add_exog(X, exog)

            beta, r2, rmse = self._fit_ols(
                X, y, add_const=hp.add_const, ridge_lambda=hp.ridge_lambda, max_cond=hp.max_cond
            )

            rv_sub = rv.iloc[:t+1]
            # if fit failed: naive fallback forecast = last observed y (level)
            if beta.size == 0 or not np.isfinite(r2):
                y_level = float(rv.iloc[t])
                rows.append({"Date": rv.index[t], "y": y_level, "yhat": y_level, "r2_is": np.nan, "rmse_is": np.nan})
                continue

            Xfull, _ = self._make_har_design(rv_sub, use_log=hp.use_log)
            Xfull.index = rv_sub.index[-len(Xfull):]
            xt = Xfull.iloc[-1:]
            if exog is not None:
                xt = self._add_exog(xt, exog)
            Xt = np.concatenate(([1.0], xt.values.flatten())) if hp.add_const else xt.values.flatten()

            yhat = float(np.dot(Xt, beta))
            yhat_level = np.exp(yhat) if hp.use_log else yhat
            y_level = float(rv.iloc[t])

            rows.append({"Date": rv.index[t], "y": y_level, "yhat": yhat_level, "r2_is": r2, "rmse_is": rmse})

        oos = pd.DataFrame(rows)
        if oos.empty:
            return oos
        oos["qlike"] = [np.nan] * len(oos)
        if len(oos) > 1:
            oos.loc[1:, "qlike"] = [
                self._qlike(oos["y"].values[:i], oos["yhat"].values[:i]) for i in range(1, len(oos))
            ]
        return oos.tail(hp.oos_window).reset_index(drop=True)

    def _selection_score(self, oos: pd.DataFrame, lookback: int = 60) -> Tuple[float, float, float]:
        if oos.empty:
            return np.nan, np.nan, np.nan
        lb = min(lookback, len(oos))
        r2_recent = float(oos["r2_is"].tail(lb).mean())
        y_hist = oos["y"].tail(lb)
        med = float(np.median(y_hist))
        mad = float(np.median(np.abs(y_hist - med))) or 1e-12
        zpred = float((oos["yhat"].iloc[-1] - med) / (1.4826 * mad))
        ql = float(oos["qlike"].tail(max(5, lb // 3)).mean()) if "qlike" in oos.columns else 0.0
        score = zpred * max(0.0, r2_recent) / (1.0 + max(0.0, ql))
        return score, zpred, r2_recent

# ---------------- CLI (optional) ----------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return HARXScreener._normalize_cols(df)

def _read_intraday(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df = _normalize_cols(df)
    if "DateTime" not in df.columns:
        if {"Date", "Time"}.issubset(df.columns):
            df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
        else:
            raise ValueError(f"{p} needs DateTime or Date+Time")
    else:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df.sort_values("DateTime")

def _load_bundle(intraday_dir: Optional[str], intraday_single: Optional[str], symbols: Optional[List[str]]) -> List[Tuple[str, pd.DataFrame]]:
    out: List[Tuple[str, pd.DataFrame]] = []
    if intraday_single:
        df = pd.read_csv(intraday_single)
        df = _normalize_cols(df)
        if "symbol" not in df.columns:
            raise ValueError("--intraday-single must include `symbol`")
        for sym, g in df.groupby("symbol"):
            g = g.copy()
            if "DateTime" not in g.columns and {"Date","Time"}.issubset(g.columns):
                g["DateTime"] = pd.to_datetime(g["Date"].astype(str) + " " + g["Time"].astype(str))
            elif "DateTime" in g.columns:
                g["DateTime"] = pd.to_datetime(g["DateTime"])
            else:
                raise ValueError("Need DateTime or Date+Time")
            out.append((str(sym), g.sort_values("DateTime")))
        return out
    if intraday_dir is None or symbols is None:
        raise ValueError("When not using --intraday-single, both --intraday-dir and --symbols are required.")
    base = Path(intraday_dir)
    for s in symbols:
        p = base / f"{s}.csv"
        if not p.exists():
            print(f"WARNING: missing {p}; skipping {s}")
            continue
        out.append((s, _read_intraday(p)))
    return out

def _load_exog_single(path: Optional[str], symbols: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
    if not path:
        return {s: None for s in symbols}
    df = pd.read_csv(path)
    df = _normalize_cols(df)
    if "symbol" not in df.columns or "Date" not in df.columns:
        raise ValueError("--exog-single must include columns: symbol, Date, <features...>")
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    out: Dict[str, Optional[pd.DataFrame]] = {}
    for s in symbols:
        e = df[df["symbol"] == s].drop(columns=["symbol"], errors="ignore").copy()
        out[s] = e
    return out

def _cli():
    ap = argparse.ArgumentParser(description="HARX Screener (robust)")
    ap.add_argument("--intraday-dir", type=str, default=None)
    ap.add_argument("--intraday-single", type=str, default=None)
    ap.add_argument("--symbols", type=str, default=None)
    ap.add_argument("--session", type=str, default="08:30-15:00")
    ap.add_argument("--cross-midnight", action="store_true")
    ap.add_argument("--tz-shift-minutes", type=int, default=0)
    ap.add_argument("--rv-sample-min", type=int, default=5)
    ap.add_argument("--min-returns", type=int, default=30)
    ap.add_argument("--winsor-pct", type=float, default=0.0)
    ap.add_argument("--train-min", type=int, default=250)
    ap.add_argument("--oos-window", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--selection-lookback", type=int, default=60)
    ap.add_argument("--exog-single", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ns = ap.parse_args()
    symbols = [s.strip() for s in ns.symbols.split(",")] if ns.symbols else None
    scr = HARXScreener(
        session=ns.session,
        cross_midnight=ns.cross_midnight,
        tz_shift_minutes=ns.tz_shift_minutes,
        rv_sample_min=ns.rv_sample_min,
        min_returns=ns.min_returns,
        winsor_pct=ns.winsor_pct,
        harx=HARXParams(train_min=ns.train_min, oos_window=ns.oos_window, use_log=True, add_const=True),
        selection_lookback=ns.selection_lookback,
    )
    bundle = _load_bundle(ns.intraday_dir, ns.intraday_single, symbols)
    syms = [s for s, _ in bundle]
    exog_map = _load_exog_single(ns.exog_single, syms)
    oos_all, top = scr.rank(bundle, exog_map, top_k=ns.top_k)
    if ns.out:
        Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
        oos_all.to_csv(ns.out, index=False)
        Path(str(Path(ns.out).with_name(Path(ns.out).stem + "_top.csv"))).write_text(top.to_csv(index=False))
    if not top.empty:
        print("Top candidates (HARX robust):")
        print(top.to_string(index=False))
    else:
        print("No candidates.")
if __name__ == "__main__":
    _cli()
