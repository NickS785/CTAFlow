"""Utilities for exporting screen outputs in tidy long-form DataFrames."""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import numpy as np
import pandas as pd

from CTAFlow.stats_utils import fdr_bh
from CTAFlow.stats_utils import regularized_beta

LOGGER = logging.getLogger(__name__)


@dataclass
class _Metadata:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    tz: Optional[str] = None
    return_mode: Optional[str] = None

    @classmethod
    def from_obj(cls, obj: Mapping[str, Any]) -> "_Metadata":
        meta = obj.get("meta") if isinstance(obj, Mapping) else None
        if isinstance(meta, Mapping):
            return cls(
                start_date=meta.get("start_date"),
                end_date=meta.get("end_date"),
                tz=meta.get("tz"),
                return_mode=meta.get("return_mode"),
            )
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "tz": self.tz,
            "return_mode": self.return_mode,
        }


_COMMON_COLUMNS = [
    "screen_type",
    "ticker",
    "region",
    "exchange",
    "slice",
    "metric",
    "value",
    "n_observations",
    "p_value",
    "q_value",
    "sig_5pct",
    "sig_fdr_5pct",
    "start_date",
    "end_date",
    "tz",
    "return_mode",
]


def _student_t_two_sided_pvalue(t_stat: float, df: float) -> float:
    """Return the two sided p-value for a Student's t statistic."""

    if df is None or df <= 0 or not np.isfinite(df):
        return np.nan
    if t_stat is None or not np.isfinite(t_stat):
        return np.nan
    x = df / (df + float(t_stat) ** 2)
    try:
        pval = regularized_beta(0.5 * df, 0.5, x)
    except ValueError:
        return np.nan
    return min(max(pval, 0.0), 1.0)


def _ensure_dataframe(rows: List[Mapping[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_COMMON_COLUMNS)
    df = pd.DataFrame(rows)
    for col in _COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[_COMMON_COLUMNS]
    df["sig_5pct"] = df["p_value"].astype(float) <= 0.05
    df.loc[~np.isfinite(df["p_value"].astype(float)), "sig_5pct"] = False
    return df


def _apply_fdr(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    if df.empty:
        return df

    def adjust(group: pd.DataFrame) -> pd.DataFrame:
        pvals = group["p_value"].astype(float).to_numpy()
        if not np.isfinite(pvals).any():
            group["q_value"] = np.nan
            group["sig_fdr_5pct"] = False
            return group
        res = fdr_bh(pvals, alpha=0.05, method="BH")
        group["q_value"] = res.q_values
        group["sig_fdr_5pct"] = res.rejected
        return group

    import warnings

    groupby_obj = df.groupby(list(group_cols), dropna=False, group_keys=False)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="DataFrameGroupBy.apply operated on the grouping columns",
            category=FutureWarning,
        )
        adjusted = groupby_obj.apply(adjust)
    adjusted = adjusted.reset_index(drop=True)
    return adjusted


def _normalise_filtered_months(data: Any) -> Optional[str]:
    if data is None:
        return None
    if isinstance(data, str):
        return ",".join(part.strip() for part in data.split(",") if part.strip())
    try:
        iter(data)
    except TypeError:
        return None
    values: List[int] = []
    for item in data:
        if pd.isna(item):
            continue
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    values = sorted(set(values))
    return ",".join(str(v) for v in values) if values else None


def _extract_metadata(obj: Mapping[str, Any]) -> Dict[str, Any]:
    return _Metadata.from_obj(obj).to_dict()


def _base_row(
    screen_type: str,
    ticker: Optional[str],
    region: Optional[str],
    exchange: Optional[str],
    slice_name: Optional[str],
    metric: str,
    value: Any,
    n_obs: Optional[int],
    p_value: Optional[float],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    row = {
        "screen_type": screen_type,
        "ticker": ticker,
        "region": region,
        "exchange": exchange,
        "slice": slice_name,
        "metric": metric,
        "value": value,
        "n_observations": n_obs,
        "p_value": p_value,
        "q_value": np.nan,
        "sig_5pct": False,
        "sig_fdr_5pct": False,
    }
    row.update(metadata)
    return row


def _iterate_entries(obj: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    if not isinstance(obj, Mapping):
        return []
    entries = obj.get("results")
    if isinstance(entries, Mapping):
        return [entries]
    if isinstance(entries, Iterable) and not isinstance(entries, (str, bytes)):
        return list(entries)
    entries = obj.get("data")
    if isinstance(entries, Mapping):
        return [entries]
    if isinstance(entries, Iterable) and not isinstance(entries, (str, bytes)):
        return list(entries)
    return []


def export_seasonality_tidy(obj: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten seasonality screen results into a tidy DataFrame.

    Parameters
    ----------
    obj:
        Nested dictionary produced by a seasonality screen.

    Returns
    -------
    pandas.DataFrame
        Long-form table where each row represents a specific metric for a
        ticker/slice combination.

    Examples
    --------
    >>> raw = {"results": [{"ticker": "ZC", "slice": "fall", "t_stat": 2.5, "n": 40}]}
    >>> export_seasonality_tidy(raw).loc[:, ["ticker", "metric"]].head(1)
      ticker       metric
    0     ZC      t_stat
    """

    metadata = _extract_metadata(obj)
    rows: List[Dict[str, Any]] = []
    for entry in _iterate_entries(obj):
        if not isinstance(entry, Mapping):
            continue
        region = entry.get("region")
        exchange = entry.get("exchange")
        base_slice = entry.get("slice")
        ticker = entry.get("ticker")
        n_obs = entry.get("n") or entry.get("n_obs")
        t_stat = entry.get("t_stat")
        p_value = entry.get("p_value")
        if p_value is None and t_stat is not None and n_obs:
            p_value = _student_t_two_sided_pvalue(t_stat, max(int(n_obs) - 1, 1))
        metrics = {
            k: entry.get(k)
            for k in (
                "mean_return",
                "median_return",
                "t_stat",
                "p_value",
                "hit_rate",
            )
            if k in entry
        }
        for metric_name, value in metrics.items():
            rows.append(
                _base_row(
                    "seasonality",
                    ticker,
                    region,
                    exchange,
                    base_slice,
                    metric_name,
                    value,
                    n_obs,
                    p_value,
                    metadata,
                )
            )
        if entry.get("strongest_patterns"):
            for pattern in entry["strongest_patterns"]:
                rows.append(
                    _base_row(
                        "seasonality",
                        ticker,
                        region,
                        exchange,
                        base_slice,
                        "strongest_pattern",
                        pattern,
                        n_obs,
                        np.nan,
                        metadata,
                    )
                )
        breakdown = entry.get("month_breakdown") or []
        if isinstance(breakdown, Mapping):
            breakdown = breakdown.values()
        for month_info in breakdown:
            if not isinstance(month_info, Mapping):
                continue
            month = month_info.get("month") or month_info.get("key")
            month_slice = f"month_{int(month):02d}" if pd.notna(month) else base_slice
            month_n = month_info.get("n") or month_info.get("n_obs")
            month_t = month_info.get("t_stat")
            month_p = month_info.get("p_value")
            if month_p is None and month_t is not None and month_n:
                month_p = _student_t_two_sided_pvalue(month_t, max(int(month_n) - 1, 1))
            for metric_name in ("mean", "median", "t_stat", "p_value"):
                if metric_name not in month_info:
                    continue
                value = month_info.get(metric_name)
                metric = (
                    "mean_return" if metric_name == "mean" else
                    "median_return" if metric_name == "median" else
                    metric_name
                )
                rows.append(
                    _base_row(
                        "seasonality",
                        ticker,
                        region,
                        exchange,
                        month_slice,
                        metric,
                        value,
                        month_n,
                        month_p,
                        metadata,
                    )
                )

    df = _ensure_dataframe(rows)
    if not df.empty:
        df = _apply_fdr(df, ["region", "slice"])
    return df


def export_momentum_tidy(obj: Mapping[str, Any]) -> pd.DataFrame:
    """Convert momentum screen results to tidy long-form output.

    Parameters
    ----------
    obj:
        Mapping describing the result of a momentum screen.

    Returns
    -------
    pandas.DataFrame
        Long-form representation keyed by ticker, session and metric.
    """

    metadata = _extract_metadata(obj)
    rows: List[Dict[str, Any]] = []
    for entry in _iterate_entries(obj):
        if not isinstance(entry, Mapping):
            continue
        ticker = entry.get("ticker")
        region = entry.get("region")
        exchange = entry.get("exchange")
        session = entry.get("session") or entry.get("slice")
        n_obs = entry.get("n") or entry.get("n_obs")
        t_stat = entry.get("t_stat")
        p_value = entry.get("p_value")
        if p_value is None and t_stat is not None and n_obs:
            p_value = _student_t_two_sided_pvalue(t_stat, max(int(n_obs) - 1, 1))
        metrics = {
            k: entry.get(k)
            for k in (
                "mean_return",
                "median_return",
                "t_stat",
                "p_value",
                "hit_rate",
            )
            if k in entry
        }
        for metric_name, value in metrics.items():
            rows.append(
                _base_row(
                    "momentum",
                    ticker,
                    region,
                    exchange,
                    session,
                    metric_name,
                    value,
                    n_obs,
                    p_value,
                    metadata,
                )
            )
        filt = _normalise_filtered_months(entry.get("filtered_months"))
        if filt:
            rows.append(
                _base_row(
                    "momentum",
                    ticker,
                    region,
                    exchange,
                    session,
                    "filtered_months",
                    filt,
                    n_obs,
                    np.nan,
                    metadata,
                )
            )
    df = _ensure_dataframe(rows)
    if not df.empty:
        df = _apply_fdr(df, ["ticker", "slice"])
    return df


def export_correlations_tidy(obj: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten correlation screen outputs.

    Parameters
    ----------
    obj:
        Mapping containing correlation pairs.

    Returns
    -------
    pandas.DataFrame
        Tidy representation with BH adjusted p-values per ticker/container.
    """

    metadata = _extract_metadata(obj)
    rows: List[Dict[str, Any]] = []
    entries = _iterate_entries(obj)
    if not entries and isinstance(obj, Mapping):
        entries = obj.get("pairs", [])
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        ticker = entry.get("ticker")
        container = entry.get("container") or entry.get("slice")
        x = entry.get("x")
        y = entry.get("y")
        corr = entry.get("r") or entry.get("corr")
        n_obs = entry.get("n") or entry.get("n_obs")
        p_value = entry.get("p_value")
        if p_value is None and corr is not None and n_obs and n_obs > 2:
            r = float(corr)
            dfree = int(n_obs) - 2
            if abs(r) < 1 and dfree > 0:
                t_stat = r * np.sqrt(dfree / (1 - r ** 2))
                p_value = _student_t_two_sided_pvalue(t_stat, dfree)
        rows.append(
            _base_row(
                "correlation",
                ticker,
                entry.get("region"),
                entry.get("exchange"),
                container,
                "correlation",
                corr,
                n_obs,
                p_value,
                metadata,
            )
        )
        if x is not None:
            rows.append(
                _base_row(
                    "correlation",
                    ticker,
                    entry.get("region"),
                    entry.get("exchange"),
                    container,
                    "x",
                    x,
                    n_obs,
                    np.nan,
                    metadata,
                )
            )
        if y is not None:
            rows.append(
                _base_row(
                    "correlation",
                    ticker,
                    entry.get("region"),
                    entry.get("exchange"),
                    container,
                    "y",
                    y,
                    n_obs,
                    np.nan,
                    metadata,
                )
            )
    df = _ensure_dataframe(rows)
    if not df.empty:
        df = _apply_fdr(df, ["ticker", "slice"])
    return df


def export_all_tidy(ag_results: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    """Run all tidy exporters and return a mapping of DataFrames.

    Parameters
    ----------
    ag_results:
        Aggregated screen results keyed by screen type.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Individual tidy tables for seasonality, momentum and correlations.
    """

    outputs: Dict[str, pd.DataFrame] = {}
    if not isinstance(ag_results, Mapping):
        return {"seasonality": pd.DataFrame(), "momentum": pd.DataFrame(), "correlations": pd.DataFrame()}
    if ag_results.get("seasonality") is not None:
        outputs["seasonality"] = export_seasonality_tidy(ag_results["seasonality"])
    else:
        outputs["seasonality"] = pd.DataFrame()
    if ag_results.get("momentum") is not None:
        outputs["momentum"] = export_momentum_tidy(ag_results["momentum"])
    else:
        outputs["momentum"] = pd.DataFrame()
    if ag_results.get("correlations") is not None:
        outputs["correlations"] = export_correlations_tidy(ag_results["correlations"])
    else:
        outputs["correlations"] = pd.DataFrame()
    return outputs


def write_tidy(dfs: Mapping[str, pd.DataFrame], folder: Path) -> None:
    """Persist tidy DataFrames to ``folder`` as CSV and (optionally) Parquet.

    Parameters
    ----------
    dfs:
        Mapping of names to tidy DataFrames.
    folder:
        Destination directory.  It is created if missing.
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in dfs.items():
        if df is None or df.empty:
            continue
        csv_path = folder / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        parquet_path = folder / f"{name}.parquet"
        try:
            df.to_parquet(parquet_path, index=False)
        except (ImportError, ValueError) as exc:
            LOGGER.warning("Parquet output for %s skipped: %s", name, exc)


def _load_ag_results(path: Path) -> Mapping[str, Any]:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, Mapping):
        return data
    raise TypeError("Loaded object is not a mapping of screen results")


def _main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export tidy screen outputs")
    parser.add_argument("--input", required=True, type=Path, help="Pickle file containing aggregated results")
    parser.add_argument("--out", required=True, type=Path, help="Output folder for tidy files")
    args = parser.parse_args(list(argv) if argv is not None else None)
    results = _load_ag_results(args.input)
    dfs = export_all_tidy(results)
    write_tidy(dfs, args.out)
    summary = {name: len(df) for name, df in dfs.items()}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
