from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from contextlib import nullcontext
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .prediction_to_position import PredictionToPosition
from .gpu_acceleration import (
    GPU_AVAILABLE,
    _cupy_cummax,
    gpu_backtest_threshold,
    gpu_batch_threshold_sweep,
    to_backend_array,
    to_cpu,
)

@dataclass
class BacktestSummary:
    """Container summarising threshold backtest results."""

    total_return: float
    mean_return: float
    hit_rate: float
    sharpe: float
    max_drawdown: float
    trades: int

@dataclass
class SignalSpec:
    """
    Configuration for turning a feature into entry signals.

    - signal_col: column containing the feature/signal
    - trigger: threshold; long if signal >= trigger; (optionally short if <= -trigger)
    - allow_short: if True, symmetric short entries are allowed
    """
    signal_col: str = "signal"
    trigger: float = 0.0
    allow_short: bool = False


@dataclass
class ExecutionPolicy:
    """
    Execution rules for FeatureBacktester.

    - price_in / price_out: columns used for entry/exit prices
    - bias:
        'long'   -> take only long signals
        'short'  -> take only short signals
        'signed' -> direction = sign(signal)
    - exit_policy:
        'bars'          -> hold fixed number of bars
        'session_close' -> exit at same-session close
        'next_day_close'-> exit at next calendar day close
        'monday_close'  -> exit at next Monday session close
        'target_pct'    -> exit when pnl >= target_pct (per trade)
    - holding_bars: used when exit_policy == 'bars'
    - target_pct: threshold for 'target_pct' exit policy (must be > 0)
    - pattern_gate_col: optional gate column; entries only allowed when this column != 0
    - tz: optional timezone for session-based exits; currently only used for future extensions
    - weekend_aware: when True, auto-detects Friday entries with 'session_close' and switches
                     to 'monday_close' exit to handle weekend hedging patterns correctly (default: True)
    """
    price_in: str = "Close"
    price_out: str = "Close"
    bias: Literal["long", "short", "signed"] = "long"
    exit_policy: Literal[
        "bars",
        "session_close",
        "next_day_close",
        "monday_close",
        "target_pct",
    ] = "session_close"
    holding_bars: int = 1
    target_pct: float = 0.0
    pattern_gate_col: Optional[str] = None
    tz: Optional[str] = None
    weekend_aware: bool = True  # Auto-switch session_close to monday_close for Friday entries


class ScreenerBacktester:
    """Lightweight backtester operating on ``ScreenerPipeline.build_xy`` outputs.

    Args:
        annualisation: Number of periods per year for Sharpe ratio calculation
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        use_gpu: Enable GPU acceleration if available (default: True)
        gpu_device_id: GPU device ID to use for computations (default: 0)
    """

    def __init__(
        self,
        *,
        annualisation: int = 252,
        risk_free_rate: float = 0.0,
        use_gpu: bool = True,
        gpu_device_id: int = 0,
        gpu_stream: Optional[Any] = None,
    ) -> None:
        self.annualisation = annualisation
        self.risk_free_rate = risk_free_rate
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu_device_id = gpu_device_id
        self.gpu_stream = gpu_stream
        self._collision_resolver = PredictionToPosition()
        self._prepared_cache: Dict[Tuple[Any, ...], "ScreenerBacktester._PreparedFrame"] = {}

    @dataclass
    class _PreparedFrame:
        frame: pd.DataFrame
        returns_x: np.ndarray
        returns_y: np.ndarray
        trade_index: pd.Index
        correlation: Optional[pd.Series]
        group_field: Optional[str]

    @staticmethod
    def ranking_score(summary: BacktestSummary) -> float:
        """Return a stability-aware score for ranking backtests."""

        drawdown = abs(summary.max_drawdown)
        if drawdown == 0:
            return float("inf") if summary.total_return > 0 else float("-inf")
        return summary.total_return / drawdown

    @classmethod
    def rank_results(
        cls, results: Mapping[str, Mapping[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Order backtest results by ``total_return / max_drawdown`` descending."""

        def _score(entry: Mapping[str, Any]) -> float:
            summary = entry.get("summary")
            return cls.ranking_score(summary) if isinstance(summary, BacktestSummary) else float("-inf")

        scored = [(key, _score(value)) for key, value in results.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _prepare_frame_for_backtest(
        self,
        xy: pd.DataFrame,
        *,
        use_side_hint: bool,
        group_field: Optional[str],
        prediction_resolver: Optional["PredictionToPosition"],
    ) -> Optional["ScreenerBacktester._PreparedFrame"]:
        required = {"returns_x", "returns_y"}
        missing = required.difference(xy.columns)
        if missing:
            raise KeyError(f"XY frame missing required columns: {sorted(missing)}")

        valid_mask = xy[["returns_x", "returns_y"]].notna().all(axis=1)
        frame = xy.loc[valid_mask]

        if prediction_resolver is not None and not frame.empty:
            frame = prediction_resolver.aggregate(frame)

        if frame.empty:
            return None

        frame = self._collision_resolver.resolve(frame, group_field=group_field)
        frame = self._drop_duplicate_rows(frame)

        returns_x = np.asarray(frame["returns_x"].to_numpy(copy=False), dtype=float)
        returns_y = np.asarray(frame["returns_y"].to_numpy(copy=False), dtype=float)

        if "ts_decision" in frame.columns:
            trade_index = pd.to_datetime(frame["ts_decision"], errors="coerce")
            trade_index.name = "ts_decision"
        else:
            trade_index = pd.Index(frame.index, name=frame.index.name or "row")

        correlation = frame["correlation"] if (use_side_hint and "correlation" in frame.columns) else None

        return self._PreparedFrame(
            frame=frame,
            returns_x=returns_x,
            returns_y=returns_y,
            trade_index=trade_index,
            correlation=correlation,
            group_field=group_field,
        )

    def _get_prepared_frame(
        self,
        xy: pd.DataFrame,
        *,
        use_side_hint: bool,
        group_field: Optional[str],
        prediction_resolver: Optional["PredictionToPosition"],
        cache_key: Optional[Tuple[Any, ...]] = None,
    ) -> Optional["ScreenerBacktester._PreparedFrame"]:
        """Prepare and optionally cache a frame for repeated backtests."""

        if cache_key is None:
            cache_key = (
                id(xy),
                use_side_hint,
                group_field,
                bool(prediction_resolver),
            )

        if cache_key in self._prepared_cache:
            return self._prepared_cache[cache_key]

        prepared = self._prepare_frame_for_backtest(
            xy,
            use_side_hint=use_side_hint,
            group_field=group_field,
            prediction_resolver=prediction_resolver,
        )

        if prepared is not None:
            self._prepared_cache[cache_key] = prepared

        return prepared

    def _finalize_result(
        self,
        prepared: "ScreenerBacktester._PreparedFrame",
        positions_array: np.ndarray,
        pnl_array: np.ndarray,
        *,
        cumulative_array: Optional[np.ndarray] = None,
        drawdown_array: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        frame = prepared.frame
        trade_index = prepared.trade_index

        raw_positions = pd.Series(positions_array, index=frame.index)
        raw_pnl = pd.Series(pnl_array, index=frame.index)
        signal_mask = positions_array != 0.0

        trade_rows = frame.loc[signal_mask].copy()
        if not trade_rows.empty and "ts_decision" in trade_rows.columns:
            trade_rows["_trade_day"] = pd.to_datetime(
                trade_rows["ts_decision"], errors="coerce"
            ).dt.normalize()
        else:
            trade_rows["_trade_day"] = pd.NaT

        if cumulative_array is None:
            raw_cumulative = raw_pnl.cumsum()
            rolling_max = raw_cumulative.cummax()
            drawdown_series = raw_cumulative - rolling_max
            max_drawdown = float(drawdown_series.min()) if not drawdown_series.empty else 0.0
        else:
            raw_cumulative = pd.Series(cumulative_array, index=frame.index)
            drawdown = drawdown_array if drawdown_array is not None else raw_cumulative - np.maximum.accumulate(raw_cumulative)
            max_drawdown = float(drawdown.min()) if drawdown.size else 0.0

        def _assign_trade_index(series: pd.Series) -> pd.Series:
            aligned = series.copy()
            aligned.index = trade_index
            return aligned

        positions = _assign_trade_index(raw_positions)
        pnl = _assign_trade_index(raw_pnl)
        cumulative = _assign_trade_index(raw_cumulative)
        predictor_values = _assign_trade_index(frame["returns_x"])

        if "correlation" in frame.columns:
            correlation_values = _assign_trade_index(frame["correlation"])
        else:
            correlation_values = pd.Series(index=trade_index, dtype=float)

        trades = int(signal_mask.sum())
        if not trade_rows.empty:
            trade_days = trade_rows["_trade_day"].dropna()
            if not trade_days.empty:
                if "_group_members" in trade_rows.columns:
                    member_lists = trade_rows["_group_members"]
                    member_lists = member_lists.apply(
                        lambda v: v if isinstance(v, (list, tuple, set)) else ([] if pd.isna(v) else [v])
                    )
                    trades = int(member_lists.apply(len).sum())
                elif prepared.group_field and prepared.group_field in trade_rows.columns:
                    # Vectorized group counting - avoid duplicate row operations
                    combos = trade_rows.loc[trade_days.index, ["_trade_day", prepared.group_field]]
                    valid_mask = combos[prepared.group_field].notna()
                    trades = int(combos[valid_mask].drop_duplicates().shape[0])
                    if not valid_mask.all():
                        trades += int(combos.loc[~valid_mask, "_trade_day"].nunique())
                else:
                    trades = int(trade_days.nunique())

        mean_return = float(pnl.mean()) if not pnl.empty else 0.0
        total_return = float(pnl.sum())
        hit_rate = float((pnl > 0).mean()) if not pnl.empty else np.nan
        std_return = float(pnl.std(ddof=0))
        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(self.annualisation)
        else:
            sharpe = np.nan

        if isinstance(trade_index, pd.DatetimeIndex):
            valid = trade_index.notna()
            if valid.any():
                monthly = (
                    pnl.iloc[valid]
                    .groupby(trade_index[valid].to_period("M"))
                    .sum()
                    .astype(float)
                )
                daily_pnl = (
                    pnl.iloc[valid]
                    .groupby(trade_index[valid].normalize())
                    .sum()
                    .astype(float)
                )
            else:
                monthly = pd.Series(dtype=float)
                daily_pnl = pd.Series(dtype=float)
        else:
            monthly = pd.Series(dtype=float)
            daily_pnl = pd.Series(dtype=float)

        summary = BacktestSummary(
            total_return=total_return,
            mean_return=mean_return,
            hit_rate=hit_rate,
            sharpe=float(sharpe) if np.isfinite(sharpe) else np.nan,
            max_drawdown=max_drawdown,
            trades=trades,
        )

        result: Dict[str, Any] = {
            "pnl": pnl,
            "positions": positions,
            "summary": summary,
            "monthly": monthly,
            "cumulative": cumulative,
            "daily_pnl": daily_pnl,
            "predictor_values": predictor_values,
            "correlation": correlation_values,
        }

        return result

    def threshold(
        self,
        xy: pd.DataFrame,
        *,
        threshold: float = 0.0,
        use_side_hint: bool = True,
        group_field: Optional[str] = None,
        prediction_resolver: Optional["PredictionToPosition"] = None,
        prepared: Optional["ScreenerBacktester._PreparedFrame"] = None,
        cache_prepared: bool = False,
    ) -> Dict[str, Any]:
        def _empty_result() -> Dict[str, Any]:
            empty = pd.Series(dtype=float)
            return {
                "pnl": empty,
                "positions": empty,
                "summary": BacktestSummary(0.0, 0.0, np.nan, np.nan, 0.0, 0),
                "monthly": pd.Series(dtype=float),
                "cumulative": empty,
                "daily_pnl": pd.Series(dtype=float),
                "predictor_values": empty,
                "correlation": empty,
            }

        if xy.empty:
            return _empty_result()

        cache_key = (
            id(xy),
            threshold,
            use_side_hint,
            group_field,
            bool(prediction_resolver),
        ) if cache_prepared else None

        if prepared is None:
            prepared = self._get_prepared_frame(
                xy,
                use_side_hint=use_side_hint,
                group_field=group_field,
                prediction_resolver=prediction_resolver,
                cache_key=cache_key,
            )

        if prepared is None:
            return _empty_result()

        frame = prepared.frame

        if self.use_gpu:
            correlation = prepared.correlation
            stream_cm = self.gpu_stream if self.gpu_stream is not None else nullcontext()
            with stream_cm:
                raw_positions_backend, raw_pnl_backend, xp = gpu_backtest_threshold(
                    returns_x=prepared.returns_x,
                    returns_y=prepared.returns_y,
                    correlation=correlation,
                    threshold=threshold,
                    use_side_hint=use_side_hint,
                    use_gpu=True,
                    device_id=self.gpu_device_id,
                    stream=self.gpu_stream,
                    return_backend=True,
                )

                raw_cumulative_backend = xp.cumsum(raw_pnl_backend)
                rolling_max_backend = (
                    _cupy_cummax(raw_cumulative_backend, xp)
                    if xp.__name__ == "cupy"
                    else xp.maximum.accumulate(raw_cumulative_backend)
                )
                drawdown_backend = raw_cumulative_backend - rolling_max_backend

            positions_array = to_cpu(raw_positions_backend)
            pnl_array = to_cpu(raw_pnl_backend)
            cumulative_array = to_cpu(raw_cumulative_backend)
            drawdown_array = to_cpu(drawdown_backend)

            result = self._finalize_result(
                prepared,
                positions_array,
                pnl_array,
                cumulative_array=cumulative_array,
                drawdown_array=drawdown_array,
            )
        else:
            direction = np.sign(prepared.returns_x)
            if use_side_hint and prepared.correlation is not None:
                corr_sign = np.sign(np.asarray(prepared.correlation.to_numpy(copy=False), dtype=float))
                direction = np.sign(prepared.returns_x * corr_sign)
            if use_side_hint and "side_hint" in frame.columns:
                hinted = np.asarray(frame["side_hint"].to_numpy(copy=False), dtype=float)
                direction = np.where(hinted != 0, hinted, direction)
            if prediction_resolver is not None and "prediction_position" in frame.columns:
                hinted = np.asarray(
                    frame["prediction_position"].to_numpy(copy=False), dtype=float
                )
                direction = np.where(hinted != 0, hinted, direction)

            signal_mask = np.abs(prepared.returns_x) >= float(threshold)
            positions_array = np.where(signal_mask, direction, 0.0)
            pnl_array = positions_array * prepared.returns_y

            result = self._finalize_result(prepared, positions_array, pnl_array)

        if group_field and group_field in frame.columns:
            trade_rows = frame.loc[positions_array != 0].copy()
            if not trade_rows.empty:
                if "_group_members" in trade_rows.columns:
                    member_lists = trade_rows["_group_members"]
                    if group_field in trade_rows.columns:
                        fallback_values = trade_rows[group_field]
                        member_lists = member_lists.where(
                            member_lists.astype(bool),
                            fallback_values.apply(lambda v: [] if pd.isna(v) else [v]),
                        )
                    member_lists = member_lists.apply(
                        lambda v: v if isinstance(v, (list, tuple, set)) else ([] if pd.isna(v) else [v])
                    )
                else:
                    member_lists = trade_rows[group_field].apply(
                        lambda v: [] if pd.isna(v) else [v]
                    )

                member_lengths = member_lists.apply(len)
                valid_members = member_lengths > 0
                if valid_members.any():
                    raw_positions = pd.Series(positions_array, index=frame.index)
                    raw_pnl = pd.Series(pnl_array, index=frame.index)
                    expanded = trade_rows.loc[valid_members].assign(
                        _members=member_lists[valid_members],
                        _share=1.0 / member_lengths[valid_members],
                        _trade_day=pd.to_datetime(
                            trade_rows.loc[valid_members.index, "ts_decision"], errors="coerce"
                        ).dt.normalize()
                        if "ts_decision" in trade_rows.columns
                        else pd.NaT,
                        _position=raw_positions.loc[valid_members.index].to_numpy(),
                        _pnl=raw_pnl.loc[valid_members.index].to_numpy(),
                    )
                    exploded = expanded.explode("_members")
                    exploded = exploded.dropna(subset=["_members"])
                    if not exploded.empty:
                        exploded["_weighted_pnl"] = (
                            exploded["_pnl"].to_numpy() * exploded["_share"].to_numpy()
                        )
                        pnl_sum = exploded.groupby("_members")["_weighted_pnl"].sum()
                        share_count = exploded.groupby("_members")["_share"].sum()
                        nonzero_positions = exploded[exploded["_position"] != 0]
                        trade_day_counts = nonzero_positions.dropna(subset=["_trade_day"]).groupby("_members")["_trade_day"].nunique()
                        fallback_counts = (
                            nonzero_positions[nonzero_positions["_trade_day"].isna()]
                            .groupby("_members")
                            .size()
                        )
                        trade_counts = trade_day_counts.add(fallback_counts, fill_value=0).astype(int)

                        # Vectorized formatting - avoid per-member iteration
                        denominators = share_count.replace(0, 1.0)
                        formatted = {
                            member: {
                                "trades": int(trade_counts.get(member, 0)),
                                "total_return": float(pnl_sum.loc[member]),
                                "mean_return": float(pnl_sum.loc[member] / denominators.loc[member]),
                            }
                            for member in pnl_sum.index
                        }
                        result["group_breakdown"] = formatted
                        result["group_field"] = group_field

        return result

    def batch_patterns(
        self,
        xy_map: Mapping[str, pd.DataFrame],
        *,
        threshold: float = 0.0,
        use_side_hint: bool = True,
        group_field: Optional[str] = None,
        prediction_resolver: Optional["PredictionToPosition"] = None,
        parallel_prep: bool = False,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Backtest multiple screen outputs concurrently using shared GPU transfers.

        Args:
            xy_map: Dictionary mapping pattern names to DataFrames
            threshold: Signal threshold value
            use_side_hint: Whether to use correlation hints for position direction
            group_field: Optional field for grouping trades
            prediction_resolver: Optional resolver for overlapping predictions
            parallel_prep: Enable parallel data preparation using multiprocessing (default: False)
            max_workers: Maximum number of parallel workers (default: CPU count)
        """

        def _empty_result() -> Dict[str, Any]:
            empty = pd.Series(dtype=float)
            return {
                "pnl": empty,
                "positions": empty,
                "summary": BacktestSummary(0.0, 0.0, np.nan, np.nan, 0.0, 0),
                "monthly": pd.Series(dtype=float),
                "cumulative": empty,
                "daily_pnl": pd.Series(dtype=float),
                "predictor_values": empty,
                "correlation": empty,
            }

        # Prepare frames - optionally in parallel
        prepared_entries: Dict[str, Optional[ScreenerBacktester._PreparedFrame]] = {}

        # Use parallel processing if explicitly enabled
        # Recommended for: many patterns (20+) or when batching diverse datasets
        use_parallel = parallel_prep and len(xy_map) > 1

        if use_parallel:
            # Parallel preparation using ProcessPoolExecutor
            from functools import partial
            prep_fn = partial(
                self._prepare_single_pattern,
                use_side_hint=use_side_hint,
                group_field=group_field,
                prediction_resolver=prediction_resolver,
            )

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(prep_fn, key, xy): key
                    for key, xy in xy_map.items()
                }

                for future in as_completed(futures):
                    try:
                        key, prepared = future.result()
                        prepared_entries[key] = prepared
                    except Exception:
                        prepared_entries[futures[future]] = None
        else:
            # Sequential preparation (original behavior)
            for key, xy in xy_map.items():
                if xy is None or xy.empty:
                    prepared_entries[key] = None
                    continue

                try:
                    prepared_entries[key] = self._prepare_frame_for_backtest(
                        xy,
                        use_side_hint=use_side_hint,
                        group_field=group_field,
                        prediction_resolver=prediction_resolver,
                    )
                except Exception:
                    prepared_entries[key] = None

        # Group prepared frames by length to maintain alignment during batching
        length_buckets: Dict[int, List[Tuple[str, ScreenerBacktester._PreparedFrame]]] = {}
        for key, prepared in prepared_entries.items():
            if prepared is None:
                continue
            length_buckets.setdefault(len(prepared.frame), []).append((key, prepared))

        results: Dict[str, Dict[str, Any]] = {}

        for key, prepared in prepared_entries.items():
            if prepared is None and key not in results:
                results[key] = _empty_result()

        for length, bucket in length_buckets.items():
            if length == 0:
                continue

            returns_x_stack = np.vstack([entry.returns_x for _, entry in bucket])
            returns_y_stack = np.vstack([entry.returns_y for _, entry in bucket])

            if use_side_hint:
                corr_stack = []
                for _, entry in bucket:
                    if entry.correlation is None:
                        corr_stack.append(np.ones(length, dtype=float))
                    else:
                        corr_stack.append(np.asarray(entry.correlation.to_numpy(copy=False), dtype=float))
                correlation_matrix = np.vstack(corr_stack)
            else:
                correlation_matrix = None

            backend_stream = self.gpu_stream if self.gpu_stream is not None else nullcontext()
            with backend_stream:
                rx_backend, xp = to_backend_array(
                    returns_x_stack, use_gpu=self.use_gpu, device_id=self.gpu_device_id, stream=self.gpu_stream
                )
                ry_backend, _ = to_backend_array(
                    returns_y_stack, use_gpu=self.use_gpu, device_id=self.gpu_device_id, stream=self.gpu_stream
                )

                if correlation_matrix is not None:
                    corr_backend, _ = to_backend_array(
                        correlation_matrix, use_gpu=self.use_gpu, device_id=self.gpu_device_id, stream=self.gpu_stream
                    )
                    adjusted_x = rx_backend * xp.sign(corr_backend)
                else:
                    adjusted_x = rx_backend

                positions_backend = xp.where(
                    adjusted_x >= threshold,
                    1.0,
                    xp.where(adjusted_x <= -threshold, -1.0, 0.0),
                )
                pnl_backend = positions_backend * ry_backend
                cumulative_backend = xp.cumsum(pnl_backend, axis=1)
                rolling_max_backend = (
                    _cupy_cummax(cumulative_backend, xp) if xp.__name__ == "cupy" else xp.maximum.accumulate(cumulative_backend, axis=1)
                )
                drawdown_backend = cumulative_backend - rolling_max_backend

            positions_cpu = to_cpu(positions_backend)
            pnl_cpu = to_cpu(pnl_backend)
            cumulative_cpu = to_cpu(cumulative_backend)
            drawdown_cpu = to_cpu(drawdown_backend)

            for idx, (key, entry) in enumerate(bucket):
                results[key] = self._finalize_result(
                    entry,
                    positions_cpu[idx],
                    pnl_cpu[idx],
                    cumulative_array=cumulative_cpu[idx],
                    drawdown_array=drawdown_cpu[idx],
                )

        return results

    def batch_threshold_sweep(
        self,
        xy: pd.DataFrame,
        thresholds: Union[np.ndarray, List[float], Tuple[float, ...]],
        *,
        use_side_hint: bool = True,
        group_field: Optional[str] = None,
        prediction_resolver: Optional["PredictionToPosition"] = None,
        prepared: Optional["ScreenerBacktester._PreparedFrame"] = None,
        cache_prepared: bool = False,
    ) -> Dict[float, Dict[str, Any]]:
        """Run backtests for multiple threshold values with GPU batching.

        Performs threshold backtesting for multiple threshold values simultaneously
        when GPU is available, significantly reducing CPU↔GPU transfer overhead
        compared to calling threshold() in a loop.

        Args:
            xy: DataFrame with returns_x, returns_y columns from ScreenerPipeline.build_xy
            thresholds: Array of threshold values to test (e.g., [0.0, 0.5, 1.0, 1.5, 2.0])
            use_side_hint: Whether to use correlation sign for position direction
            group_field: Optional field for grouping trades
            prediction_resolver: Optional resolver for overlapping predictions

        Returns:
            Dictionary mapping each threshold to a results dict (same format as threshold())

        Example:
            >>> tester = ScreenerBacktester(use_gpu=True)
            >>> thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]
            >>> results = tester.batch_threshold_sweep(xy, thresholds)
            >>> for threshold, result in results.items():
            ...     print(f"Threshold {threshold}: Sharpe = {result['summary'].sharpe:.2f}")
        """

        # Convert thresholds to array
        thresholds_array = np.asarray(thresholds, dtype=float).ravel()
        if len(thresholds_array) == 0:
            return {}

        cache_key = (
            id(xy),
            tuple(thresholds_array.tolist()),
            use_side_hint,
            group_field,
            bool(prediction_resolver),
        ) if cache_prepared else None

        prepared_frame = prepared
        if prepared_frame is None:
            prepared_frame = self._get_prepared_frame(
                xy,
                use_side_hint=use_side_hint,
                group_field=group_field,
                prediction_resolver=prediction_resolver,
                cache_key=cache_key,
            )

        if prepared_frame is None:
            empty_result = self.threshold(
                xy,
                threshold=0.0,
                use_side_hint=use_side_hint,
                group_field=group_field,
                prediction_resolver=prediction_resolver,
            )
            return {float(t): empty_result for t in thresholds_array}

        frame = prepared_frame.frame
        returns_x = prepared_frame.returns_x
        returns_y = prepared_frame.returns_y

        # Get GPU batch results
        correlation = prepared_frame.correlation if (use_side_hint and prepared_frame.correlation is not None) else None

        gpu_results = gpu_batch_threshold_sweep(
            returns_x=returns_x,
            returns_y=returns_y,
            thresholds=thresholds_array,
            correlation=correlation,
            use_side_hint=use_side_hint,
            use_gpu=self.use_gpu,
            device_id=self.gpu_device_id,
            stream=self.gpu_stream,
        )

        # Process each threshold's results into full backtest format
        final_results = {}

        for threshold_val, gpu_metrics in gpu_results.items():
            positions_array = gpu_metrics['positions']
            pnl_array = gpu_metrics['pnl']
            cumulative_array = gpu_metrics['cumulative']
            drawdown_array = cumulative_array - np.maximum.accumulate(cumulative_array)

            final_results[threshold_val] = self._finalize_result(
                prepared_frame,
                positions_array,
                pnl_array,
                cumulative_array=cumulative_array,
                drawdown_array=drawdown_array,
            )

        return final_results

    @staticmethod
    def _prepare_single_pattern(
        key: str,
        xy: pd.DataFrame,
        use_side_hint: bool,
        group_field: Optional[str],
        prediction_resolver: Optional["PredictionToPosition"],
    ) -> Tuple[str, Optional["ScreenerBacktester._PreparedFrame"]]:
        """Helper for parallel preparation of patterns. Must be static for pickling."""
        if xy is None or xy.empty:
            return key, None

        try:
            # Create a temporary backtester instance for preparation
            temp_backtester = ScreenerBacktester()
            prepared = temp_backtester._prepare_frame_for_backtest(
                xy,
                use_side_hint=use_side_hint,
                group_field=group_field,
                prediction_resolver=prediction_resolver,
            )
            return key, prepared
        except Exception:
            return key, None

    @staticmethod
    def _drop_duplicate_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        subset = [
            col
            for col in [
                "ts_decision",
                "gate",
                "pattern_type",
                "returns_x",
                "returns_y",
                "side_hint",
            ]
            if col in frame.columns
        ]
        if subset:
            return frame.drop_duplicates(subset=subset).copy()
        return frame.drop_duplicates().copy()


class MomentumBacktester:
    """Utility for running light backtests on historical momentum screen outputs."""

    def __init__(
        self,
        *,
        annualisation: int = 252,
        risk_free_rate: float = 0.0,
        backtester: Optional[ScreenerBacktester] = None,
    ) -> None:
        self._backtester = backtester or ScreenerBacktester(
            annualisation=annualisation,
            risk_free_rate=risk_free_rate,
        )

    @staticmethod
    def _resolve_screen_results(
        results: Mapping[str, Any], screen_name: Optional[str]
    ) -> Mapping[str, Any]:
        if screen_name is None:
            return results
        try:
            screen_results = results[screen_name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Screen '{screen_name}' not found in results") from exc
        if not isinstance(screen_results, Mapping):  # pragma: no cover - defensive
            raise TypeError(
                f"Screen '{screen_name}' results must be a mapping, got {type(screen_results)!r}"
            )
        return screen_results

    def build_xy(
        self,
        results: Mapping[str, Any],
        *,
        screen_name: Optional[str] = None,
        session_key: str = "session_0",
        predictor: str = "closing_returns",
        target: str = "rest_of_session_returns",
        tickers: Optional[Sequence[str]] = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Convert a nested momentum screen result dictionary into an ``XY`` frame.

        Parameters
        ----------
        results
            Mapping produced by :meth:`HistoricalScreener.intraday_momentum_screen`.
        screen_name
            Optional name of the screen to extract. If ``None`` the top-level
            mapping is assumed to already be ticker keyed.
        session_key
            Session identifier such as ``"session_0"``.
        predictor
            Key inside ``return_series`` used for ``returns_x``.
        target
            Key inside ``return_series`` used for ``returns_y``.
        tickers
            Optional subset of tickers to include.
        dropna
            Whether to drop rows missing either ``returns_x`` or ``returns_y``.
        """

        if not isinstance(results, Mapping):  # pragma: no cover - defensive
            raise TypeError(f"results must be a mapping, got {type(results)!r}")

        screen_results = self._resolve_screen_results(results, screen_name)
        columns = [
            "returns_x",
            "returns_y",
            "ticker",
            "session_key",
            "screen_name",
            "ts_decision",
        ]
        frames: List[pd.DataFrame] = []
        if tickers is not None:
            iterable = list(dict.fromkeys(tickers))
        else:
            iterable = screen_results.keys()
        for ticker in iterable:
            ticker_data = screen_results.get(ticker)
            if not isinstance(ticker_data, Mapping):
                continue
            session_data = ticker_data.get(session_key)
            if not isinstance(session_data, Mapping):
                continue
            return_series = session_data.get("return_series")
            if not isinstance(return_series, Mapping):
                continue
            predictor_series = return_series.get(predictor)
            target_series = return_series.get(target)
            if not isinstance(predictor_series, pd.Series) or not isinstance(
                target_series, pd.Series
            ):
                continue
            predictor_clean = pd.to_numeric(predictor_series, errors="coerce")
            target_clean = pd.to_numeric(target_series, errors="coerce")
            combined = pd.concat(
                [
                    predictor_clean.rename("returns_x"),
                    target_clean.rename("returns_y"),
                ],
                axis=1,
                join="inner",
            )
            if dropna:
                combined = combined.dropna(subset=["returns_x", "returns_y"])
            if combined.empty:
                continue
            decision_ts = pd.to_datetime(combined.index)
            combined = combined.assign(
                ticker=ticker,
                session_key=session_key,
                screen_name=screen_name,
                ts_decision=decision_ts,
            )
            frames.append(combined)

        if not frames:
            return pd.DataFrame(columns=columns)

        combined = pd.concat(frames).reset_index(drop=True)
        return combined.drop_duplicates().reset_index(drop=True)

    def threshold_backtest(
        self,
        results: Mapping[str, Any],
        *,
        screen_name: Optional[str] = None,
        session_key: str = "session_0",
        predictor: str = "closing_returns",
        target: str = "rest_of_session_returns",
        tickers: Optional[Sequence[str]] = None,
        threshold: float = 0.0,
        use_side_hint: bool = False,
        group_field: str = "ticker",
        prediction_resolver: Optional["PredictionToPosition"] = None,
    ) -> Dict[str, Any]:
        """Run a quick threshold backtest over the extracted return pairs."""

        xy = self.build_xy(
            results,
            screen_name=screen_name,
            session_key=session_key,
            predictor=predictor,
            target=target,
            tickers=tickers,
        )
        return self._backtester.threshold(
            xy,
            threshold=threshold,
            use_side_hint=use_side_hint,
            group_field=group_field,
            prediction_resolver=prediction_resolver,
        )


class FeatureBacktester:
    """Configurable backtester operating on a full feature/price DataFrame.

    This is complementary to ScreenerBacktester:

    - ScreenerBacktester: works on ScreenerPipeline.build_xy outputs (returns_x / returns_y).
    - FeatureBacktester : works on raw features on the full time series and lets you
      define entries/exits via SignalSpec + ExecutionPolicy.
    """

    def __init__(self, *, annualisation: int = 252, risk_free_rate: float = 0.0) -> None:
        self.annualisation = annualisation
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        data: pd.DataFrame,
        *,
        signal: SignalSpec,
        execution: ExecutionPolicy,
        ts_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a backtest using the provided signal and execution policy.

        Returns a dict with:
          - pnl: per-bar pnl (booked at exit)
          - positions: entry impulses (+1/-1)
          - cumulative: cumulative pnl series
          - trades: trade-level DataFrame (ts_in/ts_out/dir/px_in/px_out/ret)
          - summary: BacktestSummary
        """
        if data is None or data.empty:
            empty = pd.Series(dtype=float)
            return {
                "pnl": empty,
                "positions": empty,
                "cumulative": empty,
                "trades": pd.DataFrame(
                    columns=["ts_in", "ts_out", "dir", "px_in", "px_out", "ret"]
                ),
                "summary": BacktestSummary(0.0, 0.0, np.nan, np.nan, 0.0, 0),
            }

        frame = data.copy()

        # --- Index handling: require a proper time axis ----------------------
        if ts_col is not None and ts_col in frame.columns:
            frame["ts"] = pd.to_datetime(frame[ts_col], errors="coerce")
            frame = frame.dropna(subset=["ts"]).set_index("ts").sort_index()
        elif isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.sort_index()
        else:
            raise TypeError("FeatureBacktester requires a DatetimeIndex or a valid ts_col.")

        # --- Build signal and pattern gate ----------------------------------
        if signal.signal_col not in frame.columns:
            raise KeyError(f"Signal column '{signal.signal_col}' not found in DataFrame.")
        sig = pd.to_numeric(frame[signal.signal_col], errors="coerce").fillna(0.0)

        if execution.pattern_gate_col is not None:
            if execution.pattern_gate_col not in frame.columns:
                raise KeyError(
                    f"pattern_gate_col '{execution.pattern_gate_col}' not found in DataFrame."
                )
            gate_s = pd.to_numeric(frame[execution.pattern_gate_col], errors="coerce").fillna(0.0)
            # Any non-zero value means the pattern is active
            pattern_mask = gate_s != 0.0
        else:
            # No gating: all rows are eligible
            pattern_mask = pd.Series(True, index=frame.index)

        # Long / short entry conditions; gate acts as a hard precondition
        long_entries = (sig >= float(signal.trigger)) & pattern_mask
        if signal.allow_short:
            short_entries = (sig <= -float(signal.trigger)) & pattern_mask
        else:
            short_entries = pd.Series(False, index=frame.index)

        # --- Direction construction -----------------------------------------
        if execution.bias == "signed":
            # direction = sign(signal) when gated; 0 otherwise
            direction = np.sign(sig).where(long_entries | short_entries, 0.0)
        elif execution.bias == "short":
            direction = -1.0 * short_entries.astype(float)
        else:  # "long"
            direction = long_entries.astype(float)

        # --- Price series ---------------------------------------------------
        if execution.price_in not in frame.columns:
            raise KeyError(f"price_in '{execution.price_in}' not found in DataFrame.")
        if execution.price_out not in frame.columns:
            raise KeyError(f"price_out '{execution.price_out}' not found in DataFrame.")

        px_in = pd.to_numeric(frame[execution.price_in], errors="coerce")
        px_out = pd.to_numeric(frame[execution.price_out], errors="coerce")

        entries_idx = frame.index[direction != 0]
        trades: List[Dict[str, Any]] = []

        # --- Trade generation ----------------------------------------------
        for i in entries_idx:
            dir_i = int(np.sign(direction.loc[i]))
            if dir_i == 0:
                continue

            px_i = float(px_in.loc[i])
            if not np.isfinite(px_i):
                continue

            # Determine exit index j based on exit_policy
            if execution.exit_policy == "bars":
                pos = frame.index.get_loc(i)
                jpos = min(pos + int(execution.holding_bars), len(frame.index) - 1)
                j = frame.index[jpos]

            elif execution.exit_policy == "session_close":
                # Auto-detect weekend patterns: if entry is Friday and weekend_aware is True,
                # use monday_close logic instead to handle weekend hedging patterns
                is_friday = i.weekday() == 4
                if execution.weekend_aware and is_friday:
                    # Switch to monday_close logic for Friday entries
                    search = frame.loc[i + pd.Timedelta(seconds=1) :]
                    if search.empty:
                        continue
                    monday_days = pd.Index(
                        d
                        for d in pd.to_datetime(search.index.normalize().unique())
                        if d.weekday() == 0
                    )
                    if monday_days.empty:
                        continue
                    monday = monday_days[0]
                    mon = frame.loc[
                        monday : monday + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                    ]
                    if mon.empty:
                        continue
                    j = mon.index[-1]
                else:
                    # Normal session_close behavior
                    day = i.normalize()
                    same_day = frame.loc[
                        day : day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                    ]
                    j = same_day.index[-1]

            elif execution.exit_policy == "next_day_close":
                next_day = i.normalize() + pd.Timedelta(days=1)
                nxt = frame.loc[
                    next_day : next_day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                ]
                if nxt.empty:
                    continue
                j = nxt.index[-1]

            elif execution.exit_policy == "monday_close":
                # Find next Monday close
                search = frame.loc[i + pd.Timedelta(seconds=1) :]
                if search.empty:
                    continue
                monday_days = pd.Index(
                    d
                    for d in pd.to_datetime(search.index.normalize().unique())
                    if d.weekday() == 0
                )
                if monday_days.empty:
                    continue
                monday = monday_days[0]
                mon = frame.loc[
                    monday : monday + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                ]
                if mon.empty:
                    continue
                j = mon.index[-1]

            elif execution.exit_policy == "target_pct":
                # Exit when per-trade return hits target_pct in the favourable direction.
                if execution.target_pct <= 0.0:
                    raise ValueError(
                        "ExecutionPolicy.target_pct must be > 0 for 'target_pct' exit_policy."
                    )
                search = frame.loc[i + pd.Timedelta(seconds=1) :]
                if search.empty:
                    continue

                out_series = px_out.loc[search.index]

                if dir_i > 0:
                    rets = out_series / px_i - 1.0
                    hit_mask = rets >= execution.target_pct
                else:
                    rets = px_i / out_series - 1.0
                    hit_mask = rets >= execution.target_pct

                hit_idx = rets.index[hit_mask]
                if len(hit_idx) == 0:
                    # No hit: optional behaviour; here we hold to last available bar
                    j = search.index[-1]
                else:
                    j = hit_idx[0]

            else:
                raise ValueError(f"Unknown exit_policy: {execution.exit_policy}")

            px_j = float(px_out.loc[j])
            if not np.isfinite(px_j):
                continue

            if dir_i > 0:
                ret = px_j / px_i - 1.0
            else:
                ret = px_i / px_j - 1.0

            trades.append(
                {
                    "ts_in": i,
                    "ts_out": j,
                    "dir": dir_i,
                    "px_in": px_i,
                    "px_out": px_j,
                    "ret": ret,
                }
            )

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            empty = pd.Series(dtype=float)
            return {
                "pnl": empty,
                "positions": empty,
                "cumulative": empty,
                "trades": trades_df,
                "summary": BacktestSummary(0.0, 0.0, np.nan, np.nan, 0.0, 0),
            }

        # --- P&L aggregation ------------------------------------------------
        pnl = pd.Series(0.0, index=frame.index)
        positions = pd.Series(0.0, index=frame.index)
        for _, row in trades_df.iterrows():
            pnl.loc[row["ts_out"]] += row["ret"]      # book PnL at exit
            positions.loc[row["ts_in"]] += row["dir"] # entry impulse

        cumulative = pnl.cumsum()
        mean_return = float(pnl.mean()) if not pnl.empty else 0.0
        total_return = float(pnl.sum())
        hit_rate = float((pnl > 0).mean()) if not pnl.empty else np.nan
        std_return = float(pnl.std(ddof=0))
        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(self.annualisation)
        else:
            sharpe = np.nan

        drawdown = cumulative - cumulative.cummax()
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        summary = BacktestSummary(
            total_return=total_return,
            mean_return=mean_return,
            hit_rate=hit_rate,
            sharpe=float(sharpe) if np.isfinite(sharpe) else np.nan,
            max_drawdown=max_drawdown,
            trades=len(trades_df),
        )

        return {
            "pnl": pnl,
            "positions": positions,
            "cumulative": cumulative,
            "trades": trades_df,
            "summary": summary,
        }


def diagnose_alignment(a: pd.Series, b: pd.Series) -> dict:
    def _info(s: pd.Series):
        idx = s.index
        tzs = str(getattr(idx, "tz", None))
        return {
            "first": str(idx.min()) if len(idx) else "NA",
            "last":  str(idx.max()) if len(idx) else "NA",
            "len":   int(len(idx)),
            "na":    int(s.isna().sum()),
            "tz":    tzs,
        }
    info_a = _info(a); info_b = _info(b)
    overlap_ts = len(a.index.intersection(b.index))
    return {"a": info_a, "b": info_b, "overlap_exact_timestamps": overlap_ts}

def _ensure_dt_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    return s.dropna().sort_index()

def _coerce_tz(s: pd.Series, tz: Optional[str]) -> pd.Series:
    s = _ensure_dt_index(s.copy())
    if tz is not None:
        if s.index.tz is None:
            s.index = s.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        else:
            s.index = s.index.tz_convert(tz)
        s.index = s.index.tz_localize(None)  # make naive for exact equality
    return s

def _align_on_date(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    a = _ensure_dt_index(a); b = _ensure_dt_index(b)
    # normalize to midnight (date) and keep the last value per day (close)
    a = a.groupby(a.index.normalize()).last()
    b = b.groupby(b.index.normalize()).last()
    common = a.index.intersection(b.index)
    return a.reindex(common), b.reindex(common)

def build_overlay_frame(
    cumulative: pd.Series,
    price: Optional[pd.Series] = None,
    *,
    scale: Literal["minmax","rebase100","zscore"] = "minmax",
    align: Literal["date","timestamp"] = "date",
    tz: Optional[str] = "America/Chicago",
    dropna: bool = True,
    require_overlap: bool = True,
) -> pd.DataFrame:
    """
    Returns:
      DataFrame with columns:
        - cum_pnl
        - price_scaled (if price is provided)
    Defaults:
      - align='date' (daily close alignment)
      - tz='America/Chicago' (both series normalized then made tz-naive)
    """
    if cumulative is None or len(cumulative) == 0:
        return pd.DataFrame(columns=["cum_pnl","price_scaled"])

    cum = pd.to_numeric(pd.Series(cumulative.squeeze()), errors="coerce").dropna()
    cum = _coerce_tz(cum, tz)

    if price is None or len(price) == 0:
        return pd.DataFrame({"cum_pnl": cum})

    px = pd.to_numeric(pd.Series(price.squeeze()), errors="coerce").dropna()
    px = _coerce_tz(px, tz)

    if align == "date":
        cum_al, px_al = _align_on_date(cum, px)
    else:
        idx = cum.index.intersection(px.index)
        cum_al = cum.reindex(idx)
        px_al  = px.reindex(idx)

    if dropna:
        m = cum_al.notna() & px_al.notna()
        cum_al, px_al = cum_al[m], px_al[m]

    if require_overlap and (len(cum_al) == 0 or len(px_al) == 0):
        diag = diagnose_alignment(cum, px)
        raise ValueError(
            "No overlap after alignment. "
            f"cum[{diag['a']['first']} → {diag['a']['last']} tz={diag['a']['tz']}], "
            f"price[{diag['b']['first']} → {diag['b']['last']} tz={diag['b']['tz']}]."
        )

    # scale price to the cum range
    lo, hi = float(cum_al.min()), float(cum_al.max())
    span = max(hi - lo, 1e-12)

    if scale == "minmax":
        plo, phi = float(px_al.min()), float(px_al.max())
        pspan = max(phi - plo, 1e-12)
        px_scaled = (px_al - plo) / pspan
    elif scale == "rebase100":
        base = px_al.iloc[0]
        px_scaled = (px_al / base)
        px_scaled = (px_scaled - px_scaled.min()) / max(px_scaled.max() - px_scaled.min(), 1e-12)
    elif scale == "zscore":
        mu, sd = float(px_al.mean()), float(px_al.std(ddof=0)) or 1.0
        px_scaled = (px_al - mu) / sd
        px_scaled = (px_scaled - px_scaled.min()) / max(px_scaled.max() - px_scaled.min(), 1e-12)
    else:
        raise ValueError(scale)

    return pd.DataFrame({"cum_pnl": cum_al, "price_scaled": lo + px_scaled * span})

def plot_backtest_results(
    cumulative: pd.Series,
    price: Optional[pd.Series] = None,
    *,
    scale: Literal["minmax", "rebase100", "zscore"] = "minmax",
    ax=None,
):
    """
    Convenience wrapper. Returns the matplotlib Axes.
    (Leave styling to caller; this function only plots two aligned lines.)
    """
    frame = build_overlay_frame(cumulative, price, scale=scale)
    if frame.empty:
        return None
    if ax is None:
        fig, ax = plt.subplots()
    frame["cum_pnl"].plot(ax=ax, label="Cumulative PnL")
    if "price_scaled" in frame:
        frame["price_scaled"].plot(ax=ax, label="Price (scaled)")
    ax.legend()
    ax.set_title("Cumulative PnL with optional price overlay")
    return ax
