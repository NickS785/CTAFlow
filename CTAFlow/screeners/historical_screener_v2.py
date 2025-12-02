"""
Pluggable historical screener orchestrator.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Sequence, Optional, Union, List, Tuple
import logging

import pandas as pd

from .screener_types import (
    SCREEN_EVENT,
    SCREEN_MOMENTUM,
    SCREEN_ORDERFLOW,
    SCREEN_SEASONALITY,
)
from .params import BaseScreenParams
from .base_engine import BaseScreenEngine
from .momentum_engine import MomentumScreenEngine
from .seasonality_engine import SeasonalityScreenEngine
from .orderflow_engine import OrderflowScreenEngine
from .event_engine import EventScreenEngine
from ..features.regime_classification import BaseRegimeClassifier
from ..data import ResultsClient
from ..strategy.gpu_acceleration import GPU_AVAILABLE, GPU_DEVICE_COUNT


class HistoricalScreenerV2:
    """
    Pluggable historical screener for multiple screen types.
    """

    def __init__(
        self,
        ticker_data: Dict[str, pd.DataFrame],
        results_client: Optional[ResultsClient] = None,
        auto_write_results: bool = False,
        engines: Optional[Dict[str, BaseScreenEngine]] = None,
        regime_classifier: Optional[BaseRegimeClassifier] = None,
        logger: Optional[logging.Logger] = None,
        use_gpu: bool = True,
        gpu_device_id: int = 0,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Initialize HistoricalScreenerV2.

        Parameters
        ----------
        ticker_data : Dict[str, pd.DataFrame]
            Dictionary mapping ticker symbols to DataFrames
        results_client : Optional[ResultsClient]
            Client for writing results
        auto_write_results : bool
            Automatically write results to results_client
        engines : Optional[Dict[str, BaseScreenEngine]]
            Custom screen engines (uses defaults if None)
        regime_classifier : Optional[BaseRegimeClassifier]
            Regime classifier for filtering
        logger : Optional[logging.Logger]
            Custom logger instance
        use_gpu : bool
            Enable GPU acceleration for underlying computations (default: True)
        gpu_device_id : int
            GPU device ID when multiple GPUs available (default: 0)
        max_workers : Optional[int]
            Number of parallel workers for ticker processing.
            If None, auto-determines based on ticker count (max 8).
            Set to 1 for sequential processing.
        """
        self.ticker_data = ticker_data
        self.results_client = results_client
        self.auto_write_results = auto_write_results
        self.regime_classifier = regime_classifier
        self.logger = logger or logging.getLogger(__name__)
        self.use_gpu = bool(use_gpu and GPU_AVAILABLE)
        self.gpu_device_id = gpu_device_id
        self.max_workers = max_workers

        if use_gpu and not self.use_gpu:
            self.logger.info("GPU requested but unavailable; defaulting to CPU execution")
        elif self.use_gpu:
            self.logger.info(
                "GPU acceleration enabled (device %s, %s GPUs detected)",
                self.gpu_device_id,
                GPU_DEVICE_COUNT,
            )

        self.engines: Dict[str, BaseScreenEngine] = engines or {
            SCREEN_MOMENTUM: MomentumScreenEngine(),
            SCREEN_SEASONALITY: SeasonalityScreenEngine(),
            SCREEN_ORDERFLOW: OrderflowScreenEngine(),
            SCREEN_EVENT: EventScreenEngine(),
        }

    def run_screens(
        self,
        screen_params: Sequence[BaseScreenParams],
        output_format: str = "dict",
    ) -> Union[Dict[str, Dict[str, Any]], pd.DataFrame]:
        """Run one or more screens over the ticker universe.

        Processes tickers in parallel when max_workers > 1, significantly
        improving performance for large ticker sets.
        """

        results: Dict[str, Dict[str, Any]] = {}

        for params in screen_params:
            engine = self.engines.get(params.screen_type)
            if engine is None:
                raise ValueError(f"Unregistered screen_type: {params.screen_type}")

            screen_name = params.name or self._default_screen_name(params)
            self.logger.info("Running screen %s (%s)", screen_name, params.screen_type)

            # Use parallel execution for ticker processing
            results[screen_name] = self._run_screen_parallel(
                screen_name=screen_name,
                engine=engine,
                params=params,
            )

        if output_format == "dict":
            return results
        return self._flatten_results(results)

    def _resolve_workers(self) -> int:
        if self.max_workers is not None and self.max_workers > 0:
            return int(self.max_workers)

        if self.use_gpu and GPU_AVAILABLE:
            gpu_workers = GPU_DEVICE_COUNT or 1
            return max(1, min(len(self.ticker_data), gpu_workers))

        n_tickers = len(self.ticker_data)
        return min(max(1, n_tickers), 8)

    def _run_screen_parallel(
        self,
        screen_name: str,
        engine: BaseScreenEngine,
        params: BaseScreenParams,
    ) -> Dict[str, Any]:
        """Run a single screen across all tickers, optionally in parallel.

        Parameters
        ----------
        screen_name : str
            Name of the screen being run
        engine : BaseScreenEngine
            Engine to use for processing
        params : BaseScreenParams
            Screen parameters

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping ticker symbols to screen results
        """

        def _process_ticker(ticker: str, df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
            """Process single ticker (runs in thread pool)."""
            # Prepare context
            ctx = engine.prepare(
                ticker=ticker,
                data=df,
                params=params,
                regime_classifier=self.regime_classifier,
            )

            # Add GPU settings to context if GPU is enabled
            if self.use_gpu:
                ctx["use_gpu"] = self.use_gpu
                ctx["gpu_device_id"] = self.gpu_device_id

            # Run screen
            res = engine.run(
                ticker=ticker,
                data=df,
                params=params,
                context=ctx,
            )

            # Write results if auto-writing enabled
            if self.auto_write_results and self.results_client:
                self._write_results(screen_name, ticker, params, res)

            return ticker, res

        # Determine number of workers
        max_workers = self._resolve_workers()

        # Parallel execution if multiple workers and multiple tickers
        if max_workers > 1 and len(self.ticker_data) > 1:
            screen_results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_process_ticker, ticker, df): ticker
                    for ticker, df in self.ticker_data.items()
                }

                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        tick, res = future.result()
                        screen_results[tick] = res
                    except Exception as exc:
                        self.logger.error("Screen failed for ticker %s: %s", ticker, exc)
                        screen_results[ticker] = {"error": str(exc)}

            return screen_results

        else:
            # Sequential fallback for single ticker or max_workers=1
            screen_results = {}
            for ticker, df in self.ticker_data.items():
                try:
                    tick, res = _process_ticker(ticker, df)
                    screen_results[tick] = res
                except Exception as exc:
                    self.logger.error("Screen failed for ticker %s: %s", ticker, exc)
                    screen_results[ticker] = {"error": str(exc)}

            return screen_results

    def _default_screen_name(self, params: BaseScreenParams) -> str:
        if getattr(params, "season", None):
            return f"{params.season}_{params.screen_type}"
        if getattr(params, "months", None):
            month_str = "_".join(map(str, getattr(params, "months")))
            return f"months_{month_str}_{params.screen_type}"
        return f"all_{params.screen_type}"

    def _write_results(
        self,
        screen_name: str,
        ticker: str,
        params: BaseScreenParams,
        res: Dict[str, Any],
    ) -> None:
        if not self.results_client:
            return
        stats = res.get("stats")
        if stats is None:
            return
        if isinstance(stats, dict):
            stats_df = pd.DataFrame([stats])
        elif isinstance(stats, pd.DataFrame):
            stats_df = stats
        else:
            stats_df = pd.DataFrame(stats)

        metadata = res.get("metadata") or {}
        metadata.update({"screen_type": params.screen_type, "screen_name": screen_name})
        self.results_client.write_scan_results(params.screen_type, ticker, screen_name, stats_df, metadata=metadata)

    def _flatten_results(self, nested: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for screen_name, per_ticker in nested.items():
            for ticker, payload in per_ticker.items():
                stats = payload.get("stats") if isinstance(payload, dict) else payload
                if isinstance(stats, pd.DataFrame):
                    df = stats.copy()
                    df["screen_name"] = screen_name
                    df["ticker"] = ticker
                    rows.append(df)
                elif isinstance(stats, dict):
                    row = dict(stats)
                    row.update({"screen_name": screen_name, "ticker": ticker})
                    rows.append(pd.DataFrame([row]))
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)
