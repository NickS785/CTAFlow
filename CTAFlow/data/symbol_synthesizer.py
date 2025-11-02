"""Utility for building predefined synthetic intraday instruments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from ..config import INTRADAY_DATA_PATH
from .raw_formatting.synthetic import IntradayLeg, SyntheticSymbol

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_RESAMPLE = "1S"
SUMMARIZABLE_COLUMNS = {
    "Volume",
    "TotalVolume",
    "BidVolume",
    "AskVolume",
    "NumberOfTrades",
}


@dataclass(frozen=True)
class SyntheticComponent:
    """Single leg in a synthetic symbol definition."""

    symbol: str
    weight: float = 1.0


@dataclass(frozen=True)
class SyntheticSymbolDefinition:
    """Description for a synthetic instrument."""

    name: str
    components: Tuple[SyntheticComponent, ...]
    description: str
    operation: str = "spread"  # Allowed values: 'spread', 'ratio'
    metadata: Optional[Dict[str, str]] = None


SYNTHETIC_SYMBOLS: Dict[str, SyntheticSymbolDefinition] = {
    "CRACK_321": SyntheticSymbolDefinition(
        name="CRACK_321",
        description="3-2-1 crack spread (2*RBOB + 1*Heating Oil - 3*WTI Crude)",
        components=(
            SyntheticComponent("RB_F", weight=2.0),
            SyntheticComponent("HO_F", weight=1.0),
            SyntheticComponent("CL_F", weight=-3.0),
        ),
        operation="spread",
    ),
    "SOYBEAN_CRUSH": SyntheticSymbolDefinition(
        name="SOYBEAN_CRUSH",
        description="Soybean crush (11*Soybean Meal + 9*Soybean Oil - 10*Soybeans)",
        components=(
            SyntheticComponent("ZM_F", weight=11.0),
            SyntheticComponent("ZL_F", weight=9.0),
            SyntheticComponent("ZS_F", weight=-10.0),
        ),
        operation="spread",
    ),
    "GOLD_SILVER_RATIO": SyntheticSymbolDefinition(
        name="GOLD_SILVER_RATIO",
        description="Gold versus Silver price ratio",
        components=(
            SyntheticComponent("GC_F", weight=1.0),
            SyntheticComponent("SI_F", weight=1.0),
        ),
        operation="ratio",
    ),
    "LIVE_FEEDER_CATTLE_RATIO": SyntheticSymbolDefinition(
        name="LIVE_FEEDER_CATTLE_RATIO",
        description="Live cattle versus feeder cattle price ratio",
        components=(
            SyntheticComponent("LE_F", weight=1.0),
            SyntheticComponent("GF_F", weight=1.0),
        ),
        operation="ratio",
    ),
}


class SymbolSynthesizer:
    """Generate synthetic intraday symbols and persist them to Parquet."""

    def __init__(self,
                 base_path: Optional[Path] = None,
                 compression: str = "snappy",
                 logger: Optional[logging.Logger] = None) -> None:
        self.base_path = Path(base_path) if base_path else INTRADAY_DATA_PATH
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.logger = logger if logger is not None else LOGGER

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def available_symbols(self) -> List[str]:
        """Return the list of built-in synthetic symbol identifiers."""

        return sorted(SYNTHETIC_SYMBOLS.keys())

    def synthesize(
            self,
            symbol_name: str,
            output_rule: Optional[str] = "1T",
            align_index: str = "intersection",
            base_resample: str = DEFAULT_BASE_RESAMPLE,
            return_ohlc: bool = True,
    ) -> Dict[str, object]:
        """Create a synthetic symbol and store it as a Parquet file."""

        definition = self._resolve_definition(symbol_name)
        legs = [
            self._build_leg(component, base_resample=base_resample)
            for component in definition.components
        ]

        synthetic = SyntheticSymbol(
            legs=legs,
            ticker=definition.name,
            full_name=definition.description,
            intraday=True,
        )

        # Build close series to enforce alignment and validation.
        close_series = synthetic.data_engine.build_spread_series(
            align_index=align_index,
            return_ohlc=False,
        )

        if definition.operation == "ratio":
            result_df = self._build_ratio_dataframe(
                synthetic=synthetic,
                index=close_series.index,
                output_rule=output_rule,
            )
        else:
            result_df = self._build_spread_dataframe(
                synthetic=synthetic,
                align_index=align_index,
                output_rule=output_rule,
                return_ohlc=return_ohlc,
            )

        output_path = self._write_parquet(
            definition.name,
            output_rule=output_rule,
            df=result_df,
        )

        if self.logger:
            self.logger.info(
                "Synthesized %s (%s rows) -> %s",
                definition.name,
                len(result_df),
                output_path,
            )

        return {
            "success": True,
            "symbol": definition.name,
            "file_path": str(output_path),
            "records_written": int(len(result_df)),
            "output_rule": output_rule,
            "operation": definition.operation,
            "components": [
                {"symbol": leg.symbol, "weight": leg.base_weight}
                for leg in synthetic.legs
            ],
        }

    def synthesize_all(
            self,
            output_rule: Optional[str] = "1T",
            align_index: str = "intersection",
            base_resample: str = DEFAULT_BASE_RESAMPLE,
            return_ohlc: bool = True,
    ) -> Dict[str, Dict[str, object]]:
        """Generate Parquet files for all built-in synthetic symbols."""

        results: Dict[str, Dict[str, object]] = {}
        for symbol_name in self.available_symbols():
            try:
                results[symbol_name] = self.synthesize(
                    symbol_name,
                    output_rule=output_rule,
                    align_index=align_index,
                    base_resample=base_resample,
                    return_ohlc=return_ohlc,
                )
            except Exception as exc:  # pragma: no cover - depends on filesystem state
                if self.logger:
                    self.logger.error("Failed to synthesize %s: %s", symbol_name, exc)
                results[symbol_name] = {
                    "symbol": symbol_name,
                    "success": False,
                    "error": str(exc),
                }

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_definition(self, symbol_name: str) -> SyntheticSymbolDefinition:
        key = symbol_name.upper()
        if key not in SYNTHETIC_SYMBOLS:
            raise KeyError(f"Unknown synthetic symbol: {symbol_name}")
        return SYNTHETIC_SYMBOLS[key]

    def _build_leg(
            self,
            component: SyntheticComponent,
            base_resample: str = DEFAULT_BASE_RESAMPLE,
    ) -> IntradayLeg:
        df = self._load_component_frame(component.symbol)
        df = self._normalize_intraday_frame(df, base_resample=base_resample)
        return IntradayLeg(
            symbol=component.symbol,
            data=df,
            base_weight=component.weight,
        )

    def _load_component_frame(self, symbol: str) -> pd.DataFrame:
        symbol_dir = self.base_path / symbol
        candidates = [
            symbol_dir / f"{symbol}_raw.parquet",
            symbol_dir / f"{symbol}.parquet",
        ]

        if symbol_dir.exists():
            for path in sorted(symbol_dir.glob(f"{symbol}_*.parquet")):
                if path not in candidates:
                    candidates.append(path)

        for path in candidates:
            if path.exists():
                df = pd.read_parquet(path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError(f"Parquet {path} missing DatetimeIndex")
                return df.sort_index()

        raise FileNotFoundError(
            f"Unable to locate intraday parquet for {symbol} in {symbol_dir}"
        )

    def _normalize_intraday_frame(
            self,
            df: pd.DataFrame,
            base_resample: str = DEFAULT_BASE_RESAMPLE,
    ) -> pd.DataFrame:
        if df.index.tz is not None:
            df = df.tz_convert(None)

        df = df.sort_index()

        if self._is_tick_data(df.index):
            df = self._resample_ohlcv(df, rule=base_resample)

        return df

    @staticmethod
    def _is_tick_data(index: pd.DatetimeIndex) -> bool:
        if len(index) < 3:
            return False

        diffs = index.to_series().diff().dropna()
        if diffs.empty:
            return False

        value_counts = diffs.value_counts(normalize=True)
        most_common = value_counts.iloc[0]
        # Treat as tick data when there isn't a dominant spacing between rows.
        return most_common < 0.8

    def _build_spread_dataframe(
            self,
            synthetic: SyntheticSymbol,
            align_index: str,
            output_rule: Optional[str],
            return_ohlc: bool,
    ) -> pd.DataFrame:
        try:
            spread = synthetic.data_engine.build_spread_series(
                align_index=align_index,
                return_ohlc=return_ohlc,
            )
        except ValueError:
            spread = synthetic.data_engine.build_spread_series(
                align_index=align_index,
                return_ohlc=False,
            )
            return_ohlc = False

        if isinstance(spread, pd.Series):
            df = spread.to_frame("Close")
        else:
            df = spread

        if output_rule:
            df = self._resample_output(df, rule=output_rule)

        return df.dropna(how="all")

    def _build_ratio_dataframe(
            self,
            synthetic: SyntheticSymbol,
            index: pd.Index,
            output_rule: Optional[str],
    ) -> pd.DataFrame:
        if len(synthetic.legs) < 2:
            raise ValueError("Ratio synthetic requires at least two legs")

        numerator = synthetic.legs[0].data["Close"].reindex(index)
        denominator = synthetic.legs[1].data["Close"].reindex(index)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_series = numerator / denominator

        ratio_series = ratio_series.replace([np.inf, -np.inf], np.nan).dropna()

        if not output_rule:
            return ratio_series.to_frame("Close")

        ohlc = ratio_series.resample(output_rule).ohlc()
        ohlc.columns = [col.capitalize() for col in ohlc.columns]
        return ohlc.dropna(how="all")

    def _resample_output(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Synthetic output must have a DatetimeIndex for resampling")

        df = df.sort_index()

        if {"Open", "High", "Low", "Close"}.issubset(df.columns):
            price = df[["Open", "High", "Low", "Close"]].resample(rule).agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
            })
        elif "Close" in df.columns and df.shape[1] == 1:
            price = df["Close"].resample(rule).ohlc()
            price.columns = [col.capitalize() for col in price.columns]
        else:
            price = df.resample(rule).last()

        extras: Dict[str, pd.Series] = {}
        for column in df.columns:
            if column in {"Open", "High", "Low", "Close"}:
                continue
            agg = "sum" if column in SUMMARIZABLE_COLUMNS else "last"
            extras[column] = getattr(df[column].resample(rule), agg)()

        if extras:
            price = price.join(pd.DataFrame(extras))

        return price.dropna(how="all")

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        if {"Open", "High", "Low", "Close"}.issubset(df.columns):
            price = df[["Open", "High", "Low", "Close"]].resample(rule).agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
            })
        elif "Close" in df.columns:
            price = df["Close"].resample(rule).ohlc()
            price.columns = [col.capitalize() for col in price.columns]
        elif "Price" in df.columns:
            price = df["Price"].resample(rule).ohlc()
            price.columns = [col.capitalize() for col in price.columns]
        else:
            raise ValueError("Unable to derive OHLC data from intraday frame")

        if "Volume" in df.columns:
            price["Volume"] = df["Volume"].resample(rule).sum()

        return price.dropna(how="all")

    def _write_parquet(
            self,
            symbol: str,
            output_rule: Optional[str],
            df: pd.DataFrame,
    ) -> Path:
        output_dir = self.base_path / symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = output_rule if output_rule else "raw"
        file_path = output_dir / f"{symbol}_{suffix}.parquet"
        df.to_parquet(file_path, compression=self.compression)
        return file_path


def synthesize_symbol(
        symbol_name: str,
        output_rule: Optional[str] = "1T",
        align_index: str = "intersection",
        base_resample: str = DEFAULT_BASE_RESAMPLE,
        return_ohlc: bool = True,
        base_path: Optional[Path] = None,
        compression: str = "snappy",
) -> Dict[str, object]:
    """Convenience function mirroring :meth:`SymbolSynthesizer.synthesize`."""

    synthesizer = SymbolSynthesizer(base_path=base_path, compression=compression)
    return synthesizer.synthesize(
        symbol_name,
        output_rule=output_rule,
        align_index=align_index,
        base_resample=base_resample,
        return_ohlc=return_ohlc,
    )


def synthesize_all_symbols(
        output_rule: Optional[str] = "1T",
        align_index: str = "intersection",
        base_resample: str = DEFAULT_BASE_RESAMPLE,
        return_ohlc: bool = True,
        base_path: Optional[Path] = None,
        compression: str = "snappy",
) -> Dict[str, Dict[str, object]]:
    """Convenience helper to generate every built-in synthetic symbol."""

    synthesizer = SymbolSynthesizer(base_path=base_path, compression=compression)
    return synthesizer.synthesize_all(
        output_rule=output_rule,
        align_index=align_index,
        base_resample=base_resample,
        return_ohlc=return_ohlc,
    )


__all__ = [
    "SyntheticComponent",
    "SyntheticSymbolDefinition",
    "SYNTHETIC_SYMBOLS",
    "SymbolSynthesizer",
    "synthesize_symbol",
    "synthesize_all_symbols",
]
