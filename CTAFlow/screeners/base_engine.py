"""
Abstract base class for pluggable screener engines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from .params import BaseScreenParams
from ..features.regime_classification import BaseRegimeClassifier


class BaseScreenEngine(ABC):
    """
    Base interface for per-screen-type engines.

    Implementations can use :meth:`prepare` for per-ticker precomputation and
    :meth:`run` for the main screening logic.
    """

    screen_type: str

    @abstractmethod
    def prepare(
        self,
        ticker: str,
        data: pd.DataFrame,
        params: BaseScreenParams,
        regime_classifier: Optional[BaseRegimeClassifier] = None,
    ) -> Dict[str, Any]:
        """Optional precomputation hook executed once per ticker."""

    @abstractmethod
    def run(
        self,
        ticker: str,
        data: pd.DataFrame,
        params: BaseScreenParams,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the screener for a single ticker and return structured results."""
