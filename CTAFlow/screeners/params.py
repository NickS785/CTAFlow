"""
Parameter dataclasses for pluggable screener engines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time, timedelta
from typing import List, Optional, Sequence, TYPE_CHECKING, Union

from .screener_types import (
    SCREEN_EVENT,
    SCREEN_MOMENTUM,
    SCREEN_ORDERFLOW,
    SCREEN_SEASONALITY,
)

if TYPE_CHECKING:
    from ..features.regime_classification import RegimeSpecificationLike


@dataclass
class BaseScreenParams:
    """
    Generic configuration for a historical screener run.

    Attributes
    ----------
    screen_type : str
        One of the values in :data:`CTAFlow.screeners.screener_types.VALID_SCREEN_TYPES`.
    name : str, optional
        Human readable name for the screen; a default is derived if omitted.
    tz : str
        Timezone identifier used to interpret sessions and event releases.
    use_regime_filtering : bool
        Whether to apply an optional regime classifier before running the screener.
    regime_col : str, optional
        Column name containing regime ids when supplying your own regimes.
    target_regimes : sequence of int, optional
        Regime ids that are retained when ``use_regime_filtering`` is enabled.
    regime_settings : RegimeSpecificationLike, optional
        Optional specification for building a regime classifier on the fly.
    horizons : sequence of str, optional
        Optional labels that can be propagated into downstream pattern summaries.
    """

    screen_type: str
    name: Optional[str] = None
    tz: str = "America/Chicago"

    use_regime_filtering: bool = False
    regime_col: Optional[str] = None
    target_regimes: Optional[Sequence[int]] = None
    regime_settings: Optional["RegimeSpecificationLike"] = None

    horizons: Optional[Sequence[str]] = None


@dataclass
class MomentumParams(BaseScreenParams):
    """Configuration for momentum screeners."""

    screen_type: str = SCREEN_MOMENTUM
    session_starts: List[str] = field(default_factory=list)
    session_ends: List[str] = field(default_factory=list)
    st_momentum_days: int = 3
    sess_start_hrs: Optional[int] = None
    sess_start_minutes: Optional[int] = None
    sess_end_hrs: Optional[int] = None
    sess_end_minutes: Optional[int] = None
    test_vol: bool = True
    months: Optional[List[int]] = None
    season: Optional[str] = None


@dataclass
class SeasonalityParams(BaseScreenParams):
    """Configuration for seasonality screeners."""

    screen_type: str = SCREEN_SEASONALITY
    season: Optional[str] = None
    months: Optional[List[int]] = None
    target_times: Optional[List[Union[str, time]]] = None
    period_length: Optional[Union[int, timedelta]] = None
    dayofweek_screen: bool = True
    seasonality_session_start: Optional[str] = None
    seasonality_session_end: Optional[str] = None
    include_calendar_effects: bool = True
    calendar_horizons: Optional[dict] = None
    calendar_min_obs: int = 50


@dataclass
class OrderflowParams(BaseScreenParams):
    """Configuration wrapper for orderflow screeners."""

    screen_type: str = SCREEN_ORDERFLOW
    session_start: str = "08:30"
    session_end: str = "15:00"
    tz: str = "America/Chicago"
    bucket_size: Union[int, str] = "auto"
    vpin_window: int = 50
    threshold_z: float = 2.0
    min_days: int = 30
    cadence_target: int = 50
    grid_multipliers: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5)
    month_filter: Optional[Sequence[int]] = None
    season_filter: Optional[Sequence[str]] = None
    name: Optional[str] = None
    use_gpu: bool = False
    gpu_device_id: int = 0


@dataclass
class EventParams(BaseScreenParams):
    """
    Configuration for event (data release) screeners.

    This is a light wrapper; the heavy lifting lives in :mod:`CTAFlow.screeners.event_screener`.
    """

    screen_type: str = SCREEN_EVENT
    event_code: str = ""
    event_window_pre_minutes: int = 5
    event_window_post_minutes: int = 15
    target_close_hhmm: str = "14:30"

    include_t1_close: bool = True
    extra_daily_horizons: List[int] = field(default_factory=list)

    value_col: Optional[str] = "actual"
    consensus_col: Optional[str] = "consensus"
    surprise_mode: str = "diff"

    min_events: int = 40
    fdr_alpha: float = 0.05
    corr_threshold: float = 0.25
    z_threshold: float = 2.0

    # Optional orderflow/tick integration
    use_orderflow: bool = False
    use_tick_data: bool = False
    orderflow_window_pre_minutes: int = 5
    orderflow_window_post_minutes: int = 15
    orderflow_bucket_volume: int = 100
