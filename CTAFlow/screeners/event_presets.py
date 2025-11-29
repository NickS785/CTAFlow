"""
Hard-wired event definitions and per-ticker mappings.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time, date, datetime
from typing import Dict, List, Optional, Union

from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class EventDefinition:
    """Static metadata for a recurring event type."""

    code: str
    description: str
    local_tz: str
    release_time_local: time
    weekday: Optional[int] = None
    week_of_month: Optional[int] = None


EIA_NG_STORAGE = EventDefinition(
    code="EIA_NG_STORAGE",
    description="Weekly EIA Natural Gas Storage Report",
    local_tz="America/New_York",
    release_time_local=time(10, 30),
    weekday=3,
    week_of_month=None,
)

EIA_CRUDE_INVENTORY = EventDefinition(
    code="EIA_CRUDE_INV",
    description="Weekly EIA Crude Oil & Petroleum Status Report",
    local_tz="America/New_York",
    release_time_local=time(10, 30),
    weekday=2,
    week_of_month=None,
)

API_CRUDE_INVENTORY = EventDefinition(
    code="API_CRUDE_INV",
    description="Weekly API Crude Oil Inventory Report",
    local_tz="America/New_York",
    release_time_local=time(16, 30),
    weekday=1,
    week_of_month=None,
)

BAKER_HUGHES_RIGS = EventDefinition(
    code="BAKER_RIGS",
    description="Weekly Baker Hughes North America Rig Count",
    local_tz="America/Chicago",
    release_time_local=time(12, 0),
    weekday=4,
    week_of_month=None,
)

USDA_WASDE = EventDefinition(
    code="USDA_WASDE",
    description="USDA World Agricultural Supply and Demand Estimates",
    local_tz="America/New_York",
    release_time_local=time(12, 0),
    weekday=None,
    week_of_month=2,
)

USDA_GRAIN_STOCKS = EventDefinition(
    code="USDA_GRAIN_STOCKS",
    description="USDA Quarterly Grain Stocks Report",
    local_tz="America/New_York",
    release_time_local=time(12, 0),
    weekday=None,
    week_of_month=None,
)

USDA_PROSPECTIVE_PLANTINGS = EventDefinition(
    code="USDA_PROSPECTIVE_PLANTINGS",
    description="USDA Prospective Plantings Report",
    local_tz="America/New_York",
    release_time_local=time(12, 0),
    weekday=None,
    week_of_month=None,
)

USDA_ACREAGE = EventDefinition(
    code="USDA_ACREAGE",
    description="USDA Acreage Report",
    local_tz="America/New_York",
    release_time_local=time(12, 0),
    weekday=None,
    week_of_month=None,
)

US_CPI = EventDefinition(
    code="US_CPI",
    description="US CPI (Consumer Price Index) Release",
    local_tz="America/New_York",
    release_time_local=time(8, 30),
    weekday=None,
    week_of_month=None,
)

US_NFP = EventDefinition(
    code="US_NFP",
    description="US Nonfarm Payrolls / Employment Situation",
    local_tz="America/New_York",
    release_time_local=time(8, 30),
    weekday=4,
    week_of_month=1,
)

US_GDP_ADVANCE = EventDefinition(
    code="US_GDP_ADV",
    description="US GDP Advance Release",
    local_tz="America/New_York",
    release_time_local=time(8, 30),
    weekday=None,
    week_of_month=None,
)

FOMC_RATE_DECISION = EventDefinition(
    code="FOMC_RATE",
    description="FOMC Policy Statement and Rate Decision",
    local_tz="America/New_York",
    release_time_local=time(14, 0),
    weekday=None,
    week_of_month=None,
)

ALL_EVENT_DEFS: Dict[str, EventDefinition] = {
    e.code: e
    for e in [
        EIA_NG_STORAGE,
        EIA_CRUDE_INVENTORY,
        API_CRUDE_INVENTORY,
        BAKER_HUGHES_RIGS,
        USDA_WASDE,
        USDA_GRAIN_STOCKS,
        USDA_PROSPECTIVE_PLANTINGS,
        USDA_ACREAGE,
        US_CPI,
        US_NFP,
        US_GDP_ADVANCE,
        FOMC_RATE_DECISION,
    ]
}

TICKER_EVENT_MAP: Dict[str, List[str]] = {
    "NG": ["EIA_NG_STORAGE", "BAKER_RIGS"],
    "NG_F": ["EIA_NG_STORAGE", "BAKER_RIGS"],
    "CL": ["EIA_CRUDE_INV", "API_CRUDE_INV", "BAKER_RIGS"],
    "CL_F": ["EIA_CRUDE_INV", "API_CRUDE_INV", "BAKER_RIGS"],
    "RB": ["EIA_CRUDE_INV"],
    "HO": ["EIA_CRUDE_INV"],
    "ZC": ["USDA_WASDE", "USDA_GRAIN_STOCKS", "USDA_PROSPECTIVE_PLANTINGS", "USDA_ACREAGE"],
    "ZC_F": ["USDA_WASDE", "USDA_GRAIN_STOCKS", "USDA_PROSPECTIVE_PLANTINGS", "USDA_ACREAGE"],
    "ZS": ["USDA_WASDE", "USDA_GRAIN_STOCKS"],
    "ZW": ["USDA_WASDE", "USDA_GRAIN_STOCKS"],
    "GC": ["US_CPI", "US_NFP", "US_GDP_ADV", "FOMC_RATE"],
    "SI": ["US_CPI", "US_NFP", "US_GDP_ADV", "FOMC_RATE"],
}

DEFAULT_EVENTS_ENERGY: List[str] = ["EIA_CRUDE_INV", "API_CRUDE_INV", "BAKER_RIGS"]
DEFAULT_EVENTS_GRAINS: List[str] = ["USDA_WASDE", "USDA_GRAIN_STOCKS"]
DEFAULT_EVENTS_PRECIOUS: List[str] = ["US_CPI", "US_NFP", "FOMC_RATE"]


def get_events_for_ticker(ticker: str, *, asset_class: Optional[str] = None) -> List[EventDefinition]:
    if ticker in TICKER_EVENT_MAP:
        codes = TICKER_EVENT_MAP[ticker]
    else:
        if asset_class == "energy":
            codes = DEFAULT_EVENTS_ENERGY
        elif asset_class == "grains":
            codes = DEFAULT_EVENTS_GRAINS
        elif asset_class == "precious":
            codes = DEFAULT_EVENTS_PRECIOUS
        else:
            return []
    return [ALL_EVENT_DEFS[c] for c in codes if c in ALL_EVENT_DEFS]


def event_release_dt_for_date(
    event_def: EventDefinition,
    session_date: Union[date, datetime],
    instrument_tz: str,
) -> datetime:
    if isinstance(session_date, datetime):
        d = session_date.date()
    else:
        d = session_date
    local_zone = ZoneInfo(event_def.local_tz)
    inst_zone = ZoneInfo(instrument_tz)
    dt_local = datetime(
        year=d.year,
        month=d.month,
        day=d.day,
        hour=event_def.release_time_local.hour,
        minute=event_def.release_time_local.minute,
        tzinfo=local_zone,
    )
    return dt_local.astimezone(inst_zone)


def week_of_month_index(d: date) -> int:
    return (d.day - 1) // 7 + 1


def is_matching_event_slot(d: date, event_def: EventDefinition) -> bool:
    if event_def.weekday is not None and d.weekday() != event_def.weekday:
        return False
    if event_def.week_of_month is None:
        return True
    return week_of_month_index(d) == event_def.week_of_month
