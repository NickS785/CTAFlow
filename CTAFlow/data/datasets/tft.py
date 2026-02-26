"""
TFT-Aligned Dataset & Data Utilities
=====================================

Data pipeline for TFTAlignedWSPR / TFTAlignedMamba models.

Follows the TFT input taxonomy:
  - Static Covariates -> ticker_id, asset_class_id, asset_subclass_id
  - Known Future      -> calendar (month, dow, doy) + event schedule
  - Past-Observed     -> market branches + macro + event outcomes
  - Target            -> classification label or regression value

Event Routing
-------------
Events are routed by ticker type. Each ticker type receives:
  1. UNIVERSAL events (affect ALL tickers): FOMC, CPI, PPI, NFP, etc.
  2. SECTOR-SPECIFIC events: EIA/OPEC for energy, USDA for ag, LME for metals
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from CTAFlow.models.deep_learning.multi_branch.macro_event_encoding import (
    COMMODITY_EVENT_TYPES,
    N_EVENT_TYPES,
)


# ============================================================================
# Event Routing by Ticker Type
# ============================================================================

UNIVERSAL_EVENTS: Set[str] = {
    "fomc_decision", "fomc_minutes", "cpi", "ppi", "nfp",
    "ism_manufacturing", "ism_services", "gdp", "pce", "retail_sales",
    "china_pmi", "china_trade", "ecb_decision", "boj_decision",
    "geopolitical",
}

ENERGY_SPECIFIC_EVENTS: Set[str] = {
    "eia_petroleum", "eia_natgas", "api_weekly", "baker_hughes_rig",
    "opec_meeting", "opec_plus_meeting", "weather_event",
    "supply_disruption", "emergency_opec",
}

AG_SPECIFIC_EVENTS: Set[str] = {
    "usda_wasde", "usda_crop_report", "usda_export_sales", "weather_event",
}

METALS_SPECIFIC_EVENTS: Set[str] = {
    "lme_warehouse", "comex_delivery", "china_reserves",
}

EQUITY_SPECIFIC_EVENTS: Set[str] = set()

EVENTS_BY_TICKER_TYPE: Dict[str, Set[str]] = {
    "energy":       UNIVERSAL_EVENTS | ENERGY_SPECIFIC_EVENTS,
    "agriculture":  UNIVERSAL_EVENTS | AG_SPECIFIC_EVENTS,
    "metals":       UNIVERSAL_EVENTS | METALS_SPECIFIC_EVENTS,
    "equity":       UNIVERSAL_EVENTS | EQUITY_SPECIFIC_EVENTS,
    "rates":        UNIVERSAL_EVENTS,
    "fx":           UNIVERSAL_EVENTS,
    "softs":        UNIVERSAL_EVENTS | AG_SPECIFIC_EVENTS,
    "other":        UNIVERSAL_EVENTS,
}


# ============================================================================
# Ticker / Asset Class / Subclass Registries
# ============================================================================

@dataclass
class TickerMeta:
    """Metadata for a single ticker."""
    ticker: str
    ticker_id: int
    asset_class: str
    asset_class_id: int
    asset_subclass: str
    asset_subclass_id: int
    ticker_type: str

ASSET_CLASS_IDS: Dict[str, int] = {
    "commodity": 0, "equity": 1, "fx": 2, "rates": 3,
}

ASSET_SUBCLASS_IDS: Dict[str, int] = {
    "energy": 0, "agriculture": 1, "metals": 2, "softs": 3,
    "index": 4, "sector": 5, "single_stock": 6, "major": 7,
    "em": 8, "treasuries": 9, "rates_other": 10,
}

_DEFAULT_TICKER_TABLE: Dict[str, Dict[str, str]] = {
    "CL": {"asset_class": "commodity", "asset_subclass": "energy",      "ticker_type": "energy"},
    "NG": {"asset_class": "commodity", "asset_subclass": "energy",      "ticker_type": "energy"},
    "HO": {"asset_class": "commodity", "asset_subclass": "energy",      "ticker_type": "energy"},
    "RB": {"asset_class": "commodity", "asset_subclass": "energy",      "ticker_type": "energy"},
    "BZ": {"asset_class": "commodity", "asset_subclass": "energy",      "ticker_type": "energy"},
    "ZC": {"asset_class": "commodity", "asset_subclass": "agriculture", "ticker_type": "agriculture"},
    "ZS": {"asset_class": "commodity", "asset_subclass": "agriculture", "ticker_type": "agriculture"},
    "ZW": {"asset_class": "commodity", "asset_subclass": "agriculture", "ticker_type": "agriculture"},
    "ZM": {"asset_class": "commodity", "asset_subclass": "agriculture", "ticker_type": "agriculture"},
    "ZL": {"asset_class": "commodity", "asset_subclass": "agriculture", "ticker_type": "agriculture"},
    "CT": {"asset_class": "commodity", "asset_subclass": "softs",       "ticker_type": "softs"},
    "KC": {"asset_class": "commodity", "asset_subclass": "softs",       "ticker_type": "softs"},
    "SB": {"asset_class": "commodity", "asset_subclass": "softs",       "ticker_type": "softs"},
    "CC": {"asset_class": "commodity", "asset_subclass": "softs",       "ticker_type": "softs"},
    "OJ": {"asset_class": "commodity", "asset_subclass": "softs",       "ticker_type": "softs"},
    "GC": {"asset_class": "commodity", "asset_subclass": "metals",      "ticker_type": "metals"},
    "SI": {"asset_class": "commodity", "asset_subclass": "metals",      "ticker_type": "metals"},
    "HG": {"asset_class": "commodity", "asset_subclass": "metals",      "ticker_type": "metals"},
    "PA": {"asset_class": "commodity", "asset_subclass": "metals",      "ticker_type": "metals"},
    "PL": {"asset_class": "commodity", "asset_subclass": "metals",      "ticker_type": "metals"},
    "ES": {"asset_class": "equity",   "asset_subclass": "index",        "ticker_type": "equity"},
    "NQ": {"asset_class": "equity",   "asset_subclass": "index",        "ticker_type": "equity"},
    "YM": {"asset_class": "equity",   "asset_subclass": "index",        "ticker_type": "equity"},
    "RTY": {"asset_class": "equity",  "asset_subclass": "index",        "ticker_type": "equity"},
    "HE": {"asset_class": "commodity", "asset_subclass": "agriculture", "ticker_type": "agriculture"},
    "LE": {"asset_class": "commodity", "asset_subclass": "agriculture", "ticker_type": "agriculture"},
}


def build_ticker_registry(
    tickers: List[str],
    custom_table: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, TickerMeta]:
    """Build ticker -> TickerMeta registry from a list of tickers."""
    table = {**_DEFAULT_TICKER_TABLE}
    if custom_table:
        table.update(custom_table)

    registry: Dict[str, TickerMeta] = {}
    for tid, ticker in enumerate(sorted(tickers)):
        info = table.get(ticker, {})
        ac = info.get("asset_class", "commodity")
        asc = info.get("asset_subclass", "other")
        tt = info.get("ticker_type", "other")

        registry[ticker] = TickerMeta(
            ticker=ticker,
            ticker_id=tid,
            asset_class=ac,
            asset_class_id=ASSET_CLASS_IDS.get(ac, 0),
            asset_subclass=asc,
            asset_subclass_id=ASSET_SUBCLASS_IDS.get(asc, 0),
            ticker_type=tt,
        )
    return registry


# ============================================================================
# Calendar Feature Builder
# ============================================================================

def build_calendar_features(
    dates: Sequence[Union[date, datetime]],
) -> Dict[str, np.ndarray]:
    """Build calendar features for a window of dates.

    Returns dict with month (1-12), dow (0-6), doy_sin, doy_cos.
    """
    W = len(dates)
    month = np.zeros(W, dtype=np.int64)
    dow = np.zeros(W, dtype=np.int64)
    doy_sin = np.zeros(W, dtype=np.float32)
    doy_cos = np.zeros(W, dtype=np.float32)

    for i, d in enumerate(dates):
        if isinstance(d, datetime):
            d = d.date()
        month[i] = d.month
        dow[i] = d.weekday()
        doy = d.timetuple().tm_yday
        doy_sin[i] = math.sin(2 * math.pi * doy / 365.25)
        doy_cos[i] = math.cos(2 * math.pi * doy / 365.25)

    return {"month": month, "dow": dow, "doy_sin": doy_sin, "doy_cos": doy_cos}


# ============================================================================
# Event Window Builder
# ============================================================================

@dataclass
class ScheduledEvent:
    """A single event on a specific date."""
    date: date
    event_type: str
    surprise: float = 0.0
    direction: float = 0.0
    magnitude: float = 0.0
    revision: float = 0.0


class EventWindowBuilder:
    """Builds event tensors for a lookback window, filtered by ticker type.

    Parameters
    ----------
    event_registry : dict
        Full event type -> ID mapping.
    max_events_per_day : int
        Maximum events per day slot.
    n_outcome_features : int
        Continuous outcome features per event.
    surprise_threshold : float
        Minimum |surprise| to mark event_mask = 1.0.
    anticipation_horizon : int
        Days ahead to include upcoming scheduled events.
    """

    def __init__(
        self,
        event_registry: Optional[Dict[str, int]] = None,
        max_events_per_day: int = 3,
        n_outcome_features: int = 4,
        surprise_threshold: float = 1.0,
        anticipation_horizon: int = 5,
    ):
        self.registry = event_registry or COMMODITY_EVENT_TYPES
        self.max_events = max_events_per_day
        self.n_outcomes = n_outcome_features
        self.surprise_threshold = surprise_threshold
        self.anticipation_horizon = anticipation_horizon

    def build_window(
        self,
        window_dates: Sequence[date],
        events_by_date: Dict[date, List[ScheduledEvent]],
        allowed_events: Set[str],
        reference_date: Optional[date] = None,
    ) -> Dict[str, np.ndarray]:
        """Build event tensors for one sample's lookback window.

        Returns dict with event_type_ids, event_outcomes,
        days_until_event, event_mask.
        """
        W = len(window_dates)
        E = self.max_events
        if reference_date is None:
            reference_date = window_dates[-1]

        type_ids = np.zeros((W, E), dtype=np.int64)
        outcomes = np.zeros((W, E, self.n_outcomes), dtype=np.float32)
        days_until = np.full((W, E), 6, dtype=np.int64)
        mask = np.zeros(W, dtype=np.float32)

        for d_idx, d in enumerate(window_dates):
            if isinstance(d, datetime):
                d = d.date()

            day_events = events_by_date.get(d, [])
            filtered = [ev for ev in day_events if ev.event_type in allowed_events]

            for ahead in range(1, self.anticipation_horizon + 1):
                future_date = d + timedelta(days=ahead)
                for ev in events_by_date.get(future_date, []):
                    if ev.event_type in allowed_events:
                        filtered.append(ScheduledEvent(
                            date=future_date, event_type=ev.event_type,
                        ))

            # Deduplicate by event_type
            seen_types: Dict[str, ScheduledEvent] = {}
            for ev in filtered:
                key = ev.event_type
                if key not in seen_types:
                    seen_types[key] = ev
                elif ev.date == d and abs(ev.surprise) > 0:
                    seen_types[key] = ev

            deduped = sorted(
                seen_types.values(),
                key=lambda ev: (ev.date != d, ev.date),
            )

            for e_idx in range(min(len(deduped), E)):
                ev = deduped[e_idx]
                type_ids[d_idx, e_idx] = self.registry.get(ev.event_type, 0)
                delta = (ev.date - d).days if isinstance(ev.date, date) else 0
                days_until[d_idx, e_idx] = max(min(delta, 5), 0)

                if ev.date <= reference_date:
                    outcomes[d_idx, e_idx, 0] = ev.surprise
                    outcomes[d_idx, e_idx, 1] = ev.direction
                    outcomes[d_idx, e_idx, 2] = ev.magnitude
                    if self.n_outcomes > 3:
                        outcomes[d_idx, e_idx, 3] = ev.revision

                if ev.date <= reference_date and abs(ev.surprise) > self.surprise_threshold:
                    mask[d_idx] = 1.0

        return {
            "event_type_ids": type_ids,
            "event_outcomes": outcomes,
            "days_until_event": days_until,
            "event_mask": mask,
        }


# ============================================================================
# TFT Aligned Sample
# ============================================================================

@dataclass
class TFTAlignedSample:
    """A single prepared sample ready for the dataset."""
    summary_days: np.ndarray        # (W, f_sum)
    profile_days: np.ndarray        # (W, f_profile, bins)
    raster_recent: np.ndarray       # (T_bars, f_raster, bins)
    seq_recent: np.ndarray          # (T_seq, f_seq)
    seq_len: int
    macro_days: np.ndarray          # (W, f_macro)
    ticker_id: int
    asset_class_id: int
    asset_subclass_id: int
    month: np.ndarray               # (W,)
    dow: np.ndarray                 # (W,)
    doy_sin: np.ndarray             # (W,)
    doy_cos: np.ndarray             # (W,)
    event_type_ids: np.ndarray      # (W, max_events)
    event_outcomes: np.ndarray      # (W, max_events, n_outcomes)
    days_until_event: np.ndarray    # (W, max_events)
    event_mask: np.ndarray          # (W,)
    target: Union[int, float, np.ndarray]
    ae_input: Optional[np.ndarray] = None  # (W, f_ae) daily returns window for VAE
    ticker: str = ""
    prediction_date: Optional[date] = None


# ============================================================================
# PyTorch Dataset
# ============================================================================

class TFTAlignedDataset(Dataset):
    """PyTorch Dataset for TFTAligned models.

    Wraps a list of TFTAlignedSample objects and returns dicts of tensors.
    """

    def __init__(
        self,
        samples: List[TFTAlignedSample],
        return_metadata: bool = False,
    ):
        self.samples = samples
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        item = {
            "summary_days":      torch.from_numpy(s.summary_days).float(),
            "profile_days":      torch.from_numpy(s.profile_days).float(),
            "raster_recent":     torch.from_numpy(s.raster_recent).float(),
            "seq_recent":        torch.from_numpy(s.seq_recent).float(),
            "seq_lens_recent":   torch.tensor(s.seq_len, dtype=torch.long),
            "macro_days":        torch.from_numpy(s.macro_days).float(),
            "ticker_id":         torch.tensor(s.ticker_id, dtype=torch.long),
            "asset_class_id":    torch.tensor(s.asset_class_id, dtype=torch.long),
            "asset_subclass_id": torch.tensor(s.asset_subclass_id, dtype=torch.long),
            "month":             torch.from_numpy(s.month).long(),
            "dow":               torch.from_numpy(s.dow).long(),
            "doy_sin":           torch.from_numpy(s.doy_sin).float(),
            "doy_cos":           torch.from_numpy(s.doy_cos).float(),
            "event_type_ids":    torch.from_numpy(s.event_type_ids).long(),
            "event_outcomes":    torch.from_numpy(s.event_outcomes).float(),
            "days_until_event":  torch.from_numpy(s.days_until_event).long(),
            "event_mask":        torch.from_numpy(s.event_mask).float(),
            "target": (
                torch.tensor(s.target, dtype=torch.long)
                if isinstance(s.target, (int, np.integer))
                else torch.tensor(s.target, dtype=torch.float32)
            ),
        }

        if s.ae_input is not None:
            item["ae_input"] = torch.from_numpy(s.ae_input).float()

        if self.return_metadata:
            item["_ticker"] = s.ticker
            item["_prediction_date"] = str(s.prediction_date) if s.prediction_date else ""

        return item


# ============================================================================
# Collate Function
# ============================================================================

def tft_aligned_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Custom collate that handles variable-length sequential data.

    Pads raster_recent and seq_recent to the max length in the batch.
    Returns a dict that can be unpacked into model.forward(**batch)
    after removing 'target' and metadata keys.
    """
    fixed_keys = [
        "summary_days", "profile_days", "macro_days",
        "ticker_id", "asset_class_id", "asset_subclass_id",
        "month", "dow", "doy_sin", "doy_cos",
        "event_type_ids", "event_outcomes", "days_until_event", "event_mask",
        "target",
    ]
    collated: Dict[str, torch.Tensor] = {}
    for key in fixed_keys:
        collated[key] = torch.stack([b[key] for b in batch])

    # Stack ae_input if present (from daily returns VAE dataset)
    if "ae_input" in batch[0]:
        collated["ae_input"] = torch.stack([b["ae_input"] for b in batch])

    collated["seq_lens_recent"] = torch.stack(
        [b["seq_lens_recent"] for b in batch]
    )

    # Pad raster_recent to max T_bars
    rasters = [b["raster_recent"] for b in batch]
    max_t_bars = max(r.shape[0] for r in rasters)
    padded_rasters = []
    for r in rasters:
        T, C, H = r.shape
        if T < max_t_bars:
            pad = torch.zeros(max_t_bars - T, C, H, dtype=r.dtype)
            r = torch.cat([r, pad], dim=0)
        padded_rasters.append(r)
    collated["raster_recent"] = torch.stack(padded_rasters)

    # Pad seq_recent to max T_seq
    seqs = [b["seq_recent"] for b in batch]
    max_t_seq = max(s.shape[0] for s in seqs)
    padded_seqs = []
    for s in seqs:
        T, F = s.shape
        if T < max_t_seq:
            pad = torch.zeros(max_t_seq - T, F, dtype=s.dtype)
            s = torch.cat([s, pad], dim=0)
        padded_seqs.append(s)
    collated["seq_recent"] = torch.stack(padded_seqs)

    if "_ticker" in batch[0]:
        collated["_ticker"] = [b["_ticker"] for b in batch]
        collated["_prediction_date"] = [b["_prediction_date"] for b in batch]

    return collated


def unpack_batch_for_model(
    batch: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Split a collated batch into model inputs and targets.

    Returns (model_inputs, targets) where model_inputs can be passed
    directly to model.forward(**model_inputs).
    """
    MODEL_KEYS = [
        "summary_days", "profile_days", "raster_recent",
        "seq_recent", "seq_lens_recent", "macro_days",
        "ticker_id", "asset_class_id", "asset_subclass_id",
        "month", "dow", "doy_sin", "doy_cos",
        "event_type_ids", "event_outcomes", "days_until_event", "event_mask",
    ]
    # Include ae_input when present (daily returns for VAE conditioning)
    OPTIONAL_KEYS = ["ae_input"]

    model_inputs = {}
    for key in MODEL_KEYS:
        t = batch[key]
        if device is not None:
            t = t.to(device)
        model_inputs[key] = t

    for key in OPTIONAL_KEYS:
        if key in batch:
            t = batch[key]
            if device is not None:
                t = t.to(device)
            model_inputs[key] = t

    targets = batch["target"]
    if device is not None:
        targets = targets.to(device)

    return model_inputs, targets


# ============================================================================
# Convenience: Event Calendar from DataFrame
# ============================================================================

def events_from_dataframe(
    df,
    date_col: str = "date",
    type_col: str = "event_type",
    surprise_col: str = "surprise",
    direction_col: str = "direction",
    magnitude_col: str = "magnitude",
    revision_col: str = "revision",
) -> Dict[date, List[ScheduledEvent]]:
    """Convert a pandas DataFrame into events_by_date dict format."""
    import pandas as pd

    events: Dict[date, List[ScheduledEvent]] = {}

    for _, row in df.iterrows():
        d = row[date_col]
        if isinstance(d, pd.Timestamp):
            d = d.date()
        elif isinstance(d, datetime):
            d = d.date()

        ev = ScheduledEvent(
            date=d,
            event_type=str(row[type_col]),
            surprise=float(row.get(surprise_col, 0.0) or 0.0),
            direction=float(row.get(direction_col, 0.0) or 0.0),
            magnitude=float(row.get(magnitude_col, 0.0) or 0.0),
            revision=float(row.get(revision_col, 0.0) or 0.0),
        )

        if d not in events:
            events[d] = []
        events[d].append(ev)

    return events


def macro_from_dataframe(
    df,
    date_col: str = "date",
    feature_cols: Optional[List[str]] = None,
) -> Dict[date, np.ndarray]:
    """Convert a pandas DataFrame of daily macro features into a dict."""
    import pandas as pd

    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c != date_col
        ]

    macro: Dict[date, np.ndarray] = {}
    for _, row in df.iterrows():
        d = row[date_col]
        if isinstance(d, pd.Timestamp):
            d = d.date()
        elif isinstance(d, datetime):
            d = d.date()
        macro[d] = np.array([row[c] for c in feature_cols], dtype=np.float32)

    return macro


def summarize_event_routing(tickers: List[str]) -> str:
    """Print a summary of which events each ticker type will receive."""
    registry = build_ticker_registry(tickers)
    lines = ["Event Routing Summary", "=" * 60]

    types_seen: Dict[str, Tuple[str, Set[str]]] = {}
    for ticker, meta in registry.items():
        tt = meta.ticker_type
        allowed = EVENTS_BY_TICKER_TYPE.get(tt, UNIVERSAL_EVENTS)
        if tt not in types_seen:
            types_seen[tt] = (meta.asset_subclass, allowed)

    for tt, (subclass, allowed) in sorted(types_seen.items()):
        universal = allowed & UNIVERSAL_EVENTS
        specific = allowed - UNIVERSAL_EVENTS
        tickers_of_type = [t for t, m in registry.items() if m.ticker_type == tt]
        lines.append(f"\n{tt.upper()} ({', '.join(sorted(tickers_of_type))})")
        lines.append(f"  Universal ({len(universal)}): {', '.join(sorted(universal))}")
        if specific:
            lines.append(f"  Specific  ({len(specific)}): {', '.join(sorted(specific))}")
        else:
            lines.append(f"  Specific  (0): none")
        lines.append(f"  Total: {len(allowed)} event types")

    return "\n".join(lines)
