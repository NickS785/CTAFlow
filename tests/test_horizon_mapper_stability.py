from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "CTAFlow" / "strategy" / "screener_pipeline.py"

spec = importlib.util.spec_from_file_location("screener_pipeline", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

HorizonMapper = module.HorizonMapper
HorizonInputError = module.HorizonInputError


class ListLogger:
    def __init__(self) -> None:
        self.records: list[str] = []

    def warning(self, message: str, *args: Any) -> None:  # pragma: no cover - signature shim
        if args:
            try:
                rendered = message % args
            except Exception:  # pragma: no cover - defensive formatting
                rendered = " ".join([message, *map(str, args)])
        else:
            rendered = message
        self.records.append(rendered)


@pytest.fixture
def clean_bars() -> pd.DataFrame:
    tz = "UTC"
    sessions = []
    timestamps = []
    opens = []
    closes = []
    base_open = 100.0
    for idx, session_start in enumerate(pd.date_range("2024-01-02", periods=4, freq="D", tz=tz)):
        session_label = chr(ord("A") + idx)
        for offset in (pd.Timedelta(minutes=25), pd.Timedelta(minutes=30), pd.Timedelta(minutes=35)):
            ts = session_start + pd.Timedelta(hours=13) + offset
            timestamps.append(ts)
            sessions.append(session_label)
            open_price = base_open + idx
            close_price = open_price + (0.3 * (offset.seconds // 60))
            opens.append(open_price)
            closes.append(close_price)

    df = pd.DataFrame({
        "ts": timestamps,
        "open": opens,
        "close": closes,
        "session_id": sessions,
    }).sort_values("ts").reset_index(drop=True)
    return df


def _time_pattern(time_text: str, *, key: str | None = None) -> dict[str, Any]:
    payload = {"time": time_text}
    return {
        "pattern_type": "time_predictive_nextday",
        "pattern_payload": payload,
        "key": key or f"time_nextday_{time_text.replace(':', '')}",
    }


def test_build_xy_materializes_time_nextday_gates_nonempty(clean_bars: pd.DataFrame) -> None:
    mapper = HorizonMapper()
    pattern = _time_pattern("13:30:00")

    result = mapper.build_xy(clean_bars, [pattern], predictor_minutes=5)

    assert not result.empty
    assert (result["gate"].str.contains("time_nextday")).all()


def test_missing_gate_becomes_warning_not_error(clean_bars: pd.DataFrame) -> None:
    logger = ListLogger()
    mapper = HorizonMapper(log=logger)
    pattern = _time_pattern("14:00:00")

    # Disable automatic gate materialisation so the gate is absent.
    result = mapper.build_xy(clean_bars, [pattern], ensure_gates=False)

    assert result.empty
    assert any("Gate column missing" in msg for msg in logger.records)


def test_dtype_coercion_and_tz_localization() -> None:
    data = pd.DataFrame({
        "ts": ["2024-01-02 13:25", "2024-01-02 13:30", "2024-01-03 13:25", "2024-01-03 13:30"],
        "open": ["100", "100", "101", "101"],
        "close": ["100.6", "100.8", "101.6", "101.9"],
        "session_id": ["A", "A", "B", "B"],
    })

    mapper = HorizonMapper(allow_naive_ts=True, tz="UTC")
    pattern = _time_pattern("13:30:00")

    result = mapper.build_xy(data, [pattern], predictor_minutes=5)
    assert not result.empty
    assert result["ts_decision"].dt.tz is not None


def test_naive_ts_raises_or_coerces_per_flag(clean_bars: pd.DataFrame) -> None:
    naive = clean_bars.copy()
    naive["ts"] = naive["ts"].dt.tz_convert(None)
    pattern = _time_pattern("13:30:00")

    mapper_fail = HorizonMapper()
    with pytest.raises(HorizonInputError):
        mapper_fail.build_xy(naive, [pattern])

    mapper_ok = HorizonMapper(allow_naive_ts=True)
    result = mapper_ok.build_xy(naive, [pattern], predictor_minutes=5)
    assert not result.empty


def test_returns_no_infs_nans_after_policy_drop(clean_bars: pd.DataFrame) -> None:
    noisy = clean_bars.copy()
    noisy.loc[noisy.index[-1], "close"] = 0.0
    pattern = _time_pattern("13:30:00")

    mapper = HorizonMapper()
    result = mapper.build_xy(noisy, [pattern], predictor_minutes=5)

    assert not result.empty
    assert result["returns_x"].notna().all()
    assert result["returns_y"].notna().all()
    assert np.isfinite(result["returns_x"]).all()
    assert np.isfinite(result["returns_y"]).all()


def test_time_match_auto_picks_microsecond_suffix() -> None:
    tz = "UTC"
    ts_base = pd.Timestamp("2024-01-02 13:30", tz=tz)
    df = pd.DataFrame(
        {
            "ts": [
                ts_base - pd.Timedelta(days=1),
                ts_base - pd.Timedelta(days=1) + pd.Timedelta(minutes=1),
                ts_base - pd.Timedelta(minutes=1) + pd.Timedelta(microseconds=500000),
                ts_base + pd.Timedelta(microseconds=500000),
                ts_base + pd.Timedelta(days=1),
            ],
            "open": [100.0, 100.0, 101.0, 101.0, 102.0],
            "close": [100.4, 100.6, 101.2, 101.4, 102.5],
            "session_id": ["A", "A", "B", "B", "C"],
        }
    )

    pattern = _time_pattern("13:30:00.500000")
    mapper = HorizonMapper()

    result = mapper.build_xy(df, [pattern], predictor_minutes=1)
    assert not result.empty
    assert result["gate"].str.contains(".500000").any()


def test_sessionized_next_day_vs_calendar_day() -> None:
    tz = "UTC"
    df = pd.DataFrame({
        "ts": [
            pd.Timestamp("2024-01-02 20:55", tz=tz),
            pd.Timestamp("2024-01-02 21:00", tz=tz),
            pd.Timestamp("2024-01-02 23:00", tz=tz),
            pd.Timestamp("2024-01-02 23:30", tz=tz),
            pd.Timestamp("2024-01-03 00:25", tz=tz),
            pd.Timestamp("2024-01-03 00:30", tz=tz),
        ],
        "open": [100.0, 100.0, 100.0, 101.0, 101.0, 101.0],
        "close": [100.3, 100.5, 100.6, 101.2, 101.3, 101.6],
        "session_id": ["session_0", "session_0", "session_0", "session_1", "session_1", "session_1"],
    })
    pattern = _time_pattern("21:00:00", key="overnight-test")

    mapper = HorizonMapper()
    result = mapper.build_xy(df, [pattern], predictor_minutes=5)

    assert not result.empty
    first_row = result.iloc[0]
    expected = np.log(101.6 / 100.6)
    assert pytest.approx(expected) == first_row["returns_y"]


def test_asof_alignment_with_tolerance_and_warning(clean_bars: pd.DataFrame) -> None:
    logger = ListLogger()
    mapper = HorizonMapper(log=logger)
    pattern = _time_pattern("13:30:00")

    noisy = clean_bars.copy()
    drop_sessions = {"B", "C", "E"}
    drop_mask = (noisy["ts"].dt.minute == 25) & noisy["session_id"].isin(drop_sessions)
    noisy = noisy.loc[~drop_mask].reset_index(drop=True)

    result = mapper.build_xy(
        noisy,
        [pattern],
        predictor_minutes=5,
        asof_tolerance="10s",
        debug=True,
    )

    assert not result.empty
    assert any("merge_asof" in msg for msg in logger.records)


def test_gate_inference_slug_over_time_fallback(clean_bars: pd.DataFrame) -> None:
    df = clean_bars.copy()
    df["custom_slug_gate"] = 0
    df.loc[df["ts"].dt.time == pd.Timestamp("2024-01-02 13:30").time(), "custom_slug_gate"] = 1
    df["time_nextday_133000_gate"] = df["custom_slug_gate"]

    pattern = {
        "pattern_type": "time_predictive_nextday",
        "pattern_payload": {"time": "13:30:00"},
        "key": "Custom Slug",
    }

    mapper = HorizonMapper()
    result = mapper.build_xy(df, [pattern], ensure_gates=False, predictor_minutes=5)

    assert not result.empty
    assert (result["gate"] == "custom_slug_gate").all()
