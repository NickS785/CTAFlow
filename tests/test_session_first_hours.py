from __future__ import annotations

import numpy as np
import pandas as pd

from screeners import session_first_hours as sfh


def _build_intraday(
    days: int,
    *,
    first_hours: int = 2,
    bar_seconds: int = 60,
    tz: str = "America/Chicago",
    base_price: float = 100.0,
    intra_session_jump: float = 0.2,
    final_session_jump: float | None = None,
    base_volume: float = 10.0,
    final_volume_scale: float = 1.0,
) -> pd.DataFrame:
    """Construct a synthetic intraday series for testing."""

    start = pd.Timestamp("2025-09-01", tz=tz)
    frames: list[pd.DataFrame] = []
    bars_per_session = int((first_hours * 3600) / bar_seconds)
    if bars_per_session <= 0:
        bars_per_session = int((2 * 3600) / bar_seconds)

    for day in range(days):
        open_time = (start + pd.Timedelta(days=day)).normalize() + pd.Timedelta(hours=17)
        idx = pd.date_range(
            open_time,
            periods=bars_per_session,
            freq=f"{bar_seconds}s",
            tz=tz,
        )
        session_base = base_price + day * 0.5
        jump = intra_session_jump
        if day == days - 1 and final_session_jump is not None:
            jump = final_session_jump
        close_price = session_base + jump
        prices = np.linspace(session_base, close_price, bars_per_session)
        volumes = np.full(bars_per_session, base_volume, dtype=float)
        if day == days - 1:
            volumes *= final_volume_scale
        frame = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 0.25,
                "low": prices - 0.25,
                "close": prices,
                "volume": volumes,
                "num_trades": np.ones(bars_per_session, dtype=float),
                "bid_vol": volumes / 2,
                "ask_vol": volumes / 2,
            },
            index=idx,
        )
        frames.append(frame)

    return pd.concat(frames)


def test_relative_volume_is_one(monkeypatch):
    def fake_loader(symbol: str, *_, **__) -> pd.DataFrame:
        return _build_intraday(8)

    monkeypatch.setattr(sfh, "_load_intraday", fake_loader)

    params = sfh.SessionFirstHoursParams(
        symbols=["CL"],
        start_date="2025-09-01",
        end_date="2025-09-08",
        lookback_days=4,
        session_start_hhmm="17:00",
        first_hours=2,
        bar_seconds=60,
    )

    result = sfh.run_session_first_hours(params)
    rel = result[("relative_volume_tod", "CL")].dropna()
    assert not rel.empty
    assert np.allclose(rel.iloc[1:], 1.0, atol=1e-6)


def test_min_bars_filter(monkeypatch):
    base = _build_intraday(1)
    truncated = base.iloc[:4]

    def fake_loader(symbol: str, *_, **__) -> pd.DataFrame:
        return truncated

    monkeypatch.setattr(sfh, "_load_intraday", fake_loader)

    params = sfh.SessionFirstHoursParams(
        symbols=["NG"],
        start_date="2025-09-01",
        end_date="2025-09-02",
        first_hours=2,
        bar_seconds=60,
        min_bars_in_window=10,
        lookback_days=2,
    )

    result = sfh.run_session_first_hours(params)
    assert result.empty


def test_rank_ordering_and_metrics(monkeypatch):
    mapping = {
        "A": _build_intraday(5, final_session_jump=0.4, final_volume_scale=0.8),
        "B": _build_intraday(5, final_session_jump=0.8, final_volume_scale=1.6),
    }

    def fake_loader(symbol: str, *_, **__) -> pd.DataFrame:
        return mapping[symbol]

    monkeypatch.setattr(sfh, "_load_intraday", fake_loader)

    params = sfh.SessionFirstHoursParams(
        symbols=["A", "B"],
        start_date="2025-09-01",
        end_date="2025-09-05",
        first_hours=2,
        bar_seconds=60,
        lookback_days=3,
    )

    result = sfh.run_session_first_hours(params)
    assert ("momentum", "A") in result.columns
    assert ("momentum", "B") in result.columns
    latest_idx = result.index.max()
    latest = result.loc[latest_idx]

    assert latest[("momentum", "B")] > latest[("momentum", "A")]
    assert latest[("rank_momentum", "B")] == 1.0
    assert latest[("rank_momentum", "A")] > latest[("rank_momentum", "B")]

    rel_vol = result[("relative_volume_tod", "B")].dropna()
    assert rel_vol.iloc[-1] > 1.0
