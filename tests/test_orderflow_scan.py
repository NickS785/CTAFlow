import pandas as pd
import numpy as np

from CTAFlow.CTAFlow.utils.session import filter_session_ticks, filter_session_bars
from CTAFlow.CTAFlow.utils.volume_bucket import auto_bucket_size, ticks_to_volume_buckets
from CTAFlow.CTAFlow.screeners.orderflow_scan import OrderflowParams, orderflow_scan


def test_filter_session_helpers():
    ticks = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01 08:00", periods=5, freq="15min"),
            "AskVolume": [1, 2, 3, 4, 5],
            "BidVolume": [1, 1, 1, 1, 1],
        }
    )
    filtered_ticks = filter_session_ticks(ticks, "America/Chicago", "08:30", "09:00")
    assert filtered_ticks["ts"].dt.tz.zone == "America/Chicago"
    assert filtered_ticks["ts"].dt.strftime("%H:%M").tolist() == ["08:30", "08:45", "09:00"]

    bars = pd.DataFrame(
        {
            "value": range(7),
        },
        index=pd.date_range("2024-01-01 07:45", periods=7, freq="15min"),
    )
    filtered_bars = filter_session_bars(bars, "America/Chicago", "08:30", "09:15")
    assert filtered_bars.index.tz.zone == "America/Chicago"
    assert filtered_bars.index.strftime("%H:%M").tolist() == ["08:30", "08:45", "09:00", "09:15"]


def test_auto_bucket_size_and_bucketization():
    ts = pd.date_range("2024-01-01 08:30", periods=10, freq="30min", tz="America/Chicago")
    ts = ts.append(pd.date_range("2024-01-02 08:30", periods=10, freq="30min", tz="America/Chicago"))
    ticks = pd.DataFrame(
        {
            "ts": ts,
            "AskVolume": np.full(len(ts), 5.0),
            "BidVolume": np.full(len(ts), 5.0),
        }
    )

    bucket = auto_bucket_size(ticks, cadence_target=10, grid_multipliers=(1.0,))
    assert bucket == 10

    bucketed = ticks_to_volume_buckets(ticks, bucket)
    assert bucketed["TotalVolume"].eq(10).all()
    assert bucketed["bucket"].iloc[-1] == len(bucketed) - 1


def _make_tick_series(day: str, neutral_buckets: int, pressure_buckets: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{day} 08:30", tz="America/Chicago")
    rows = []
    for i in range(neutral_buckets + pressure_buckets):
        ts = start + pd.Timedelta(minutes=10 * i)
        if i < neutral_buckets:
            ask, bid = 20.0, 20.0
        else:
            ask, bid = 36.0, 4.0
        rows.append({"ts": ts, "AskVolume": ask, "BidVolume": bid})
    return pd.DataFrame(rows)


def test_orderflow_scan_pipeline():
    day1 = _make_tick_series("2024-01-02", neutral_buckets=8, pressure_buckets=0)
    day2 = _make_tick_series("2024-01-03", neutral_buckets=6, pressure_buckets=4)
    ticks = pd.concat([day1, day2], ignore_index=True)

    def loader(_symbol: str) -> pd.DataFrame:
        return ticks.copy()

    params = OrderflowParams(
        session_start="08:30",
        session_end="10:30",
        tz="America/Chicago",
        bucket_size=40,
        threshold_z=1.0,
        vpin_window=3,
        min_days=1,
    )

    out = orderflow_scan(loader, ["TEST"], params)
    assert "TEST" in out
    payload = out["TEST"]
    assert "df_buckets" in payload and not payload["df_buckets"].empty
    meta = payload["metadata"]
    assert meta["bucket_size"] == 40
    assert meta["n_sessions"] == 2
    assert "df_weekly" in payload
    assert "df_wom_weekday" in payload
    events = payload["df_events"]
    if not events.empty:
        assert (events["run_len"] >= 2).any()
