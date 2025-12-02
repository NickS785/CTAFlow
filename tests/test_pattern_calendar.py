import json

import pandas as pd

from CTAFlow.screeners.pattern_calendar import ActivePatternCalendar, PatternVault
from CTAFlow.data import ResultsClient
from CTAFlow.strategy.backtester import BacktestSummary


def test_pattern_vault_stores_and_refreshes_active(tmp_path):
    hdf_path = tmp_path / "results.h5"
    results_client = ResultsClient(results_path=hdf_path)
    vault = PatternVault(results_client=results_client, features_root=tmp_path)

    active_payload = pd.DataFrame(
        {
            "pattern": ["gap_fill"],
            "event_time": [pd.Timestamp("2024-01-01T12:00:00Z")],
            "expected_turnout": [0.8],
        }
    )

    vault.store_patterns("ES", "winter_screen", active_payload, active=True)

    stored = vault.get_patterns("ES", "winter_screen")
    assert stored.loc[0, "features_fp"].endswith("ES_features.parquet")
    assert stored.loc[0, "active"] == 1

    # Disabled payload should not appear in consolidated active view
    disabled_payload = active_payload.assign(event_time=pd.Timestamp("2024-01-02T12:00:00Z"))
    disabled_payload["active"] = 0
    vault.store_patterns("ES", "summer_screen", disabled_payload, active=False)

    active = vault.get_active_patterns("ES")
    assert len(active) == 1
    assert active.loc[0, "pattern"] == "gap_fill"
    assert active.loc[0, "source_key"] == "screeners/ES/winter_screen"


def test_active_pattern_calendar_filters_horizon(tmp_path):
    hdf_path = tmp_path / "results.h5"
    vault = PatternVault(results_client=ResultsClient(results_path=hdf_path))

    now = pd.Timestamp("2024-01-01T00:00:00Z")
    payload = pd.DataFrame(
        {
            "pattern": ["overnight_break"],
            "event_time": [now + pd.Timedelta(hours=5)],
            "expected_turnout": [1.2],
        }
    )
    vault.store_patterns("CL", "session_scan", payload, active=True)

    calendar = ActivePatternCalendar(vault).build_calendar(["CL"], now=now, horizon_hours=24)

    assert len(calendar) == 1
    row = calendar.iloc[0]
    assert row.ticker == "CL"
    assert row.pattern == "overnight_break"
    assert row.expected_turnout == 1.2
    assert row.event_time == payload.loc[0, "event_time"]


def test_backtest_summary_is_serialized_for_storage(tmp_path):
    hdf_path = tmp_path / "results.h5"
    vault = PatternVault(results_client=ResultsClient(results_path=hdf_path))

    summary = BacktestSummary(0.5, 0.1, 0.6, 1.2, 0.05, 10)
    payload = pd.DataFrame(
        {
            "pattern": ["carry_spread"],
            "event_time": [pd.Timestamp("2024-06-01T15:00:00Z")],
            "backtest_summary": [summary],
        }
    )

    vault.store_patterns("NG", "carry_screen", payload, active=True)

    stored = vault.get_patterns("NG", "carry_screen")
    serialized = stored.loc[0, "backtest_summary"]
    assert isinstance(serialized, str)

    decoded = json.loads(serialized)
    assert decoded["total_return"] == summary.total_return
    assert decoded["trades"] == summary.trades

    active = vault.get_active_patterns("NG")
    assert active.loc[0, "backtest_summary"] == serialized
