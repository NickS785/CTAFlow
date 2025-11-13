import pytest


@pytest.fixture()
def historical_screener(example_dataframe):
    from CTAFlow.screeners.historical_screener import HistoricalScreener

    return HistoricalScreener({"HO": example_dataframe}, file_mgr=None, verbose=False)


def test_run_screens_returns_patterns_structure(historical_screener):
    from CTAFlow.screeners.historical_screener import ScreenParams

    params = ScreenParams(
        screen_type="seasonality",
        name="example_screen",
        target_times=["00:30", "00:45"],
        season="fall",
        seasonality_session_end="23:59",
    )

    results = historical_screener.run_screens([params])

    assert "example_screen" in results
    screen_payload = results["example_screen"]
    assert "HO" in screen_payload

    ticker_payload = screen_payload["HO"]
    assert "error" not in ticker_payload
    assert isinstance(ticker_payload.get("strongest_patterns"), list)
    assert "time_predictability" in ticker_payload

    # Each time bucket should expose context needed by PatternExtractor
    for time_payload in ticker_payload["time_predictability"].values():
        assert "target_times_hhmm" in time_payload
        assert "period_length_min" in time_payload


def test_rank_seasonal_strength_includes_weekday_context(historical_screener):
    months_meta = {
        "months_active": [9, 10, 11],
        "months_mask_12": "000000001110",
        "months_names": ["Sep", "Oct", "Nov"],
    }

    ticker_results = {
        "pattern_context": {
            **months_meta,
            "period_length_min": 120,
            "target_times_hhmm": ["09:30"],
            "regime_filter": None,
        },
        "months_active": months_meta["months_active"],
        "months_mask_12": months_meta["months_mask_12"],
        "months_names": months_meta["months_names"],
        "time_predictability": {
            "09:30": {
                "time": "09:30",
                "next_week_significant": True,
                "next_week_corr": 0.42,
                "next_week_pvalue": 0.01,
                "next_day_significant": False,
                "months_active": months_meta["months_active"],
                "months_mask_12": months_meta["months_mask_12"],
                "months_names": months_meta["months_names"],
                "target_times_hhmm": ["09:30"],
                "period_length_min": 120,
                "regime_filter": None,
                "weekday_prevalence": {
                    "most_prevalent_day": "Friday",
                    "strongest_days": ["Friday", "Monday"],
                },
            }
        },
    }

    patterns = historical_screener._rank_seasonal_strength(ticker_results)
    time_patterns = [p for p in patterns if p.get("type") == "time_predictive_nextweek"]

    assert time_patterns, "expected at least one time predictive pattern"
    pattern = time_patterns[0]

    assert "09:30" in pattern["description"]
    assert pattern.get("most_prevalent_day") == "Friday"
    assert pattern.get("strongest_days") == ["Friday", "Monday"]
