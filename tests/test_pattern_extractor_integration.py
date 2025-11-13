import pandas as pd
import pytest


@pytest.fixture()
def pattern_extractor(example_dataframe):
    from CTAFlow.screeners.historical_screener import HistoricalScreener, ScreenParams
    from CTAFlow.screeners.pattern_extractor import PatternExtractor

    screener = HistoricalScreener({"HO": example_dataframe}, file_mgr=None, verbose=False)
    params = ScreenParams(
        screen_type="seasonality",
        name="example_screen",
        target_times=["00:30", "00:45"],
        season="fall",
        seasonality_session_end="23:59",
    )
    results = screener.run_screens([params])
    extractor = PatternExtractor(screener, results, [params])
    return extractor, params


def test_pattern_extractor_emits_named_patterns(pattern_extractor):
    extractor, params = pattern_extractor
    patterns = extractor.patterns.get("HO", {})

    assert patterns, "expected at least one pattern for HO"
    for key, payload in patterns.items():
        assert key.startswith(params.name)
        assert payload.get("key") == key
        assert payload.get("symbol") == "HO"
        metadata = payload.get("metadata", {})
        assert metadata.get("target_times_hhmm")
        assert "period_length_min" in metadata
        assert metadata.get("returns_period")
        assert metadata.get("period_length") == metadata.get("returns_period")
        assert metadata.get("months_active")
        assert metadata.get("months_names")
        assert metadata.get("months_mask_12")
        assert metadata.get("pattern_origin")
        assert metadata.get("screen_type") is not None
        assert payload["pattern_payload"].get("time") is not None


def test_screener_pipeline_accepts_summary_dicts(pattern_extractor, example_dataframe):
    from CTAFlow.strategy.screener_pipeline import ScreenerPipeline

    extractor, _ = pattern_extractor
    summaries = [summary.as_dict() for summary in extractor._pattern_index.get("HO", {}).values()]
    assert summaries, "pattern index unexpectedly empty"

    bars = example_dataframe.reset_index().rename(columns={"Datetime": "ts", "Open": "open", "Close": "close"})
    bars["ts"] = pd.to_datetime(bars["ts"]).dt.tz_localize("America/Chicago")
    bars["session_id"] = bars["ts"].dt.strftime("%Y-%m-%d")

    pipeline = ScreenerPipeline()
    enriched = pipeline.build_features(bars, summaries)

    gate_columns = [col for col in enriched.columns if col.endswith("_gate")]
    assert gate_columns, "no gate columns were produced"
    assert any(enriched[col].sum() >= 0 for col in gate_columns)
