import types
import pandas as pd

from CTAFlow.strategy.backtester import BacktestSummary, ScreenerBacktester
from CTAFlow.strategy.screener_pipeline import ScreenerPipeline


def test_ranking_scores_results():
    results = {
        "alpha": {"summary": BacktestSummary(0.2, 0.0, 0.0, 0.0, 0.05, 10)},
        "beta": {"summary": BacktestSummary(0.15, 0.0, 0.0, 0.0, 0.03, 8)},
        "gamma": {"summary": BacktestSummary(-0.1, 0.0, 0.0, 0.0, 0.02, 5)},
    }

    ranking = ScreenerBacktester.rank_results(results)
    assert [key for key, _ in ranking] == ["beta", "alpha", "gamma"]


def test_concurrent_backtests_return_ranked(monkeypatch):
    pipeline = ScreenerPipeline(tz="America/Chicago", use_gpu=False)

    bars = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=3, freq="D", tz="America/Chicago"),
            "open": [1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25],
            "high": [1.06, 1.16, 1.26],
            "low": [0.99, 1.09, 1.19],
            "volume": [100, 120, 140],
            "session_id": ["s1", "s2", "s3"],
        }
    )

    patterns = {
        "alpha": {"pattern_type": "weekday_mean", "pattern_payload": {"weekday": "monday"}},
        "beta": {"pattern_type": "weekday_mean", "pattern_payload": {"weekday": "tuesday"}},
    }

    summaries = {
        "alpha": BacktestSummary(0.2, 0.0, 0.0, 0.0, 0.04, 3),
        "beta": BacktestSummary(0.18, 0.0, 0.0, 0.0, 0.02, 3),
    }

    def fake_feature_sets(self, bars, patterns, **kwargs):
        return {key: bars for key in patterns}

    def fake_backtest(self, featured, patterns, **kwargs):
        key = next(iter(patterns))
        return {"summary": summaries[key]}

    monkeypatch.setattr(pipeline, "build_feature_sets", types.MethodType(fake_feature_sets, pipeline))
    monkeypatch.setattr(pipeline, "backtest_threshold", types.MethodType(fake_backtest, pipeline))

    results = pipeline.concurrent_pattern_backtests(bars, patterns, max_workers=1)
    assert list(results.keys()) == ["beta", "alpha"]
    assert results["beta"]["ranking_score"] > results["alpha"]["ranking_score"]
    assert results["beta"]["ranking_score"] == ScreenerBacktester.ranking_score(summaries["beta"])
