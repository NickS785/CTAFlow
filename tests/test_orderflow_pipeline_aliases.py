import importlib.util
import pathlib
import sys
import types

import numpy as np
import pandas as pd

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "CTAFlow" / "strategy" / "auction_market.py"
_SPEC = importlib.util.spec_from_file_location("_cta_strategy_auction_market", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("_cta_strategy_auction_market", _MODULE)
_SPEC.loader.exec_module(_MODULE)

OrderflowPipeline = _MODULE.OrderflowPipeline
SessionCfg = _MODULE.SessionCfg
FlowCfg = _MODULE.FlowCfg
FeatureCfg = _MODULE.FeatureCfg
LabelCfg = _MODULE.LabelCfg


def _make_sample_df(rows: int = 240) -> pd.DataFrame:
    base_ts = pd.Timestamp("2024-01-02 02:30:00", tz="America/Chicago")
    ts = pd.date_range(base_ts, periods=rows, freq="1min")
    price = 100.0 + np.sin(np.linspace(0, 4, rows))
    ask = 5.0 + np.cos(np.linspace(0, 6, rows))
    bid = 4.5 + np.sin(np.linspace(0, 6, rows))
    trades = 1 + (np.arange(rows) % 3)

    data = {
        "Open": price,
        "High": price + 0.1,
        "Low": price - 0.1,
        "Close": price,
        "NumTrades": trades,
        "TotalVolume": ask + bid,
        "BidVolume": bid,
        "AskVolume": ask,
        "Contract": "TEST",
        "Ticker": "TEST",
        "RollDate": pd.Timestamp("2024-01-01"),
        "ContractExpiry": pd.Timestamp("2024-03-01"),
        "SourceFile": "synthetic.csv",
        "ts": ts,
        "px": price,
        "buy": ask,
        "sell": bid,
        "ntr": trades,
    }
    return pd.DataFrame(data)


def test_orderflow_pipeline_detects_short_alias_columns():
    df = _make_sample_df()
    pipeline = OrderflowPipeline()

    detected = pipeline._detect_cols(df)
    assert detected["ts"] == "ts"
    assert detected["px"] in {"px", "Close"}
    assert detected["buy"] in {"buy", "AskVolume"}
    assert detected["sell"] in {"sell", "BidVolume"}
    assert detected["ntr"] in {"ntr", "NumTrades"}

    tick_size = 0.25
    session_cfg = SessionCfg("02:30:00", "11:30:00", tick=tick_size, profile_tick=tick_size * 2)
    flow_cfg = FlowCfg()
    feature_cfg = FeatureCfg()
    label_cfg = LabelCfg(tgt_ticks=40, stp_ticks=30)

    def _dummy_fit(self, entries, X):
        return types.SimpleNamespace(coef_=np.zeros(len(X.columns))), 0.0

    pipeline.fit_predictor = types.MethodType(_dummy_fit, pipeline)

    result = pipeline.run_pipeline(
        df,
        sess_cfg=session_cfg,
        flow_cfg=flow_cfg,
        feat_cfg=feature_cfg,
        lab_cfg=label_cfg,
    )

    assert set(result.keys()) == {"entries", "labeled", "X", "coef"}
    for key, value in result.items():
        assert isinstance(value, pd.DataFrame)
