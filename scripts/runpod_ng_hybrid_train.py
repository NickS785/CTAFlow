#!/usr/bin/env python
"""
RunPod Serverless handler: NatGas Hybrid Intraday Optuna + Final Training.

Combines the full ng_hybrid_intraday_optuna notebook into a single serverless
job.  Expects model data at /workspace/model_data/NG/ and env vars loaded from
/workspace/model_data/dot.env.

Inputs (via event["input"]):
    trial_rank       : int   — 0=best, 1=2nd-best, etc.  (default 0)
    n_trials         : int   — Optuna trials              (default 40)
    max_epochs_opt   : int   — max epochs per trial       (default 80)
    max_epochs_final : int   — final training epochs      (default 200)
    patience_opt     : int   — early-stop patience (opt)  (default 10)
    patience_final   : int   — early-stop patience (final)(default 20)
    bar_minutes      : int   — resample freq              (default 15)
    target_horizon   : int   — forecast horizon in bars   (default 10)
    seq_len          : int   — lookback bars              (default 20)
    ae_window_days   : int   — regime encoder days        (default 10)
    n_classes        : int   — quartile classes            (default 4)
    session_start    : str   — e.g. "02:30"               (default "02:30")
    session_end      : str   — e.g. "15:00"               (default "15:00")

Outputs:
    model_path, best_params, backtest_metrics, trial_log summary
"""

import os
import sys
import copy
import json
import math
import time
import warnings
from dataclasses import asdict
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv("/workspace/model_data/dot.env")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# CTAFlow imports
from CTAFlow.models.deep_learning.multi_branch.ng_moe import (
    HybridConfig,
    HybridMixtureNetwork,
    HybridLoss,
)
from CTAFlow.data.raw_formatting.intraday_manager import read_exported_df
from CTAFlow.models.prep.intraday_continuous import (
    ContinuousIntradayPrep,
    SessionSpec,
)
from CTAFlow.data.datasets.intraday_hybrid import IntradayHybridDataset

# macrOS-Int imports
sys.path.insert(0, "/workspace/MacrOS-Intel")
from MacrOSINT.data.sources.eia.api_tools import NatGasHelper
from MacrOSINT.models.energy.natgas_storage_forecast import (
    NatGasStorageForecaster,
    fetch_storage_data,
    ConsensusForecast,
    compute_degree_days,
    compute_spline_hdd_basis,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def _select_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        t = torch.zeros(1, device="cuda")
        _ = t + 1
        return "cuda"
    except RuntimeError:
        return "cpu"


# ===================================================================
# Data pipeline
# ===================================================================

def load_and_prepare_data(cfg):
    """Load intraday CSV, EIA storage, weather; build features and datasets."""
    DATA_DIR = Path("/workspace/model_data")
    SAVE_DIR = cfg["save_dir"]
    INTRADAY_CSV = DATA_DIR / "NG" / "intraday_2.csv"
    EIA_HDF = str(DATA_DIR / "new_ng_eia_cache.hdf")
    WEATHER_HDF = str(DATA_DIR / "new_weather.hdf")

    BAR_MINUTES = cfg["bar_minutes"]
    TARGET_HORIZON_BARS = cfg["target_horizon"]
    SEQ_LEN = cfg["seq_len"]
    AE_WINDOW_DAYS = cfg["ae_window_days"]
    N_CLASSES = cfg["n_classes"]
    SESSION = SessionSpec("USA", cfg["session_start"], cfg["session_end"])

    # --- Load & resample ---
    print(f"[DATA] Loading {INTRADAY_CSV}")
    raw_df = read_exported_df(str(INTRADAY_CSV))
    raw_df.columns = [c.lower() for c in raw_df.columns]
    for old, new in [("last", "close"), ("vol", "volume"), ("numberoftrades", "ticks")]:
        if old in raw_df.columns and new not in raw_df.columns:
            raw_df.rename(columns={old: new}, inplace=True)

    resample_rule = f"{BAR_MINUTES}min"
    ohlcv_agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in raw_df.columns:
        ohlcv_agg["volume"] = "sum"
    if "ticks" in raw_df.columns:
        ohlcv_agg["ticks"] = "sum"
    df = raw_df.resample(resample_rule).agg(ohlcv_agg).dropna(subset=["close"])
    session_start = pd.Timestamp(SESSION.start).time()
    session_end = pd.Timestamp(SESSION.end).time()
    df = df.between_time(session_start, session_end)
    print(f"[DATA] {df.shape[0]} bars @ {resample_rule}, {df.index[0].date()} to {df.index[-1].date()}")

    # --- EIA storage (always fresh) ---
    START = df.index[0].strftime("%Y-%m")
    END = df.index[-1].strftime("%Y-%m")
    ng_helper = NatGasHelper()
    storage_wkly = fetch_storage_data(ng_helper, start=START, end=END)
    NatGasStorageForecaster.save_eia_cache(storage=storage_wkly, hdf_path=EIA_HDF)
    print(f"[DATA] EIA storage: {storage_wkly.shape[0]} weeks")

    try:
        cf = ConsensusForecast()
        cf.fit(storage_wkly["storage_change"])
        surprise_df = cf.transform()
        storage_wkly = storage_wkly.join(
            surprise_df[["consensus_est", "surprise"]], how="left"
        )
    except Exception:
        chg = storage_wkly["storage_change"]
        storage_wkly["consensus_est"] = chg.rolling(4).mean()
        storage_wkly["surprise"] = chg - storage_wkly["consensus_est"]

    # --- Weather (load or regenerate) ---
    daily_weather = NatGasStorageForecaster.load_weather_hdf(hdf_path=WEATHER_HDF)
    data_start = df.index[0].normalize()
    if daily_weather is not None and not daily_weather.empty:
        if daily_weather.index[0] > data_start + pd.Timedelta(days=30):
            daily_weather = None
        else:
            daily_weather = daily_weather[
                (daily_weather.index >= data_start)
                & (daily_weather.index <= df.index[-1].normalize())
            ]
    if daily_weather is None or daily_weather.empty:
        print("[DATA] Fetching weather (epoch-aware)...")
        forecaster = NatGasStorageForecaster(
            config_dir=str(SAVE_DIR / "weather_configs"),
        )
        daily_weather = forecaster._fetch_weather_by_epoch(
            data_start.date(), df.index[-1].date()
        )
        NatGasStorageForecaster.save_weather_hdf(daily_weather, hdf_path=WEATHER_HDF)
    print(f"[DATA] Weather: {daily_weather.shape[0]} days")

    # --- Feature engineering ---
    prep = ContinuousIntradayPrep(sessions=[SESSION], bar_minutes=BAR_MINUTES)
    df_prep, train_mask, target_cols = prep.prepare(
        df.copy(),
        steps_60m=TARGET_HORIZON_BARS,
        keep_only_active=False,
        add_daily=True,
        add_overnight=True,
        add_deseas=True,
        add_time_features=True,
        add_resample_precalc=False,
        apply_scaling=False,
        add_bid_ask="bidvol" in df.columns or "askvol" in df.columns,
        add_event_markers=False,
    )
    df_prep.columns = [c.lower() for c in df_prep.columns]
    tech_feature_cols = [c.lower() for c in prep.get_feature_cols(
        steps_60m=TARGET_HORIZON_BARS, bar_minutes=BAR_MINUTES,
        add_resample_precalc=False, add_bid_ask=False, add_event_markers=False,
    )]
    tech_feature_cols = [c for c in tech_feature_cols if c in df_prep.columns]

    # --- Storage features ---
    sw = storage_wkly.copy()
    sl = sw["storage_level"]
    sw["sl_4wk_mean"] = sl.rolling(4, min_periods=2).mean()
    sw["sl_4wk_max"] = sl.rolling(4, min_periods=2).max()
    sw["sl_4wk_min"] = sl.rolling(4, min_periods=2).min()
    sw["sl_change_4wk_mean"] = sw["storage_change"].rolling(4, min_periods=2).mean()
    storage_cols = ["storage_level", "storage_change",
                    "sl_4wk_mean", "sl_4wk_max", "sl_4wk_min", "sl_change_4wk_mean"]
    for extra in ["consensus_est", "surprise"]:
        if extra in sw.columns:
            storage_cols.append(extra)

    daily_idx = pd.date_range(sw.index[0], df_prep.index[-1].normalize(), freq="D")
    storage_daily = sw[storage_cols].reindex(sw.index.union(daily_idx)).sort_index().ffill()
    bar_dates = df_prep.index.normalize()
    for col in storage_cols:
        df_prep[col] = storage_daily[col].reindex(bar_dates).values
    storage_feature_cols = list(storage_cols)

    # --- Weather features ---
    weather_feature_cols = []
    if daily_weather is not None and not daily_weather.empty:
        dd = compute_degree_days(daily_weather)
        dd["HDD_7d"] = dd["HDD"].rolling(7, min_periods=3).sum()
        dd["CDD_7d"] = dd["CDD"].rolling(7, min_periods=3).sum()
        dd["HDD_7d_chg"] = dd["HDD_7d"] - dd["HDD_7d"].shift(7)
        dd["CDD_7d_chg"] = dd["CDD_7d"] - dd["CDD_7d"].shift(7)
        dd_cols = ["HDD", "CDD", "HDD_7d", "CDD_7d", "HDD_7d_chg", "CDD_7d_chg"]
        try:
            hdd_7d_series = dd["HDD_7d"].fillna(0)
            spline_df, _ = compute_spline_hdd_basis(hdd_7d_series, n_knots=4)
            for sc in spline_df.columns:
                dd[sc] = spline_df[sc].values
                dd_cols.append(sc)
        except Exception:
            pass
        if "wtd_TAVG" in daily_weather.columns:
            dd["wtd_tavg"] = daily_weather["wtd_TAVG"].values
            dd["wtd_tavg_7d"] = daily_weather["wtd_TAVG"].rolling(7, min_periods=3).mean().values
            dd_cols.extend(["wtd_tavg", "wtd_tavg_7d"])
        for col in dd_cols:
            renamed = f"dd_{col.lower()}" if not col.startswith("dd_") else col
            df_prep[renamed] = dd[col].reindex(bar_dates).values
            weather_feature_cols.append(renamed)

    # --- Regime features ---
    daily_close = df_prep.groupby(df_prep.index.date)["close"].last()
    daily_close.index = pd.DatetimeIndex(daily_close.index)
    daily_log_ret = np.log(daily_close / daily_close.shift(1))

    REGIME_COLS = [
        "regime_ret_1d", "regime_ret_5d", "regime_ret_21d",
        "regime_rv_5d", "regime_rv_21d",
        "regime_pct_in_5y_band", "regime_dev_5y_zscore", "regime_band_width_pct",
        "regime_fc_vs_seasonal_z", "regime_chg_vs_seasonal_z",
        "regime_is_injection", "regime_dev_x_season",
    ]

    regime_daily = pd.DataFrame(index=daily_close.index)
    regime_daily["regime_ret_1d"] = daily_log_ret
    regime_daily["regime_ret_5d"] = daily_log_ret.rolling(5).sum()
    regime_daily["regime_ret_21d"] = daily_log_ret.rolling(21).sum()
    regime_daily["regime_rv_5d"] = np.sqrt((daily_log_ret ** 2).rolling(5).mean()) * np.sqrt(252)
    regime_daily["regime_rv_21d"] = np.sqrt((daily_log_ret ** 2).rolling(21).mean()) * np.sqrt(252)

    sl_wk = storage_wkly["storage_level"]
    wk_idx = sl_wk.index.isocalendar().week.values
    hi_5y = pd.Series(np.nan, index=sl_wk.index)
    lo_5y = pd.Series(np.nan, index=sl_wk.index)
    mean_5y = pd.Series(np.nan, index=sl_wk.index)
    for w in range(1, 54):
        mask = wk_idx == w
        if mask.sum() < 2:
            continue
        idx_pos = np.where(mask)[0]
        vals = sl_wk.iloc[idx_pos]
        hi_5y.iloc[idx_pos] = vals.expanding().max().shift(1).values
        lo_5y.iloc[idx_pos] = vals.expanding().min().shift(1).values
        mean_5y.iloc[idx_pos] = vals.expanding().mean().shift(1).values
    hi_5y = hi_5y.ffill().bfill()
    lo_5y = lo_5y.ffill().bfill()
    mean_5y = mean_5y.ffill().bfill()
    band_w = (hi_5y - lo_5y).clip(lower=1)
    pct_band = (sl_wk - lo_5y) / band_w
    dev_zscore = (sl_wk - mean_5y) / band_w

    regime_wkly = pd.DataFrame({
        "regime_pct_in_5y_band": pct_band.values,
        "regime_dev_5y_zscore": dev_zscore.values,
        "regime_band_width_pct": (band_w / mean_5y.clip(lower=1)).values,
    }, index=sl_wk.index)
    regime_wkly_daily = regime_wkly.reindex(
        regime_wkly.index.union(daily_close.index)
    ).sort_index().ffill().reindex(daily_close.index)
    for col in regime_wkly_daily.columns:
        regime_daily[col] = regime_wkly_daily[col].values

    sc = storage_wkly["storage_change"]
    sea_chg = pd.Series(np.nan, index=sc.index)
    for w in range(1, 54):
        mask = wk_idx == w
        if mask.sum() < 2:
            continue
        idx_pos = np.where(mask)[0]
        sea_chg.iloc[idx_pos] = sc.iloc[idx_pos].expanding().mean().shift(1).values
    sea_chg = sea_chg.ffill().bfill()
    sea_std = (sc - sea_chg).expanding().std().clip(lower=1)
    chg_vs_sea = (sc - sea_chg) / sea_std
    fc = storage_wkly.get("consensus_est", sea_chg)
    fc_vs_sea = (fc - sea_chg) / sea_std

    fc_regime = pd.DataFrame({
        "regime_fc_vs_seasonal_z": fc_vs_sea.values,
        "regime_chg_vs_seasonal_z": chg_vs_sea.values,
    }, index=sc.index)
    fc_daily = fc_regime.reindex(
        fc_regime.index.union(daily_close.index)
    ).sort_index().ffill().reindex(daily_close.index)
    for col in fc_daily.columns:
        regime_daily[col] = fc_daily[col].values

    month = daily_close.index.month
    regime_daily["regime_is_injection"] = ((month >= 4) & (month <= 10)).astype(np.float32)
    season_sign = np.where(regime_daily["regime_is_injection"].values > 0.5, 1.0, -1.0)
    regime_daily["regime_dev_x_season"] = regime_daily["regime_dev_5y_zscore"] * season_sign

    for col in REGIME_COLS:
        df_prep[col] = regime_daily[col].reindex(bar_dates).values

    # --- Assemble features ---
    feature_groups = {}
    all_feature_cols = []
    feature_groups["technical"] = list(tech_feature_cols)
    all_feature_cols.extend(tech_feature_cols)
    feature_groups["storage"] = list(storage_feature_cols)
    all_feature_cols.extend(storage_feature_cols)
    if weather_feature_cols:
        feature_groups["weather"] = list(weather_feature_cols)
        all_feature_cols.extend(weather_feature_cols)

    FEATURE_GROUP_SIZES = {k: len(v) for k, v in feature_groups.items()}
    n_features = len(all_feature_cols)

    # --- Targets ---
    target_col = f"y_fwd_{TARGET_HORIZON_BARS}"
    if target_col not in df_prep.columns:
        df_prep[target_col] = np.log(
            df_prep["close"].shift(-TARGET_HORIZON_BARS) / df_prep["close"]
        )
    df_prep["target"] = df_prep[target_col]
    bar_ret = np.log(df_prep["close"] / df_prep["close"].shift(1))
    bars_per_day = int(df_prep.groupby(df_prep.index.date).size().median())
    df_prep["target_std"] = bar_ret.rolling(
        bars_per_day * 5, min_periods=bars_per_day
    ).std() * np.sqrt(TARGET_HORIZON_BARS)

    if N_CLASSES > 0:
        from CTAFlow.models.deep_learning.multi_branch.ng_moe_dataset import NGMoEDataBuilder
        df_prep["target_class"] = NGMoEDataBuilder._compute_target_classes(
            df_prep["target"], N_CLASSES, None,
        )

    keep_cols = all_feature_cols + REGIME_COLS + ["target", "target_std"]
    if "target_class" in df_prep.columns:
        keep_cols.append("target_class")
    keep_cols = [c for c in keep_cols if c in df_prep.columns]
    df_clean = df_prep.dropna(subset=keep_cols).copy()
    print(f"[DATA] Clean bars: {len(df_clean)} ({len(df_prep) - len(df_clean)} dropped)")

    # --- Datasets ---
    n_total = len(df_clean)
    n_tr = int(n_total * 0.70)
    n_va = int(n_total * 0.15)
    train_df = df_clean.iloc[:n_tr]
    val_df = df_clean.iloc[n_tr : n_tr + n_va]
    test_df = df_clean.iloc[n_tr + n_va :]

    ds_kw = dict(
        feature_cols=all_feature_cols, regime_cols=REGIME_COLS,
        seq_len=SEQ_LEN, ae_window_days=AE_WINDOW_DAYS,
        bars_per_day=bars_per_day, n_classes=N_CLASSES,
    )
    train_ds = IntradayHybridDataset(train_df, stride=TARGET_HORIZON_BARS, **ds_kw)
    val_ds = IntradayHybridDataset(val_df, stride=1, **ds_kw)
    test_ds = IntradayHybridDataset(test_df, stride=1, **ds_kw)
    ae_window_bars = AE_WINDOW_DAYS * bars_per_day

    print(f"[DATA] Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    return dict(
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        n_features=n_features, ae_window_bars=ae_window_bars,
        bars_per_day=bars_per_day, all_feature_cols=all_feature_cols,
        feature_groups=feature_groups, FEATURE_GROUP_SIZES=FEATURE_GROUP_SIZES,
        REGIME_COLS=REGIME_COLS, storage_wkly=storage_wkly,
    )


# ===================================================================
# Metrics & selection
# ===================================================================

def compute_val_metrics(model, val_loader, loss_fn, device, bars_per_day,
                        n_classes, feature_group_sizes):
    model.eval()
    all_losses = {k: [] for k in ["total_loss", "return_loss", "vol_loss",
                                   "mdn_nll_loss", "ce_loss", "positioning_loss",
                                   "vsn_entropy_loss"]}
    all_positions, all_returns, all_pred_returns = [], [], []
    all_pred_classes, all_true_classes = [], []
    n_total = 0

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 5:
                x_seq, ae_in, y_ret, y_std, y_cls = batch
                y_cls = y_cls.to(device)
            else:
                x_seq, ae_in, y_ret, y_std = batch
                y_cls = None
            x_seq, ae_in = x_seq.to(device), ae_in.to(device)
            y_ret, y_std = y_ret.to(device), y_std.to(device)
            out = model(x_seq, ae_in)
            losses = loss_fn(out, y_ret, y_std, y_cls)
            for k in all_losses:
                if k in losses:
                    v = losses[k]
                    all_losses[k].append((v.item() if torch.is_tensor(v) else v) * len(y_ret))
            n_total += len(y_ret)
            all_pred_returns.append(out["pred_return"].cpu().numpy())
            all_returns.append(y_ret.cpu().numpy())
            if "position" in out:
                all_positions.append(out["position"].cpu().numpy())
            if "class_logits" in out and y_cls is not None:
                all_pred_classes.append(out["class_logits"].argmax(dim=-1).cpu().numpy())
                all_true_classes.append(y_cls.cpu().numpy())

    metrics = {}
    for k, vals in all_losses.items():
        if vals:
            metrics[k] = sum(vals) / max(n_total, 1)
    metrics["val_loss"] = metrics.get("total_loss", 0)

    pred_ret = np.concatenate(all_pred_returns)
    actual_ret = np.concatenate(all_returns)
    metrics["return_mae"] = np.abs(pred_ret - actual_ret).mean()
    metrics["return_corr"] = float(np.corrcoef(pred_ret, actual_ret)[0, 1]) if len(pred_ret) > 2 else 0
    dir_mask = np.abs(actual_ret) > 1e-6
    metrics["direction_accuracy"] = float((np.sign(pred_ret[dir_mask]) == np.sign(actual_ret[dir_mask])).mean()) if dir_mask.any() else 0

    if all_pred_classes and all_true_classes:
        pc, ac = np.concatenate(all_pred_classes), np.concatenate(all_true_classes)
        metrics["accuracy"] = float((pc == ac).mean())
        for c in range(int(ac.max()) + 1):
            m = ac == c
            if m.sum() > 0:
                metrics[f"acc_class_{c}"] = float((pc[m] == c).mean())

    if all_positions:
        pos = np.concatenate(all_positions)
        sr = pos * actual_ret
        std = sr.std()
        ann = np.sqrt(252 * bars_per_day)
        metrics["sharpe"] = float(sr.mean() / std * ann) if std > 0 else 0
        ds = sr[sr < 0]
        metrics["sortino"] = float(sr.mean() / ds.std() * ann) if len(ds) > 1 and ds.std() > 0 else 0
        g, l = sr[sr > 0].sum(), abs(sr[sr < 0].sum())
        metrics["profit_factor"] = float(g / max(l, 1e-8))
        metrics["win_rate"] = float((sr > 0).mean())
        metrics["mean_abs_position"] = float(np.abs(pos).mean())
        cum = np.cumsum(sr)
        metrics["max_drawdown"] = float((cum - np.maximum.accumulate(cum)).min())

    return metrics


def selection_score(metrics, n_classes):
    sharpe = metrics.get("sharpe", 0)
    sortino = metrics.get("sortino", 0)
    pf = metrics.get("profit_factor", 1e-8)
    acc = metrics.get("accuracy", 0.25)
    return 0.25 * sharpe + 0.30 * sortino + 0.20 * math.log(max(pf, 1e-8)) + 0.25 * (acc - 0.25) * 10


# ===================================================================
# Optuna search
# ===================================================================

LOCKED = dict(
    d_latent=32, d_ae_hidden=256, tcn_width=64, tcn_depth=4,
    stride=2, mdn_hidden=128, head_hidden_dim=256,
    positioning_hidden_dim=64, batch_size=128,
)


def run_optuna(data, cfg, device):
    n_features = data["n_features"]
    ae_window_bars = data["ae_window_bars"]
    bars_per_day = data["bars_per_day"]
    FGSZ = data["FEATURE_GROUP_SIZES"]
    N_CLASSES = cfg["n_classes"]
    USE_VSN = True
    USE_POSITIONING = True
    SEQ_LEN = cfg["seq_len"]
    MAX_EPOCHS_OPT = cfg["max_epochs_opt"]
    OPT_PATIENCE = cfg["patience_opt"]

    trial_log = []

    def objective(trial):
        t0 = time.time()
        mdn_n_comp = trial.suggest_int("mdn_n_components", 3, 5)
        dropout = trial.suggest_float("dropout", 0.06, 0.20)
        vsn_d_model = trial.suggest_categorical("vsn_d_model", [16, 32])
        vsn_temperature = trial.suggest_float("vsn_temperature", 1.2, 2.5)
        vsn_entropy_weight = trial.suggest_float("vsn_entropy_weight", 0.02, 0.10, log=True)
        vsn_min_weight = trial.suggest_float("vsn_min_weight", 0.02, 0.08)
        ce_weight = trial.suggest_float("ce_weight", 0.7, 2.0, log=True)
        pnl_weight = trial.suggest_float("positioning_pnl_weight", 0.15, 0.8, log=True)
        nll_weight = trial.suggest_float("nll_weight", 0.04, 0.20, log=True)
        kl_weight = trial.suggest_float("kl_weight", 0.005, 0.04, log=True)
        recon_weight = trial.suggest_float("recon_weight", 0.05, 0.30, log=True)
        tc_cost = trial.suggest_float("tc_cost", 0.0005, 0.001)
        lr = trial.suggest_float("lr", 8e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 2e-3, log=True)

        config = HybridConfig(
            n_features=n_features, seq_len=SEQ_LEN,
            f_ae=12, ae_window=ae_window_bars,
            d_latent=LOCKED["d_latent"], d_ae_hidden=LOCKED["d_ae_hidden"],
            kl_weight=kl_weight, recon_weight=recon_weight,
            tcn_channels=[LOCKED["tcn_width"]] * LOCKED["tcn_depth"],
            tcn_kernel_size=3, stride=LOCKED["stride"],
            mdn_hidden_dims=[LOCKED["mdn_hidden"], LOCKED["mdn_hidden"] // 2],
            mdn_n_components=mdn_n_comp,
            head_hidden_dim=LOCKED["head_hidden_dim"], dropout=dropout,
            nll_weight=nll_weight, n_classes=N_CLASSES,
            use_positioning_head=USE_POSITIONING,
            positioning_hidden_dim=LOCKED["positioning_hidden_dim"],
            ce_weight=ce_weight, positioning_pnl_weight=pnl_weight, tc_cost=tc_cost,
            use_vsn=USE_VSN, vsn_d_model=vsn_d_model,
            vsn_temperature=vsn_temperature,
            vsn_entropy_weight=vsn_entropy_weight, vsn_min_weight=vsn_min_weight,
        )

        model = HybridMixtureNetwork(config, feature_group_sizes=FGSZ).to(device)
        loss_fn = HybridLoss(config)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
        bs = LOCKED["batch_size"]
        tl = DataLoader(data["train_ds"], batch_size=bs, shuffle=True, drop_last=True)
        vl = DataLoader(data["val_ds"], batch_size=bs, shuffle=False)

        best_score, patience_cnt = float("-inf"), 0

        for epoch in range(1, MAX_EPOCHS_OPT + 1):
            model.train()
            for batch in tl:
                if len(batch) == 5:
                    x_seq, ae_in, y_ret, y_std, y_cls = batch
                    y_cls = y_cls.to(device)
                else:
                    x_seq, ae_in, y_ret, y_std = batch
                    y_cls = None
                x_seq, ae_in = x_seq.to(device), ae_in.to(device)
                y_ret, y_std = y_ret.to(device), y_std.to(device)
                opt.zero_grad()
                out = model(x_seq, ae_in)
                losses = loss_fn(out, y_ret, y_std, y_cls)
                losses["total_loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            metrics = compute_val_metrics(model, vl, loss_fn, device, bars_per_day,
                                          N_CLASSES, FGSZ)
            score = selection_score(metrics, N_CLASSES)
            trial.report(score, epoch)

            if epoch % 10 == 0:
                print(f"  T{trial.number:02d} E{epoch:3d} "
                      f"score={score:.3f} best={best_score:.3f} ({time.time()-t0:.0f}s)")

            if trial.should_prune():
                trial_log.append({"trial": trial.number, "status": "PRUNED",
                                  "score": score, "time": time.time() - t0})
                raise optuna.TrialPruned()
            if score > best_score:
                best_score, patience_cnt = score, 0
            else:
                patience_cnt += 1
                if patience_cnt >= OPT_PATIENCE:
                    break

        trial_log.append({"trial": trial.number, "status": "COMPLETE",
                          "score": best_score, "time": time.time() - t0})
        return best_score

    study = optuna.create_study(
        study_name=f"ng_hybrid_{cfg['bar_minutes']}min",
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=False)

    sorted_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else float("-inf"),
        reverse=True,
    )
    rank = min(cfg["trial_rank"], len(sorted_trials) - 1)
    selected = sorted_trials[rank]

    print(f"\n[OPTUNA] Top-5 trials:")
    for i, t in enumerate(sorted_trials[:5]):
        marker = " <-- selected" if i == rank else ""
        print(f"  #{t.number}: {t.value:.4f}{marker}")

    return selected, trial_log


# ===================================================================
# Final training
# ===================================================================

def train_final(data, cfg, selected_trial, device):
    bp = selected_trial.params
    n_features = data["n_features"]
    ae_window_bars = data["ae_window_bars"]
    bars_per_day = data["bars_per_day"]
    FGSZ = data["FEATURE_GROUP_SIZES"]
    N_CLASSES = cfg["n_classes"]
    SEQ_LEN = cfg["seq_len"]

    final_cfg = HybridConfig(
        n_features=n_features, seq_len=SEQ_LEN,
        f_ae=12, ae_window=ae_window_bars,
        d_latent=LOCKED["d_latent"], d_ae_hidden=LOCKED["d_ae_hidden"],
        kl_weight=bp["kl_weight"], recon_weight=bp["recon_weight"],
        tcn_channels=[LOCKED["tcn_width"]] * LOCKED["tcn_depth"],
        tcn_kernel_size=3, stride=LOCKED["stride"],
        mdn_hidden_dims=[LOCKED["mdn_hidden"], LOCKED["mdn_hidden"] // 2],
        mdn_n_components=bp["mdn_n_components"],
        head_hidden_dim=LOCKED["head_hidden_dim"], dropout=bp["dropout"],
        nll_weight=bp["nll_weight"], n_classes=N_CLASSES,
        use_positioning_head=True,
        positioning_hidden_dim=LOCKED["positioning_hidden_dim"],
        ce_weight=bp["ce_weight"], positioning_pnl_weight=bp["positioning_pnl_weight"],
        tc_cost=bp["tc_cost"],
        use_vsn=True, vsn_d_model=bp.get("vsn_d_model", 16),
        vsn_temperature=bp.get("vsn_temperature", 1.8),
        vsn_entropy_weight=bp.get("vsn_entropy_weight", 0.05),
        vsn_min_weight=bp.get("vsn_min_weight", 0.05),
    )

    model = HybridMixtureNetwork(final_cfg, feature_group_sizes=FGSZ).to(device)
    loss_fn = HybridLoss(final_cfg)
    optimizer = optim.AdamW(model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    bs = LOCKED["batch_size"]
    train_loader = DataLoader(data["train_ds"], batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(data["val_ds"], batch_size=bs, shuffle=False)
    test_loader = DataLoader(data["test_ds"], batch_size=bs, shuffle=False)

    MAX_EPOCHS = cfg["max_epochs_final"]
    PATIENCE = cfg["patience_final"]
    best_val_loss, best_state, best_metrics = float("inf"), None, {}
    patience_cnt = 0

    loss_keys = ["total_loss", "return_loss", "vol_loss", "mdn_nll_loss",
                 "ae_recon_loss", "ae_kl_loss", "vsn_entropy_loss",
                 "ce_loss", "positioning_loss"]

    print(f"\n[TRAIN] Final training: {sum(p.numel() for p in model.parameters()):,} params")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            if len(batch) == 5:
                x_seq, ae_in, y_ret, y_std, y_cls = batch
                y_cls = y_cls.to(device)
            else:
                x_seq, ae_in, y_ret, y_std = batch
                y_cls = None
            x_seq, ae_in = x_seq.to(device), ae_in.to(device)
            y_ret, y_std = y_ret.to(device), y_std.to(device)
            optimizer.zero_grad()
            out = model(x_seq, ae_in)
            losses = loss_fn(out, y_ret, y_std, y_cls)
            losses["total_loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_metrics = compute_val_metrics(model, val_loader, loss_fn, device,
                                          bars_per_day, N_CLASSES, FGSZ)
        val_loss = val_metrics["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics.copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 20 == 0 or epoch == 1:
            score = selection_score(val_metrics, N_CLASSES)
            print(f"  E{epoch:3d} val_loss={val_loss:.5f} "
                  f"acc={val_metrics.get('accuracy', 0):.3f} "
                  f"sharpe={val_metrics.get('sharpe', 0):.3f} "
                  f"score={score:.3f} patience={patience_cnt}/{PATIENCE}")

        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # --- Test evaluation ---
    test_metrics = compute_val_metrics(model, test_loader, loss_fn, device,
                                       bars_per_day, N_CLASSES, FGSZ)
    test_score = selection_score(test_metrics, N_CLASSES)
    print(f"\n[TEST] score={test_score:.3f} sharpe={test_metrics.get('sharpe', 0):.3f} "
          f"acc={test_metrics.get('accuracy', 0):.3f} "
          f"sortino={test_metrics.get('sortino', 0):.3f}")

    return model, final_cfg, best_metrics, test_metrics


# ===================================================================
# Save artifacts
# ===================================================================

def save_artifacts(model, final_cfg, data, cfg, selected_trial, trial_log,
                   best_metrics, test_metrics):
    SAVE_DIR = cfg["save_dir"]
    bp = selected_trial.params

    # Merge locked + searched params
    all_params = dict(bp)
    all_params.update(LOCKED)
    all_params["best_value"] = selected_trial.value
    all_params["trial_number"] = selected_trial.number
    all_params["trial_rank"] = cfg["trial_rank"]
    all_params["n_classes"] = cfg["n_classes"]
    all_params["bar_minutes"] = cfg["bar_minutes"]
    all_params["target_horizon_bars"] = cfg["target_horizon"]
    all_params["session"] = f"USA_{cfg['session_start']}-{cfg['session_end']}"
    all_params["n_features"] = data["n_features"]
    all_params["ae_window_bars"] = data["ae_window_bars"]

    SESSION = SessionSpec("USA", cfg["session_start"], cfg["session_end"])

    # Model checkpoint
    model_path = SAVE_DIR / "ng_hybrid_intraday_best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(final_cfg),
        "best_params": all_params,
        "n_features": data["n_features"],
        "feature_cols": data["all_feature_cols"],
        "regime_cols": data["REGIME_COLS"],
        "feature_groups": {k: list(v) for k, v in data["feature_groups"].items()},
        "bar_minutes": cfg["bar_minutes"],
        "target_horizon_bars": cfg["target_horizon"],
        "session": {"name": SESSION.name, "start": SESSION.start, "end": SESSION.end},
        "ae_window_bars": data["ae_window_bars"],
    }, model_path)

    # Params JSON
    with open(SAVE_DIR / "best_params.json", "w") as f:
        json.dump(all_params, f, indent=2)

    # Trial log
    pd.DataFrame(trial_log).to_csv(SAVE_DIR / "trial_log.csv", index=False)

    # Test metrics
    with open(SAVE_DIR / "test_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    print(f"\n[SAVE] Artifacts saved to {SAVE_DIR}")
    return str(model_path), all_params


# ===================================================================
# RunPod handler
# ===================================================================

def handler(event):
    inp = event.get("input", {})
    SAVE_DIR = Path("/workspace/results/ng_hybrid_intraday")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        save_dir=SAVE_DIR,
        trial_rank=inp.get("trial_rank", 0),
        n_trials=inp.get("n_trials", 40),
        max_epochs_opt=inp.get("max_epochs_opt", 80),
        max_epochs_final=inp.get("max_epochs_final", 200),
        patience_opt=inp.get("patience_opt", 10),
        patience_final=inp.get("patience_final", 20),
        bar_minutes=inp.get("bar_minutes", 15),
        target_horizon=inp.get("target_horizon", 10),
        seq_len=inp.get("seq_len", 20),
        ae_window_days=inp.get("ae_window_days", 10),
        n_classes=inp.get("n_classes", 4),
        session_start=inp.get("session_start", "02:30"),
        session_end=inp.get("session_end", "15:00"),
    )

    device = _select_device()
    print(f"[INIT] Device={device} | Config={json.dumps({k: v for k, v in cfg.items() if k != 'save_dir'}, indent=2)}")
    t_start = time.time()

    # 1. Data pipeline
    data = load_and_prepare_data(cfg)

    # 2. Optuna search
    selected_trial, trial_log = run_optuna(data, cfg, device)

    # 3. Final training
    model, final_cfg, best_metrics, test_metrics = train_final(
        data, cfg, selected_trial, device
    )

    # 4. Save
    model_path, all_params = save_artifacts(
        model, final_cfg, data, cfg, selected_trial, trial_log,
        best_metrics, test_metrics,
    )

    elapsed = time.time() - t_start
    print(f"\n[DONE] Total time: {elapsed / 60:.1f} min")

    return {
        "model_path": model_path,
        "best_params": all_params,
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "test_score": round(selection_score(test_metrics, cfg["n_classes"]), 4),
        "val_score": round(selected_trial.value, 4),
        "n_trials_complete": len([t for t in trial_log if t["status"] == "COMPLETE"]),
        "elapsed_minutes": round(elapsed / 60, 1),
    }


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    # Local testing: python scripts/runpod_ng_hybrid_train.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--trial-rank", type=int, default=0)
    parser.add_argument("--max-epochs-opt", type=int, default=80)
    parser.add_argument("--max-epochs-final", type=int, default=200)
    args = parser.parse_args()

    result = handler({
        "input": {
            "n_trials": args.n_trials,
            "trial_rank": args.trial_rank,
            "max_epochs_opt": args.max_epochs_opt,
            "max_epochs_final": args.max_epochs_final,
        }
    })
    print(json.dumps(result, indent=2))
else:
    # RunPod serverless
    try:
        import runpod
        runpod.serverless.start({"handler": handler})
    except ImportError:
        pass
