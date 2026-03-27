from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
DETAILED_PATH = NOTEBOOKS / "mmtfv3_backtest_detailed.ipynb"
OPTUNA_PATH = NOTEBOOKS / "mmtfv3_cl_gc_optuna.ipynb"
OUTPUT_PATH = NOTEBOOKS / "mmtfv3_backtest_detailed_train_from_best.ipynb"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines(text),
    }


def clone_cell(nb: dict, idx: int, *, source: str | None = None) -> dict:
    cell = copy.deepcopy(nb["cells"][idx])
    if source is not None:
        cell["source"] = lines(source)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main() -> None:
    detailed = load_notebook(DETAILED_PATH)
    optuna = load_notebook(OPTUNA_PATH)

    config_source = """from pathlib import Path
import numpy as np

# -- Match the Optuna/training notebook prefix pattern --------------------------
TICKERS                 = ['CL', 'GC']
TUNE_BACKBONE           = False
BACKBONE                = 'transformer'
USE_PTP                 = True
USE_FUSED_SPATIAL       = True
USE_AMP                 = True
SPATIAL_ENCODER         = 'fused' if USE_FUSED_SPATIAL else 'separate'
SPATIAL_LOOKBACK_BARS   = 8
BAR_MINUTES             = 5
TARGET_HORIZON_MINUTES  = 60
SAMPLE_SESSION          = "overlap"
SAMPLE_SESSION_START    = "06:00"
SAMPLE_SESSION_END      = "13:00"
SAMPLE_STRIDE           = 12
AE_WINDOW               = 21
ARTIFACT_STEM           = '_'.join(TICKERS)

IN_COLAB = False
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT    = Path('/content/drive/MyDrive/features')
    RESULTS_PATH = Path('/content/drive/MyDrive/results/mmtfv3_cl_gc')
else:
    DATA_ROOT    = Path(r'F:\\Upload\\s3\\model_data')
    RESULTS_PATH = Path(r'F:\\Upload\\s3\\results')

BACKTEST_PATH = RESULTS_PATH
BACKTEST_PATH.mkdir(parents=True, exist_ok=True)

bb_tag  = "tuned" if TUNE_BACKBONE else BACKBONE
ptp_tag = "_ptp" if USE_PTP else ""
prefix  = f"{ARTIFACT_STEM}_mmtfv3_{bb_tag}{ptp_tag}_optuna"

params_path         = RESULTS_PATH / f"{prefix}_best_params.json"
study_path          = RESULTS_PATH / f"{prefix}_study.pkl"
retrained_model_path = RESULTS_PATH / f"{prefix}_best_model_retrained.pth"
training_history_path = RESULTS_PATH / f"{prefix}_retrained_training_history.csv"

required_files = [params_path]
required_files.extend(DATA_ROOT / ticker / 'intraday.csv' for ticker in TICKERS)
required_files.extend(DATA_ROOT / ticker / f'{ticker}_numbars.npz' for ticker in TICKERS)
required_files.extend(DATA_ROOT / ticker / 'vpin.parquet' for ticker in TICKERS)
missing_required_files = [path for path in required_files if not path.exists()]

optional_files = [study_path]
optional_files.extend(DATA_ROOT / ticker / 'rasterized.npz' for ticker in TICKERS)
optional_files.extend(DATA_ROOT / ticker / 'profiles.npz' for ticker in TICKERS)
missing_optional_files = [path for path in optional_files if not path.exists()]

BACKTEST_BATCH_SIZE = 256
TRADE_THRESH        = 0.05
TC_COST_BPS         = 0.5
TC_COST             = TC_COST_BPS * 1e-4

BARS_PER_DAY   = int(450 / TARGET_HORIZON_MINUTES)
BARS_PER_YEAR  = 252 * BARS_PER_DAY
ANNUALIZATION  = np.sqrt(BARS_PER_YEAR)

print(f"artifact_stem       : {ARTIFACT_STEM}")
print(f"prefix              : {prefix}")
print(f"RESULTS_PATH        : {RESULTS_PATH}")
print(f"DATA_ROOT           : {DATA_ROOT}")
print(f"params_path         : {params_path}")
print(f"study_path          : {study_path}")
print(f"retrained_model_path: {retrained_model_path}")
print(f"SPATIAL_ENCODER     : {SPATIAL_ENCODER}")
print(f"BARS_PER_DAY        : {BARS_PER_DAY}   ANNUALIZATION: {ANNUALIZATION:.2f}")

if missing_required_files:
    print("\\nMissing required files:")
    for path in missing_required_files:
        print(f"  - {path}")
else:
    print("\\nAll required files are present.")

if missing_optional_files:
    print("\\nMissing optional files:")
    for path in missing_optional_files:
        print(f"  - {path}")
"""

    imports_source = """import json
import math
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from CTAFlow.data.datasets.v3_continuous import (
    V3ContinuousPrep,
    V3ContinuousDataset,
    loaders_from_samples,
    unpack_v3_batch,
    v3_collate_fn,
)
from CTAFlow.models.prep.intraday_continuous import SessionSpec
from CTAFlow.models.deep_learning.multi_branch.tft.mmtf_v3_core import (
    ContinuousTradingLoss,
    HeadAwarePTPScheduler,
    MMTFv3Core,
    PTPLoss,
    SharpeScheduler,
    StatefulMMTFv3Core,
    build_ptp_optimizer_param_groups,
    evaluate_v3_ptp,
    evaluate_v3_stateful,
    print_v3_diagnostics,
    returns_to_classes,
    train_epoch_v3_ptp,
    train_epoch_v3_stateful,
)

print(f"PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
"""

    load_params_source = """if missing_required_files:
    missing_text = '\\n'.join(f"  - {path}" for path in missing_required_files)
    raise FileNotFoundError(
        "Add the missing required files before running the notebook:\\n"
        f"{missing_text}"
    )

with open(params_path) as f:
    best = json.load(f)

print("Best hyperparameters loaded:")
for k, v in sorted(best.items()):
    print(f"  {k:35s}: {v}")

study = None
if study_path.exists():
    study = joblib.load(study_path)
    print(f"\\nOptuna study loaded  - {len(study.trials)} trials")
    print(f"  Best value : {study.best_value:.6f}")
    print(f"  Best trial : #{study.best_trial.number}")
else:
    print("\\nNo Optuna study found; training from params JSON only.")
"""

    prep_source = """session_specs = []
if SAMPLE_SESSION_START and SAMPLE_SESSION_END:
    session_specs = [SessionSpec("custom", SAMPLE_SESSION_START, SAMPLE_SESSION_END)]
else:
    session_specs = [SessionSpec("USA", "08:30", "16:00")]

print("Loading ticker data...")
prep = V3ContinuousPrep.from_directories(
    root_dir=DATA_ROOT,
    tickers=TICKERS,
    sessions=session_specs,
    bar_minutes=BAR_MINUTES,
    target_horizon_minutes=TARGET_HORIZON_MINUTES,
    ae_window=AE_WINDOW,
)

dims = prep.get_dims()
print(f"\\nFeature dimensions: {dims}")
for ticker in TICKERS:
    nb_ts, nb_vals = prep._numbars_ts.get(
        ticker,
        (np.array([], dtype='datetime64[ns]'), np.empty((0, 4, 32), dtype=np.float32)),
    )
    print(f"{ticker}: number bars loaded={len(nb_ts):,}, number bar tensor shape={nb_vals.shape}")
    assert len(nb_ts) > 0, f"{ticker} missing timestamped NumberBars ({ticker}_numbars.npz)"

print(f"Tech feature columns ({dims['f_tech']}): {prep._tech_feature_cols[:10]}...")
print(f"n_tickers: {prep.n_tickers}")
print(f"n_asset_classes: {prep.n_asset_classes}")
print(f"n_asset_subclasses: {prep.n_asset_subclasses}")

MAX_TECH_LOOKBACK = int(max(best.get('tech_lookback', 128), 128))
MAX_SEQ_LOOKBACK = int(max(best.get('seq_lookback', 64), 64))
MAX_NUMBARS_LOOKBACK = int(max(best.get('numbars_lookback', SPATIAL_LOOKBACK_BARS), 24))

print(f"\\nPre-building samples (tech={MAX_TECH_LOOKBACK}, seq={MAX_SEQ_LOOKBACK}, nb={MAX_NUMBARS_LOOKBACK})...")
all_cached_samples = prep.build_samples(
    tech_lookback=MAX_TECH_LOOKBACK,
    seq_lookback_bars=MAX_SEQ_LOOKBACK,
    numbars_lookback=MAX_NUMBARS_LOOKBACK,
    use_fused_spatial=USE_FUSED_SPATIAL,
    session_only=True,
    sample_session=SAMPLE_SESSION,
    sample_session_start=SAMPLE_SESSION_START,
    sample_session_end=SAMPLE_SESSION_END,
    stride=SAMPLE_STRIDE,
)
assert all_cached_samples, "No valid samples produced. Check diagnostics above."

all_dates = sorted(set(s["date"] for s in all_cached_samples))
n_val = max(1, int(len(all_dates) * 0.2))
VAL_CUTOFF = all_dates[-n_val]
TRAIN_SAMPLES = [s for s in all_cached_samples if s["date"] < VAL_CUTOFF]
VAL_SAMPLES   = [s for s in all_cached_samples if s["date"] >= VAL_CUTOFF]

print(f"Cached samples: {len(TRAIN_SAMPLES)} train, {len(VAL_SAMPLES)} val (cutoff={VAL_CUTOFF}, {len(all_dates)} total days)")

F_TECH = dims['f_tech']
F_SEQ = dims['f_seq']
F_AE = 4
NUMBARS_CHANNELS = dims['numbars_channels']
FUSED_SPATIAL_CHANNELS = dims['fused_spatial_channels']
FUSED_SPATIAL_BINS = dims['fused_spatial_bins']
VPIN_TIME = dims['vpin_time']
VPIN_CHANNELS = dims['vpin_channels']
VPIN_BINS = dims['vpin_bins']

print("Model input dimensions:")
print(f"  f_tech={F_TECH}")
print(f"  f_seq={F_SEQ}")
print(f"  f_ae={F_AE}")
print(f"  fused_spatial: channels={FUSED_SPATIAL_CHANNELS}, bins={FUSED_SPATIAL_BINS}")
print(f"  vpin_raster: time={VPIN_TIME}, channels={VPIN_CHANNELS}, bins={VPIN_BINS}")
"""

    train_header_source = """best_params = best
print("Training final model from best params JSON:")
for k, v in sorted(best_params.items()):
    if k not in ('best_value', 'tickers', 'target_horizon_minutes', 'backbone'):
        print(f"  {k}: {v}")
"""

    train_build_source = """tech_lookback = int(best_params['tech_lookback'])
seq_lookback = int(best_params['seq_lookback'])
numbars_lookback = int(best_params.get('numbars_lookback', SPATIAL_LOOKBACK_BARS))

train_loader, val_loader = loaders_from_samples(
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    prep,
    batch_size=int(best_params['batch_size']),
    use_fused_spatial=USE_FUSED_SPATIAL,
    tech_lookback=tech_lookback,
)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

backbone = best_params.get('backbone', BACKBONE)
print(f"\\nUsing {backbone.upper()} backbone")
print(f"Spatial path: {SPATIAL_ENCODER}")
print(f"Spatial lookback: {numbars_lookback} bars")

base_model = MMTFv3Core(
    f_tech=F_TECH,
    f_seq=F_SEQ,
    f_ae=F_AE,
    ae_type=best_params.get('ae_type', 'vae'),
    d_latent=int(best_params['d_latent']),
    d_ae_hidden=int(best_params['d_ae_hidden']),
    kl_weight=float(best_params['kl_weight']),
    recon_weight=float(best_params['recon_weight']),
    n_tickers=prep.n_tickers,
    n_asset_classes=prep.n_asset_classes,
    n_asset_subclasses=prep.n_asset_subclasses,
    d_model=int(best_params['d_model']),
    d_static_emb=int(best_params['d_static_emb']),
    backbone=backbone,
    n_heads=int(best_params['n_heads']),
    n_layers=int(best_params['n_layers']),
    d_ff=int(best_params.get('d_ff', 512)),
    d_state=int(best_params.get('d_state', 16)),
    d_conv=int(best_params.get('d_conv', 4)),
    expand=int(best_params.get('expand', 2)),
    dropout=float(best_params['dropout']),
    grn_dropout=float(best_params['grn_dropout']),
    spatial_encoder=SPATIAL_ENCODER,
    numbars_channels=NUMBARS_CHANNELS,
    vpin_channels=VPIN_CHANNELS,
    vpin_bins=VPIN_BINS,
    vpin_time=VPIN_TIME,
    seq_layers=int(best_params.get('seq_layers', 2)),
    seq_nheads=int(best_params.get('seq_nheads', 4)),
)

final_model = StatefulMMTFv3Core(
    base_model=base_model,
    n_tickers=prep.n_tickers,
    quantile_head=USE_PTP,
    ptp_temperature=float(best_params.get('ptp_temperature', 1.5)),
    state_hidden_dim=int(best_params.get('state_hidden_dim', 16)),
    state_momentum=float(best_params.get('state_momentum', 0.9)),
    update_on_eval=True,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in final_model.parameters()):,}")
"""

    save_model_source = """if best_state:
    final_model.load_state_dict(best_state)
    if hasattr(final_model, 'reset_position_state'):
        final_model.reset_position_state()
    print("Loaded best in-memory state")

model = final_model

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "best_params": best_params,
        "dims": dims,
        "tickers": TICKERS,
        "val_cutoff": str(VAL_CUTOFF),
        "training_history_path": str(training_history_path),
    },
    retrained_model_path,
)
print(f"Retrained model saved -> {retrained_model_path}")
"""

    backtest_build_source = """tech_lookback    = int(best_params['tech_lookback'])
seq_lookback     = int(best_params['seq_lookback'])
numbars_lookback = int(best_params.get('numbars_lookback', SPATIAL_LOOKBACK_BARS))

VAL_START_DATE = min(s['date'] for s in VAL_SAMPLES)
oos_samples = list(VAL_SAMPLES)
print(f"OOS samples       : {len(oos_samples):,}  ({VAL_START_DATE} -> {max(s['date'] for s in oos_samples)})")

sample_dates   = [s['date'] for s in oos_samples]
sample_tickers = [s['ticker'] for s in oos_samples]

for tk in TICKERS:
    n_tk = sum(1 for t in sample_tickers if t == tk)
    print(f"  {tk}: {n_tk:,} OOS samples")

dataset = V3ContinuousDataset(
    oos_samples,
    fused_tail_shape=(FUSED_SPATIAL_CHANNELS, FUSED_SPATIAL_BINS),
)
full_loader = DataLoader(
    dataset,
    batch_size=BACKTEST_BATCH_SIZE,
    shuffle=False,
    collate_fn=v3_collate_fn,
    num_workers=0,
)
print(f"DataLoader  : {len(full_loader)} batches of up to {BACKTEST_BATCH_SIZE}")
"""

    ticker_breakdown_source = """ticker_rows = []
for tk in sorted(bt['ticker'].unique()):
    df_tk = bt[bt['ticker'] == tk].copy()
    stats_gross = _metrics(df_tk, tk, col='strategy_ret')
    stats_tc = _metrics(df_tk, tk, col='strategy_ret_tc')
    ticker_rows.append(
        {
            'Ticker': tk,
            'Samples': len(df_tk),
            'Start': df_tk['date'].min().date(),
            'End': df_tk['date'].max().date(),
            'Gross PnL': round(df_tk['strategy_ret'].sum(), 6),
            'TC PnL': round(df_tk['strategy_ret_tc'].sum(), 6),
            'Gross Sharpe': stats_gross['Sharpe (ann)'],
            'TC Sharpe': stats_tc['Sharpe (ann)'],
            'Win Rate (%)': stats_tc['Win Rate (%)'],
            'Dir Acc (%)': stats_tc['Dir Acc (%)'],
            '# Trades': stats_tc['# Trades'],
            'Avg |Position|': stats_tc['Avg |Position|'],
            'Avg Active Position': round(df_tk.loc[df_tk['active'], 'position'].abs().mean(), 4) if df_tk['active'].any() else 0.0,
        }
    )

df_ticker_breakdown = pd.DataFrame(ticker_rows).set_index('Ticker')
print("\\nTICKER BREAKDOWN (OOS)")
print("=" * 100)
print(df_ticker_breakdown.to_string())
"""

    trade_source = """def extract_trades(df_ticker):
    \"\"\"Extract trade records from a single-ticker DataFrame ordered by time.\"\"\"
    df = df_ticker.sort_values('date').reset_index(drop=True)
    pos = df['position'].values
    ret = df['strategy_ret_tc'].values
    active = np.abs(pos) > TRADE_THRESH

    trades = []
    in_trade = False
    t_sign = 0
    t_ret = []

    for i in range(len(pos)):
        sign_i = int(np.sign(pos[i]))
        if active[i]:
            if not in_trade or sign_i != t_sign:
                if in_trade and t_ret:
                    trades.append({'sign': t_sign, 'n_bars': len(t_ret), 'gross_ret': sum(t_ret)})
                in_trade = True
                t_sign = sign_i
                t_ret = [ret[i]]
            else:
                t_ret.append(ret[i])
        else:
            if in_trade and t_ret:
                trades.append({'sign': t_sign, 'n_bars': len(t_ret), 'gross_ret': sum(t_ret)})
            in_trade = False
            t_ret = []
            t_sign = 0

    if in_trade and t_ret:
        trades.append({'sign': t_sign, 'n_bars': len(t_ret), 'gross_ret': sum(t_ret)})

    return pd.DataFrame(trades) if trades else pd.DataFrame(columns=['sign', 'n_bars', 'gross_ret'])

trade_rows = []
trade_detail_rows = []
n_tickers = len(TICKERS)
fig, axes = plt.subplots(2, max(n_tickers, 1), figsize=(8 * max(n_tickers, 1), 10), squeeze=False)

for col_idx, tk in enumerate(sorted(id_to_ticker.values())):
    df_tk = bt[bt['ticker'] == tk]
    trades = extract_trades(df_tk)
    if len(trades) == 0:
        trade_rows.append({'Ticker': tk, '# Trades': 0})
        continue

    trades = trades.copy()
    trades['Ticker'] = tk
    trades['trade_id'] = np.arange(1, len(trades) + 1)
    trade_detail_rows.append(trades)

    wins = trades['gross_ret'] > 0
    longs = trades['sign'] == 1
    row = {
        'Ticker': tk,
        '# Trades': len(trades),
        '# Long': int(longs.sum()),
        '# Short': int((~longs).sum()),
        'Avg Profit/Trade': round(trades['gross_ret'].mean(), 6),
        'Long Avg Profit': round(trades.loc[longs, 'gross_ret'].mean(), 6) if longs.any() else 0,
        'Short Avg Profit': round(trades.loc[~longs, 'gross_ret'].mean(), 6) if (~longs).any() else 0,
        'Win Rate (%)': round(wins.mean() * 100, 2),
        'Long Win (%)': round(wins[longs].mean() * 100, 2) if longs.any() else 0,
        'Short Win (%)': round(wins[~longs].mean() * 100, 2) if (~longs).any() else 0,
        'Avg Hold (bars)': round(trades['n_bars'].mean(), 1),
        'Best Trade': round(trades['gross_ret'].max(), 6),
        'Worst Trade': round(trades['gross_ret'].min(), 6),
    }
    trade_rows.append(row)

    ax0 = axes[0, col_idx]
    ax1 = axes[1, col_idx]
    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in trades['gross_ret']]
    ax0.bar(range(len(trades)), trades['gross_ret'].values, color=colors, alpha=0.75, width=1.0)
    ax0.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax0.set_title(f'{tk} OOS — Trade PnL (TC-adj)')
    ax0.set_xlabel('Trade #')
    ax0.set_ylabel('Gross Return')
    ax0.grid(True, alpha=0.25, axis='y')

    ax1.hist(trades['n_bars'].values, bins=30, color='steelblue', alpha=0.8, edgecolor='white')
    ax1.axvline(trades['n_bars'].mean(), color='red', lw=1.5, linestyle='--', label=f"mean={trades['n_bars'].mean():.1f}")
    ax1.set_title(f'{tk} OOS — Holding Duration (bars)')
    ax1.set_xlabel('Bars Held')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.25)

df_trades = pd.DataFrame(trade_rows).set_index('Ticker')
if trade_detail_rows:
    df_trade_details = pd.concat(trade_detail_rows, ignore_index=True)
else:
    df_trade_details = pd.DataFrame(columns=['Ticker', 'trade_id', 'sign', 'n_bars', 'gross_ret'])

print("\\nTRADE-LEVEL ANALYSIS (OOS)")
print("=" * 90)
print(df_trades.to_string())

plt.suptitle('Trade-Level Analysis (OOS, TC-adjusted)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(BACKTEST_PATH / f"{prefix}_bt_trades.png", dpi=150, bbox_inches='tight')
plt.show()
"""

    export_source = """bt_export_path = BACKTEST_PATH / f"{prefix}_bt_full.csv"
bt.to_csv(bt_export_path, index=False)
print(f"Full backtest saved -> {bt_export_path}")

summary_path = BACKTEST_PATH / f"{prefix}_bt_summary.csv"
df_summary.to_csv(summary_path)
print(f"Summary saved       -> {summary_path}")

ticker_breakdown_path = BACKTEST_PATH / f"{prefix}_bt_ticker_breakdown.csv"
df_ticker_breakdown.to_csv(ticker_breakdown_path)
print(f"Ticker breakdown    -> {ticker_breakdown_path}")

trades_path = BACKTEST_PATH / f"{prefix}_bt_trades.csv"
df_trades.to_csv(trades_path)
print(f"Trade table saved   -> {trades_path}")

trade_detail_path = BACKTEST_PATH / f"{prefix}_bt_trade_details.csv"
df_trade_details.to_csv(trade_detail_path, index=False)
print(f"Trade details saved -> {trade_detail_path}")

history_df = pd.DataFrame(history)
history_df.to_csv(training_history_path, index=False)
print(f"Training history    -> {training_history_path}")

print("\\n" + "=" * 70)
print(f"RETRAIN + BACKTEST COMPLETE — OOS from {VAL_START_DATE}")
print("=" * 70)
for col_label, col in [('Gross', 'strategy_ret'), ('TC-adj', 'strategy_ret_tc')]:
    sr = bt[col].values
    ann_sh = (sr.mean() / (sr.std() + 1e-8)) * ANNUALIZATION
    print(
        f"  {col_label:8s}: Ann.Sharpe={ann_sh:.3f}   "
        f"Net PnL={sr.sum():.5f}   "
        f"Max DD={_metrics(bt, '', col=col)['Max Drawdown']:.5f}"
    )
"""

    cells = [
        markdown_cell(
            "# MMTFv3 Detailed Backtest — Train From Best Params\n\n"
            "This notebook follows the detailed backtest workflow but **does not load an old checkpoint**. "
            "It loads the Optuna `{prefix}_best_params.json`, rebuilds the model with the **current feature dimensions**, "
            "trains from scratch, and then runs the detailed OOS backtest with explicit ticker and trade breakdowns."
        ),
        markdown_cell("## 1. Config"),
        code_cell(config_source),
        markdown_cell("## 2. Imports"),
        code_cell(imports_source),
        markdown_cell("## 3. Load Best Params"),
        code_cell(load_params_source),
        markdown_cell("## 4. Load Data & Build Sample Cache"),
        code_cell(prep_source),
        markdown_cell("## 5. Train Final Model From Scratch"),
        code_cell(train_header_source),
        code_cell(train_build_source),
        clone_cell(optuna, 27),
        clone_cell(optuna, 28),
        code_cell(optuna["cells"][29]["source"] if isinstance(optuna["cells"][29]["source"], str) else "".join(optuna["cells"][29]["source"])),
        code_cell(save_model_source),
        clone_cell(optuna, 31),
        markdown_cell("## 6. Model Diagnostics"),
        clone_cell(optuna, 33),
        clone_cell(optuna, 34),
        markdown_cell("## 7. Build Date-Aware OOS Backtest"),
        code_cell(backtest_build_source),
        markdown_cell("## 8. Inference Pass"),
        clone_cell(detailed, 12),
        markdown_cell("## 9. Backtest DataFrame"),
        clone_cell(detailed, 14),
        markdown_cell("## 10. Portfolio-Level Metrics"),
        clone_cell(detailed, 16),
        markdown_cell("## 11. Ticker Breakdown"),
        code_cell(ticker_breakdown_source),
        markdown_cell("## 12. Position (Weight) Evolution Over Time"),
        clone_cell(detailed, 18),
        markdown_cell("## 13. Calibration — Position vs Realized Return Quantile"),
        clone_cell(detailed, 20),
        markdown_cell("## 14. Position–Return Scatter"),
        clone_cell(detailed, 22),
        markdown_cell("## 15. Trade-Level Analysis"),
        code_cell(trade_source),
        markdown_cell("## 16. Cumulative PnL & Rolling Sharpe"),
        clone_cell(detailed, 26),
        markdown_cell("## 17. Drawdown Analysis"),
        clone_cell(detailed, 28),
        markdown_cell("## 18. Monthly PnL Calendar Heatmap"),
        clone_cell(detailed, 30),
        markdown_cell("## 19. Quantile Head — Class Logit Distribution"),
        clone_cell(detailed, 32),
        markdown_cell("## 20. Branch Attribution & Leakage Detection"),
        clone_cell(detailed, 34),
        clone_cell(detailed, 35),
        clone_cell(detailed, 36),
        markdown_cell("## 21. Export Results"),
        code_cell(export_source),
    ]

    out_nb = copy.deepcopy(detailed)
    out_nb["cells"] = cells
    OUTPUT_PATH.write_text(json.dumps(out_nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
