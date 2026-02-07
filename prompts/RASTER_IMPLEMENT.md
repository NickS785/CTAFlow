Got it — you already **have** the spatial-sequential raster, so we shouldn’t build “number bars.” We just need to **encode your existing raster slices into tokens** and feed them (as a sequence) into **CMDMamba**, alongside the **patched 15-minute candle stream**.

I peeked at your upload `GC_rasterized.npz` and it’s exactly the kind of thing we want:

* `data`: **(N, 4, 32)** → **N 15-min steps**, **4 channels**, **32 price bins**
* `idx`: **(N,)** datetime64 timestamps

So each 15-minute index is a **32-bin spatial profile** with 4 channels. To make it “image-like” for a CNN across time, we stack a window of these into:

* `x_short`: **[B, T_short, C=4, H=32]**
* then view as **[B, C, H, W=T_short]** for a 2D CNN.

Below is a refactor that does exactly what you asked:

* Short stream: **32–128 15m raster indices** → CNN → **tokens [B, T_short, D]**
* Long stream: **256 15m candles** → patcher (`patch=2,stride=2`) → **tokens [B, 128, D]**
* Concatenate tokens → **CMDMamba**

---

## 1) Time/session feature builder (customizable)

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class SessionSpec:
    name: str
    start: str  # "HH:MM"
    end: str    # "HH:MM"

def _parse_time(hhmm: str):
    return pd.Timestamp("1970-01-01 " + hhmm).time()

def _time_mask(idx: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    s, e = _parse_time(start), _parse_time(end)
    t = idx.time
    return (t >= s) & (t <= e)

def make_time_features(
    idx: pd.DatetimeIndex,
    sessions=(SessionSpec("LONDON","02:00","11:00"), SessionSpec("USA","08:30","16:00")),
) -> np.ndarray:
    """
    Returns [T, F] float32 time features:
      - tod_sin/cos, dow_sin/cos, doy_sin/cos (6)
      - session onehot: london, usa, overlap (3)
    """
    secs = idx.hour*3600 + idx.minute*60 + idx.second
    tod = secs / (24*3600)
    dow = idx.dayofweek / 7.0
    doy = (idx.dayofyear - 1) / 365.25

    def sc(x):
        return np.sin(2*np.pi*x), np.cos(2*np.pi*x)

    tod_s, tod_c = sc(tod)
    dow_s, dow_c = sc(dow)
    doy_s, doy_c = sc(doy)

    lon = _time_mask(idx, sessions[0].start, sessions[0].end)
    usa = _time_mask(idx, sessions[1].start, sessions[1].end)
    overlap = lon & usa

    sess_oh = np.stack([lon, usa, overlap], axis=1).astype(np.float32)
    cyc = np.stack([tod_s,tod_c,dow_s,dow_c,doy_s,doy_c], axis=1).astype(np.float32)
    return np.concatenate([cyc, sess_oh], axis=1).astype(np.float32)
```

---

## 2) Dataset: returns short rasters + long candles + time features (+ mask)

This dataset assumes:

* `raster_npz` has `data` and `idx` like your file
* `candles_15m` is a NumPy array aligned to the same `idx` (or you pass a mapping)

```python
import torch
from torch.utils.data import Dataset

class RasterLongDataset(Dataset):
    """
    Short stream from raster (N, C, H) -> windows of length T_short (32..128)
    Long stream from candles (N, K) -> last 512 steps, then patched in model.

    Returns:
      x_short: [T_short, C, H]
      tfeat_short: [T_short, F]
      x_long: [512, K]
      tfeat_long: [512, F]
      short_mask: [T_short] (all True if fixed length; used if you later pad)
      anchor_idx: int (optional, for debugging)
    """
    def __init__(
        self,
        raster_npz_path: str,
        candles_15m: np.ndarray,          # shape [N, K], aligned to raster idx
        short_len_range=(32, 128),
        long_len=512,
        sessions=(SessionSpec("LONDON","02:00","11:00"), SessionSpec("USA","08:30","16:00")),
        require_session_active: bool = True,
        allow_overlap: bool = True,
    ):
        z = np.load(raster_npz_path, allow_pickle=True)
        self.r = z["data"].astype(np.float32)     # [N, C, H] (your file: [N,4,32])
        self.idx = pd.to_datetime(z["idx"])
        assert candles_15m.shape[0] == self.r.shape[0], "candles_15m must align with raster length N"
        self.candles = candles_15m.astype(np.float32)

        self.tfeat = make_time_features(self.idx, sessions=sessions).astype(np.float32)  # [N,F]
        self.short_lo, self.short_hi = short_len_range
        self.long_len = int(long_len)

        # session mask for sampling anchors
        lon = _time_mask(self.idx, sessions[0].start, sessions[0].end)
        usa = _time_mask(self.idx, sessions[1].start, sessions[1].end)
        active = lon | usa
        overlap = lon & usa

        if require_session_active:
            if allow_overlap:
                ok_session = active
            else:
                ok_session = active & (~overlap)
        else:
            ok_session = np.ones(len(self.idx), dtype=bool)

        # must have enough history for max(short) and long
        min_hist = max(self.short_hi, self.long_len)
        ok_hist = np.zeros(len(self.idx), dtype=bool)
        ok_hist[min_hist-1:] = True

        self.anchor_positions = np.flatnonzero(ok_session & ok_hist)
        if len(self.anchor_positions) == 0:
            raise ValueError("No valid anchors. Check session settings and history lengths.")

    def __len__(self):
        return len(self.anchor_positions)

    def __getitem__(self, i):
        anchor = int(self.anchor_positions[i])

        # choose short length per-sample (you can also fix this per dataloader/batch)
        T_short = np.random.randint(self.short_lo, self.short_hi + 1)

        s0 = anchor - T_short + 1
        l0 = anchor - self.long_len + 1

        x_short = self.r[s0:anchor+1]          # [T_short, C, H]
        t_short = self.tfeat[s0:anchor+1]      # [T_short, F]

        x_long = self.candles[l0:anchor+1]     # [512, K]
        t_long = self.tfeat[l0:anchor+1]       # [512, F]

        # mask (all ones here; used if you pad in collate)
        short_mask = np.ones((T_short,), dtype=np.float32)

        return (
            torch.from_numpy(x_short),
            torch.from_numpy(t_short),
            torch.from_numpy(x_long),
            torch.from_numpy(t_long),
            torch.from_numpy(short_mask),
            anchor,
        )

def collate_pad_short(batch):
    """
    Pads variable-length x_short to max_T in batch:
      x_short: [B, T, C, H]
      t_short: [B, T, F]
      short_mask: [B, T]
    Long stream is fixed (512) so no padding needed.
    """
    xS, tS, xL, tL, mS, anchor = zip(*batch)
    B = len(xS)
    maxT = max(x.shape[0] for x in xS)
    C = xS[0].shape[1]
    H = xS[0].shape[2]
    F = tS[0].shape[1]

    xS_pad = torch.zeros((B, maxT, C, H), dtype=torch.float32)
    tS_pad = torch.zeros((B, maxT, F), dtype=torch.float32)
    mS_pad = torch.zeros((B, maxT), dtype=torch.float32)

    for b in range(B):
        T = xS[b].shape[0]
        xS_pad[b, :T] = xS[b]
        tS_pad[b, :T] = tS[b]
        mS_pad[b, :T] = mS[b]

    xL = torch.stack(xL, dim=0)  # [B,512,K]
    tL = torch.stack(tL, dim=0)  # [B,512,F]
    anchor = torch.tensor(anchor, dtype=torch.long)

    return xS_pad, tS_pad, xL, tL, mS_pad, anchor
```

---

## 3) Short encoder: turns raster windows into per-step tokens for Mamba

This CNN preserves the **time axis** (width) and downsamples only the **price axis** (height), then produces **one embedding per 15-min step**:

```python
import torch.nn as nn
import torch

class RasterSeqToTokens(nn.Module):
    """
    x_short: [B, T, C, H]  -> tokens: [B, T, D]
    Treat as image: [B, C, H, W=T]
    """
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # downsample height only

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # downsample height only

            nn.Conv2d(128, d_model, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x_short: torch.Tensor) -> torch.Tensor:
        # [B,T,C,H] -> [B,C,H,T]
        B, T, C, H = x_short.shape
        x = x_short.permute(0, 2, 3, 1).contiguous()
        h = self.net(x)                 # [B, D, H', T]
        h = h.mean(dim=2)               # pool over price bins -> [B, D, T]
        tokens = h.permute(0, 2, 1)     # [B, T, D]
        return tokens
```

---

## 4) Long patcher: 256×15m candles → tokens with patch=2, stride=2

```python
class TimePatcher(nn.Module):
    def __init__(self, in_features: int, d_model: int, patch: int = 4, stride: int = 4):
        super().__init__()
        self.patch = patch
        self.stride = stride
        self.proj = nn.Sequential(
            nn.Linear(patch * in_features, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,512,K] -> [B,128,D]
        B, T, K = x.shape
        p = x.unfold(1, self.patch, self.stride).contiguous()     # [B,N,patch,K]
        p = p.view(B, p.shape[1], self.patch * K)                 # [B,N,patch*K]
        return self.proj(p)
```

---

## 5) Combine + CMDMamba

This assumes your `CMDMamba` takes `[B, T, D] -> [B, T, D]`.

```python
class MultiStreamCMDMamba(nn.Module):
    def __init__(self, candle_k: int, raster_c: int, raster_h: int, d_model: int, cmdmamba: nn.Module, time_feat_dim: int):
        super().__init__()
        self.cmdmamba = cmdmamba

        self.long_patcher = TimePatcher(candle_k, d_model, patch=4, stride=4)   # 512 -> 128 tokens
        self.short_encoder = RasterSeqToTokens(in_ch=raster_c, d_model=d_model)

        self.time_proj = nn.Sequential(nn.Linear(time_feat_dim, d_model), nn.LayerNorm(d_model))
        self.time_patcher = TimePatcher(time_feat_dim, d_model, patch=4, stride=4)

        self.seg = nn.Embedding(2, d_model)  # 0 long, 1 short

        # example head (you’ll swap to your multi-step 60m head)
        self.head = nn.Linear(d_model, 12)

    def forward(self, x_short, t_short, x_long, t_long, short_mask=None):
        # Short tokens [B,T_short,D]
        zS = self.short_encoder(x_short)
        zS = zS + self.time_proj(t_short)
        zS = zS + self.seg(torch.ones(zS.size(0), zS.size(1), dtype=torch.long, device=zS.device))

        # (optional) zero padded short positions
        if short_mask is not None:
            zS = zS * short_mask.unsqueeze(-1).to(zS.dtype)

        # Long tokens [B,128,D]
        zL = self.long_patcher(x_long)
        zL = zL + self.time_patcher(t_long)
        zL = zL + self.seg(torch.zeros(zL.size(0), zL.size(1), dtype=torch.long, device=zL.device))

        # Combine: long context then short detail
        z = torch.cat([zL, zS], dim=1)     # [B, 128 + T_short, D]

        h = self.cmdmamba(z)               # [B, 128 + T_short, D]

        # take the last valid short token as summary
        if short_mask is None:
            h_last = h[:, -1, :]
        else:
            offset = zL.size(1)
            lengths = short_mask.long().sum(dim=1).clamp(min=1)
            idx = offset + lengths - 1
            h_last = h[torch.arange(h.size(0), device=h.device), idx, :]

        y = self.head(h_last)              # e.g. 12-step next-60m targets
        return {"y": y, "h_last": h_last}
```

---

## How this matches your exact spec

* **Short stream**: `T_short ∈ [32,128]` 15-minute raster indices ✅
* **Long stream**: `256` 15-minute candles ✅
* **Patcher**: `patch=4,stride=4` → `512 → 128 tokens` ✅
* **Feed both into CMDMamba**: concatenate tokens and run once ✅
* **Customizable session/time features**: handled via `make_time_features` ✅


 **channel normalization**

---

## 1) Recommended normalization per channel

Assume `x` is the **raw** raster (float32):

### Channel 0: Density

* Often sparse-ish but nonnegative.
* Use **sqrt** or **log1p** to compress heavy tails.
* Then standardize (z-score).

**Good default:** `sqrt(density)` then `(x-μ)/σ`

### Channel 1: LogVolume

* Already log-scaled, typically closer to Gaussian.
* Just z-score.

### Channel 2: Imbalance

* Bounded roughly `[-1, 1]` (or slightly outside due to smoothing).
* Clip to `[-1, 1]`.
* Optionally apply `atanh` to “expand” the midrange:

  * `atanh(0.999 * imbalance)`
* Then z-score (optional). If you atanh, z-scoring helps.

**Good default:** `clip` + `atanh` + z-score

### Channel 3: Returns

* Signed, heavy tails.
* Either:

  1. **Vol-scale** returns using a rolling RV from your long 15m stream (best), then z-score; or
  2. robust clip to a quantile (e.g. ±5σ) and z-score.

**Good default:** `returns / (rv_15m + eps)`, clip, z-score

> Compute `μ, σ` **on training set only**, per-channel (optionally per-asset), and reuse for val/test/inference.

---

## 2) Drop-in PyTorch raster preprocessor

This runs inside your model (no leakage, no dataset-side complexity).

```python
import torch
import torch.nn as nn

class RasterPreprocess(nn.Module):
    """
    x: [B, T, C(=4), H]
    Applies per-channel transforms + (train-set) affine normalization.
    """
    def __init__(
        self,
        means: torch.Tensor,  # [4]
        stds: torch.Tensor,   # [4]
        eps: float = 1e-6,
        use_atanh_imbalance: bool = True,
        returns_scale: float = 1.0,  # if you already vol-scale elsewhere, keep 1.0
    ):
        super().__init__()
        self.register_buffer("means", means.float())
        self.register_buffer("stds", stds.float())
        self.eps = eps
        self.use_atanh_imbalance = use_atanh_imbalance
        self.returns_scale = returns_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H]
        d = x[:, :, 0]                       # Density
        lv = x[:, :, 1]                      # LogVolume
        imb = x[:, :, 2]                     # Imbalance
        ret = x[:, :, 3]                     # Returns

        # Density: sqrt compress (or log1p; choose one)
        d = torch.sqrt(torch.clamp(d, min=0.0))

        # LogVolume: as-is (already log)
        # (optionally clamp)
        # lv = torch.clamp(lv, -20, 20)

        # Imbalance: clip then (optional) atanh
        imb = torch.clamp(imb, -1.0, 1.0)
        if self.use_atanh_imbalance:
            imb = torch.atanh(0.999 * imb)

        # Returns: optional scale (e.g. vol-scaled upstream)
        ret = ret * self.returns_scale
        ret = torch.clamp(ret, -10.0, 10.0)  # robust guardrail

        y = torch.stack([d, lv, imb, ret], dim=2)  # back to [B,T,4,H]

        # z-score per channel (broadcast)
        y = (y - self.means[None, None, :, None]) / (self.stds[None, None, :, None] + self.eps)
        return y
```

**Where do `means/stds` come from?**
Compute them over the training set’s rasters **after** applying the same channel transforms (sqrt/atanh/etc.). Quick approach: sample a chunk of training windows and estimate means/stds per channel.

---

## 3) Encoder that produces one token per 15m step

This keeps **time resolution** (W axis) and pools only over **price bins**.

```python
class RasterSeqToTokens(nn.Module):
    """
    x_short: [B,T,4,H] -> tokens: [B,T,D]
    CNN over (price_bins x time) with time preserved.
    """
    def __init__(self, d_model: int, in_ch: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((2, 1)),  # downsample price only

            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((2, 1)),  # downsample price only

            nn.Conv2d(128, d_model, 1),
            nn.GELU(),
        )

    def forward(self, x_short: torch.Tensor) -> torch.Tensor:
        # [B,T,C,H] -> [B,C,H,T]
        x = x_short.permute(0, 2, 3, 1).contiguous()
        h = self.net(x)          # [B,D,H',T]
        h = h.mean(dim=2)        # pool over price bins -> [B,D,T]
        return h.permute(0, 2, 1)  # [B,T,D]
```

---

## 4) Putting it into your CMDMamba pipeline

* Long stream: `x_long: [B,512,K]` → patch(4,stride=4) → `[B,128,D]`
* Short stream: `x_short: [B,T_short,4,32]` → preprocess → CNN → `[B,T_short,D]`
* Concatenate → CMDMamba

Key point: if you pad `T_short` (32–128), multiply embeddings by a mask and select the **last valid** token for the head (as in the earlier code).

---

## 5) Practical defaults for GC/CL/Cattle with your 4 channels

Because your raster is already normalized-ish (Density, LogVolume, Imbalance, Returns), the only market-specific thing I’d vary is **Returns scaling**:

* For GC/CL: returns can be spikier intraday; consider dividing returns channel by a rolling 15m RV (computed from your long stream) before rasterization or at preprocessing time.
* For Cattle: thinner, more gaps → heavier clipping, and maybe reduce atanh aggressiveness.

-------------------------------------
Preprocess Raster
-------------------------------------
* `T_long = 256` (15-min candles)
* `patch = 2`, `stride = 2`  ⟹  tokens = ((256-2)/2 + 1 = 128)

That’s a clean “30-minute token” long stream, still giving your CMDMamba plenty of context while keeping token count fixed.

Below are the minimal changes to the earlier setup.

---

## 1) Dataset change: long_len = 256

```python
# before: long_len=512
ds = RasterLongDataset(
    raster_npz_path="GC_rasterized.npz",
    candles_15m=candles_15m,     # [N,K] aligned to raster idx
    short_len_range=(32, 128),
    long_len=256,               # <-- change
)
```

Your raster `.npz` format (`data`, `idx`) is already consistent with your extraction pipeline.

---

## 2) Patcher change: patch=2 stride=2 (gives 128 tokens from 256)

```python
class TimePatcher(nn.Module):
    def __init__(self, in_features: int, d_model: int, patch: int, stride: int):
        super().__init__()
        self.patch = patch
        self.stride = stride
        self.proj = nn.Sequential(
            nn.Linear(patch * in_features, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,K]
        B, T, K = x.shape
        p = x.unfold(1, self.patch, self.stride).contiguous()  # [B,N,patch,K]
        p = p.view(B, p.shape[1], self.patch * K)              # [B,N,patch*K]
        return self.proj(p)
```

Then instantiate the *long* patchers as:

```python
self.long_patcher = TimePatcher(candle_k, d_model, patch=2, stride=2)      # 256 -> 128 tokens
self.time_patcher = TimePatcher(time_feat_dim, d_model, patch=2, stride=2) # match
```

---

## 3) Full model snippet: only the long stream changes

```python
class MultiStreamCMDMamba(nn.Module):
    def __init__(self, candle_k: int, d_model: int, cmdmamba: nn.Module, time_feat_dim: int, raster_c: int = 4):
        super().__init__()
        self.cmdmamba = cmdmamba

        # LONG: 256 15m candles -> patch=2,stride=2 -> 128 tokens
        self.long_patcher = TimePatcher(candle_k, d_model, patch=2, stride=2)
        self.time_patcher = TimePatcher(time_feat_dim, d_model, patch=2, stride=2)

        # SHORT: raster window 32..128 steps stays the same
        self.short_encoder = RasterSeqToTokens(in_ch=raster_c, d_model=d_model)
        self.time_proj = nn.Sequential(nn.Linear(time_feat_dim, d_model), nn.LayerNorm(d_model))

        self.seg = nn.Embedding(2, d_model)  # 0=long, 1=short
        self.head = nn.Linear(d_model, 12)   # example

    def forward(self, x_short, t_short, x_long, t_long, short_mask=None):
        # short tokens
        zS = self.short_encoder(x_short) + self.time_proj(t_short)
        zS = zS + self.seg(torch.ones(zS.size(0), zS.size(1), dtype=torch.long, device=zS.device))
        if short_mask is not None:
            zS = zS * short_mask.unsqueeze(-1).to(zS.dtype)

        # long tokens (now 128 tokens from 256 bars)
        zL = self.long_patcher(x_long) + self.time_patcher(t_long)
        zL = zL + self.seg(torch.zeros(zL.size(0), zL.size(1), dtype=torch.long, device=zL.device))

        # combine and run CMDMamba
        z = torch.cat([zL, zS], dim=1)  # [B, 128 + T_short, D]
        h = self.cmdmamba(z)

        # last valid short token for prediction
        if short_mask is None:
            h_last = h[:, -1, :]
        else:
            offset = zL.size(1)  # 128
            lengths = short_mask.long().sum(dim=1).clamp(min=1)
            idx = offset + lengths - 1
            h_last = h[torch.arange(h.size(0), device=h.device), idx, :]

        return {"y": self.head(h_last), "h_last": h_last}
```

---

