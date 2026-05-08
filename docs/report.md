# Quartz Solar Forecast - New Model Report

Issue #30: Building a new model that outperforms existing v1(GB) and v2(XGBoost) metrics.

## 1. Data Preparation

### 1.1 NWP Evaluation Cache
- Source: HuggingFace `openclimatefix/dwd-icon-eu` (ICON model)
- Downloaded 2,457 zarr files to `data/nwp/`
- Each file: `{YYYYMMDD_HH}_lat={lat}_lon={lon}.zarr`, ~38KB
- 9 variables (t_2m, tot_prec, clch, clcm, clcl, u, v, aswdir_s, aswdifd_s) x 54 forecast steps
- Validation: all 2,457 files passed integrity check (metadata, shape, chunk existence)
- Corrupted source: `20210510_12.zarr.zip` on HuggingFace (BadZipFile: Bad CRC-32)
  - 3 affected samples removed from testset (pv_id=7721, 7241, 3489)
  - **Testset: 2,500 -> 2,497 samples**

### 1.2 Training Data
- Prepared by `scripts/acquire_training_data.py`, period: 2018-2020
- Source: HuggingFace `openclimatefix/uk_pv` (30min) + Open-Meteo Archive API (hourly NWP)

| File | Size | Shape | Description |
|------|------|-------|-------------|
| `pv_30min_2018_2020.parquet` | 1.4GB | 739M rows x 3 cols | pv_id, timestamp, power_kw (30min, 23,623 sites) |
| `nwp_hourly_2018_2020.parquet` | 3.7GB | 185M rows x 17 cols | 14 NWP variables + time/lat/lon (1h, 7,032 locations) |
| `site_metadata.parquet` | 420KB | 30,757 rows x 8 cols | pv_id, lat/lon, orientation, tilt, kWp |
| `bad_data.parquet` | 8KB | Bad data periods to exclude |
| `test_ground_truth.parquet` | 352KB | Ground truth for testset |

**Data cleaning steps applied:**
1. Remove known bad periods from `bad_data.csv` (per-month)
2. Remove rows with NaN `generation_Wh` (per-month)
3. Remove entire days with any negative `generation_Wh`
4. Remove entire days with any `generation_Wh > kWp x 750`
5. Remove entire days with nonzero generation at night
6. Remove entire days with zero generation during all daytime hours

## 2. Baseline Evaluation

Evaluated on testset (2,497 samples, 50 sites x ~50 timestamps, 2021).
Ground truth: HuggingFace `openclimatefix/uk_pv` 5-minutely data, converted to kW.

### 2.1 Overall MAE

| | v1 (model-0.3.0) | v1-tilt (model-0.4.0) | README reference |
|---|---|---|---|
| MAE (kW) | **0.1656** | 0.194 | 0.1906 |
| MAE (%) | **5.08%** | 5.89% | - |
| MAE excl. night (kW) | **0.4177** | 0.4602 | - |
| MAE excl. night (%) | **12.77%** | 13.90% | 13.0% |

### 2.2 MAE by Horizon Group (night included)

| Horizon | v1 (0.3.0) kW | v1 (0.3.0) % | v1-tilt (0.4.0) kW | v1-tilt (0.4.0) % |
|---|---|---|---|---|
| 0 | 0.163 | 5.0% | 0.182 | 5.5% |
| 1 | 0.168 | 5.1% | 0.182 | 5.4% |
| 2 | 0.178 | 5.4% | 0.191 | 5.7% |
| 3-4 | 0.176 | 5.3% | 0.188 | 5.7% |
| 5-8 | 0.174 | 5.3% | 0.177 | 5.4% |
| 9-16 | 0.146 | 4.5% | 0.155 | 4.8% |
| 17-24 | 0.146 | 4.5% | 0.177 | 5.4% |
| 24-48 | 0.175 | 5.4% | 0.217 | 6.6% |
| 0-36 | 0.160 | 4.9% | 0.178 | 5.4% |

### 2.3 Findings
- **model-0.3.0 outperforms 0.4.0 across all horizons**
- 0.4.0 degrades significantly at long horizons (43-47h: MAE 0.28-0.29 kW)
- Our results are slightly better than README's reported MAE (0.1656 vs 0.1906), likely due to 3 corrupted samples removed
- Both models show best performance at 9-16h horizon, worst at 0-2h and 24-48h
- The 0h horizon being worse than mid-range suggests the model expects live PV data it doesn't receive

## 3. New Model: `nwp_transformer`

### 3.1 Architecture

PatchTST encoder + variable-level attention + RevIN for 48 h × 30 min PV
forecasting from NWP-only inputs.

```
Input: NWP [B, 96, 14]   kWp [B]
   |
   RevIN(per-instance, NWP-only)
   |
   Channel-independent PatchEmbedding
   patch_len=16, stride=8, padding=8 -> 12 patches per variable
   |
   Transformer encoder (e_layers=3, batch_first)
   |
   Per-variable patch aggregator (flatten + linear -> d_model)
   |
   Variable-level attention (learnable query pools 14 variable tokens)
   |
   + log1p(kWp) MLP fused post-attention (LayerNorm)
   |
   Linear head -> [B, 96] raw kW
```

Hyperparameters: `d_model=128`, `n_heads=8`, `e_layers=3`, `dropout=0.2`,
`patch_len=16`, `stride=8`, `padding=8`. Total parameters: **0.89 M**.

Design notes:
- **kWp conditioning post-attention**: an early variant broadcast `kWp` as a
  15th input channel; the encoder LayerNorm erased the (constant-in-time)
  scale signal. Fusing `log1p(kWp)` after variable attention preserves the
  capacity context end-to-end.
- **RevIN on NWP only**: power output is predicted in raw kW. RevIN absorbs
  per-instance NWP scale variability across sites/seasons.
- **Channel-independent encoder**: each NWP variable becomes its own token
  sequence (`[B*14, 12, d_model]`), routing the variable axis into the
  batch dimension before the shared Transformer.

### 3.2 Training

- **Data**: 5,000 sites sampled from training set (50 testset sites excluded,
  deterministic seed). Per-site time split 0.8/0.1/0.1 (train/val/test);
  arrays loaded once and shared across splits to fit RAM.
- **Window**: lookback 96 × 30 min → forecast 96 × 30 min. Window stride 96
  (2-day stride) — 2.7 M total windows across train+val splits, 0 windows
  dropped to NaN filter (per-site sliding avoids the multi-site
  shared-time NaN coupling).
- **Loss**: MSE on raw kW.
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4).
- **LR schedule**: CosineAnnealingLR with `T_max = max_epochs * steps`
  (decays from 1e-3 to 0 across the run).
- **Early stop**: patience 10 on val_loss (did not trigger; full 30 epochs).
- **Batch size**: 128.
- **Hardware**: single RTX 3090, ~2 h 41 min wall-clock for 30 epochs.

#### Training curve

| epoch | train_loss | val_loss | val_mae |
|---|---|---|---|
| 0 | 0.374 | 0.551 | 0.245 |
| 5 | 0.243 | 0.413 | 0.226 |
| 10 | 0.231 | 0.370 | 0.230 |
| 15 | 0.224 | 0.362 | 0.225 |
| 20 | 0.217 | 0.349 | 0.223 |
| 23 (lowest val_mae) | 0.213 | 0.342 | 0.219 |
| **28 (best val_loss)** | **0.209** | **0.341** | **0.220** |
| 29 | 0.207 | 0.342 | 0.219 |

Plateau begins around epoch 17 (val_mae ~0.22); the cosine LR finishes
the schedule without further improvement after epoch 23.

### 3.3 Results

Evaluated on the 2,497-sample testset. **0 samples skipped** — the eval
adapter handles testset timestamps on the 15-min grid (`:00 / :15 / :30 / :45`)
by anchoring a 30-min lookback grid at each `ts` and time-interpolating the
hourly NWP onto it.

| Metric | nwp_transformer |
|---|---|
| MAE (kW, all hours) | **0.2036** |
| MAE % (all hours) | **6.30 %** |
| MAE excl. night (kW) | 0.4441 |
| MAE excl. night (%) | **13.69 %** |

NWP source: **Open-Meteo Archive** 14-var (matches the training
distribution; v1/v2 use ICON 9-var via the unchanged `eval/nwp.py`).

## 4. Comparison

| Metric | v1 (0.3.0) | v1-tilt (0.4.0) | **nwp_transformer** |
|---|---|---|---|
| MAE (kW, all hours) | **0.1656** | 0.1940 | 0.2036 |
| MAE % (all hours) | **5.08 %** | 5.89 % | 6.30 % |
| MAE excl. night (%) | **12.77 %** | 13.90 % | 13.69 % |

### 4.1 Findings

- `nwp_transformer` lands **between v1-tilt and v1** on overall MAE, and
  beats `v1-tilt` slightly on the night-excluded normalized metric
  (13.69 % vs 13.90 %).
- It does **not** match the v1 (model-0.3.0) baseline.
- Information-content gap is the most likely driver of the residual gap
  to v1: PSP-based v1 ingests **PV history** (site-specific recent
  generation) in addition to NWP, while `nwp_transformer` is NWP-only
  per Issue #30's preferred form. The night-time term in particular
  benefits from PSP's domain-coded "force 0 at night" behavior; the
  Transformer must learn night handling implicitly from
  `shortwave_radiation`.
- Training plateaued by epoch 17 around val_mae 0.22; longer training
  alone is unlikely to close the gap.

### 4.2 Open improvements (not yet attempted)

| Variant | Hypothesis | Cost |
|---|---|---|
| Night mask in loss / hard-zero post-process | Removes the largest fraction of MAE that PSP wins by domain rule | 1 line + retrain (~3 h) |
| Capacity-factor target (`y / kWp`) | Equalizes loss contribution across capacity range | 1 change + retrain |
| Larger model (`d_model=256`, `e_layers=6`) | Marginal — plateau suggests capacity is not the bottleneck | ~6 h retrain |
| PV-history input | Strongest expected gain — but partially against Issue #30's NWP-only intent | Larger refactor |
