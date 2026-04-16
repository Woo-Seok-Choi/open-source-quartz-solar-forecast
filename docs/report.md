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

## 3. New Model

### 3.1 Architecture
*(TODO)*

### 3.2 Training
*(TODO)*

### 3.3 Results
*(TODO)*

## 4. Comparison
*(TODO: side-by-side comparison of baseline vs new model)*
