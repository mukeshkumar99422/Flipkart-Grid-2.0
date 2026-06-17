# Gridlock 2.0 — Traffic Demand Prediction

## Problem Statement

Predict continuous traffic **demand** at specific road locations and timestamps,
given contextual signals: weather, road type, number of lanes, landmarks,
large-vehicle permissions, and geospatial coordinates (encoded as geohashes).

---

## Pipeline Overview

```
Raw CSV
  │
  ├── TimestampEngineer         → temporal features (cyclic, slots, peaks)
  ├── GeohashEngineer           → spatial features (lat/lon, prefixes, interactions)
  ├── HierarchicalImputer       → geo-aware imputation (Temperature + Weather)
  ├── TempEngineer              → derived context features (buckets, cross-features)
  │
  ├── ColumnTransformer
  │     ├── StandardScaler      → continuous numerical columns
  │     ├── OneHotEncoder       → low-cardinality categoricals
  │     ├── OrdinalEncoder      → binary ordinals
  │     └── TargetEncoder       → high-cardinality geo×time interaction keys
  │
  ├── Log2FeatureSelector       → 3-stage funnel (variance → correlation → LGBM importance)
  │
  └── LightGBM (LGBMRegressor)  → tuned via Optuna (100 trials, 3-fold CV)

Target: log1p(demand) → predictions: expm1 + clip(0)
```

---

## Feature Engineering

### 1. Timestamp Engineering (`TimestampEngineer`)

Extracts rich temporal signals from the raw `HH:MM` timestamp string.

| Feature | Description |
|---|---|
| `hour` | Hour of the day (0–23) |
| `overall_minutes` | `day × 1440 + hour × 60 + minute` — absolute elapsed minutes across all days |
| `day_minutes` | Minutes elapsed within the current day |
| `day_minutes_sin / cos` | Cyclic encoding of `day_minutes` (2π / 1440 period) — midnight wraps smoothly |
| `day_hour_sin / cos` | Cyclic encoding of `hour` (2π / 24 period) |
| `time_slot_15` | 15-minute bucket: `hour × 4 + minute // 15` → 96 slots/day |
| `time_slot_30` | 30-minute bucket: `hour × 2 + minute // 30` → 48 slots/day |
| `is_early_morning_peak` | Flag: hour 6–7 |
| `is_morning_peak` | Flag: hour 8–10 |
| `is_afternoon_peak` | Flag: hour 11–16 |
| `is_evening_peak` | Flag: hour 17–18 |
| `is_post_evening_peak` | Flag: hour 19–22 |
| `is_night` | Flag: hour 23 or 0–5 |
| `is_business_hours` | Flag: hour 9–17 |
| `is_lunch` | Flag: hour 12–14 |
| `day_time_slot_15` | Interaction key: `day + "_" + time_slot_15` |
| `day_time_slot_30` | Interaction key: `day + "_" + time_slot_30` |

**Why cyclic encoding?** Linear representation of time creates an artificial
discontinuity at midnight. Sin/cos encoding makes 23:59 and 00:00 adjacent,
which they are in reality.

**Why granular peak flags?** Traffic demand has distinct profiles across six
sub-periods of the day. Separate flags let the model weight each independently
rather than learning them from scratch through interaction terms.

---

### 2. Geohash Engineering (`GeohashEngineer`)

Decodes geohashes into spatial coordinates and builds multi-resolution
location representations.

| Feature | Description |
|---|---|
| `lat`, `lon` | Decoded latitude / longitude (cached per unique geohash) |
| `dist_from_center` | Euclidean distance from the median lat/lon of the dataset — approximates proximity to the city core |
| `geo_l3`, `geo_l4`, `geo_l5` | Geohash prefix hierarchy (coarser → finer spatial resolution) |
| `geo5_time_slot_15` | `geo_l5 + "_" + time_slot_15` — location × 15-min slot |
| `geo5_time_slot_30` | `geo_l5 + "_" + time_slot_30` — location × 30-min slot |
| `geo5_hour` | `geo_l5 + "_" + hour` |
| `geo5_early_morning_peak` | `geo_l5 + "_" + is_early_morning_peak` |
| `geo5_morning_peak` | `geo_l5 + "_" + is_morning_peak` |
| `geo5_afternoon_peak` | `geo_l5 + "_" + is_afternoon_peak` |
| `geo5_evening_peak` | `geo_l5 + "_" + is_evening_peak` |
| `geo5_post_evening_peak` | `geo_l5 + "_" + is_post_evening_peak` |
| `geo5_night_peak` | `geo_l5 + "_" + is_night` |

**Why prefix hierarchy?** `geo_l3` captures macro zone trends (city districts),
`geo_l4` captures neighbourhood patterns, `geo_l5` captures street-level
behaviour. Using all three lets the model pick the right spatial granularity
per situation.

**Why geo×peak interaction keys?** The same road may be a bottleneck during
morning rush but empty at noon. Interaction keys bake that joint signal
directly into a single encodable key, which TargetEncoder then compresses
into a learned demand signal.

---

### 3. Hierarchical Imputation (`HierarchicalImputer`)

Handles missing values in **Temperature** and **Weather** using a
geo-time-aware hierarchy instead of a naive global fill.

**Temperature imputation order:**
1. Median Temperature for same `geo5_time_slot_15`
2. Median Temperature for same `geo5_time_slot_30`
3. Median Temperature for same `geo5_hour`
4. Global mean Temperature

**Weather imputation order:**
1. Mode Weather for same `geo5_time_slot_15`
2. Mode Weather for same `geo5_time_slot_30`
3. Mode Weather for same `geo5_hour`
4. Global mode Weather

Stats are learned on the training fold only during cross-validation, then
applied to the validation fold — no leakage.

---

### 4. Temperature & Context Engineering (`TempEngineer`)

Derives additional context-aware features after imputation.

| Feature | Description |
|---|---|
| `temp_bucket` | Discretised temperature: freezing / cold / comfortable / warm / hot |
| `weather_temp_bucket` | `Weather + "_" + temp_bucket` interaction |
| `geo_temp_bucket` | `geo_l5 + "_" + temp_bucket` — local climate context |
| `weather_geo5` | `Weather + "_" + geo_l5` — weather pattern per location |
| `is_cold_weather` | Binary: 1 if Weather == "Snowy" |
| `is_residential_road` | Binary: 1 if RoadType == "Residential" |
| `is_more_lanes` | Binary: 1 if NumberofLanes > 3 |

---

## Encoding Strategy

| Feature Group | Columns | Encoder | Reason |
|---|---|---|---|
| Continuous numerics | `day`, `overall_minutes`, `day_minutes`, `time_slot_15/30`, `lat`, `lon`, `dist_from_center`, `Temperature` | StandardScaler | Zero-mean, unit-variance scaling for LightGBM gradient stability |
| Already bounded / binary | Cyclic features, peak flags, lane/road flags | passthrough | No scaling needed; already in [−1, 1] or {0, 1} |
| Low-cardinality categorical | `RoadType`, `Weather`, `temp_bucket` | OneHotEncoder (drop first) | Few unique values; OHE is safe and interpretable |
| Binary ordinals | `LargeVehicles`, `Landmarks` | OrdinalEncoder | Intrinsically ordered binary features |
| High-cardinality interaction keys | All geo×time keys, `day_time_slot_*` | TargetEncoder (smoothing=10, min_samples_leaf=5) | Thousands of unique combos; target encoding collapses them to a single demand-correlated numeric without cardinality explosion. Smoothing prevents overfitting on rare combinations |

---

## Feature Selection (`Log2FeatureSelector`)

A 3-stage funnel reduces the feature matrix to the most informative
`log2(n_features) × 5` columns before the final model sees any data.

**Stage 1 — Variance filter**
Drop columns with variance below 0.01 (near-constant, uninformative).

**Stage 2 — Pearson correlation filter**
Among pairs with |r| ≥ 0.95, drop the one with lower variance.
Removes redundant duplicates without losing signal.

**Stage 3 — LightGBM importance ranking**
Train a quick probe LGBM (200 trees) on the remaining features.
Keep only the top `log2(n) × 5` features by split importance.

This is fit on the training fold only inside each CV fold — no leakage.

---

## Target Transformation

```python
# Before training
y_log = np.log1p(y)          # reduces right skew

# After prediction
preds = np.expm1(model.predict(X))
preds = np.clip(preds, 0, None)   # remove impossible negatives
```

`demand` is right-skewed. The log transform reduces the influence of extreme
demand spikes on gradient updates and improves R² on the held-out scale.

---

## Model

**LightGBM (`LGBMRegressor`)**

Chosen over XGBoost because:
- 3–5× faster training on large tabular datasets
- Histogram-based splitting handles high-cardinality encoded features efficiently
- Comparable or better accuracy on traffic demand tasks

---

## Hyperparameter Tuning

| Setting | Value |
|---|---|
| Optimizer | Optuna — TPESampler (Bayesian) |
| Trials | 100 |
| CV folds per trial | 3-fold KFold |
| Objective | Maximise mean R² on log-transformed target |
| Early stopping | patience = 50 rounds per fold |

**Search space:**

| Parameter | Range |
|---|---|
| `n_estimators` | 500 – 3000 (step 100) |
| `learning_rate` | 0.01 – 0.15 (log scale) |
| `num_leaves` | 31 – 255 |
| `max_depth` | 4 – 12 |
| `min_child_samples` | 10 – 100 |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `reg_alpha` | 1e-4 – 10.0 (log scale) |
| `reg_lambda` | 1e-4 – 10.0 (log scale) |
| `min_split_gain` | 0.0 – 0.5 |

---

## Cross-Validation & Leakage Prevention

- Preprocessing pipeline is **fit on the training fold only** in every CV iteration.
- `TargetEncoder`, `HierarchicalImputer`, and `Log2FeatureSelector` all
  follow this discipline — no target signal from the validation fold leaks
  into the transformers.
- After tuning, a single final model is trained on the **full training dataset**
  using the best Optuna parameters.

---

## Tools & Libraries

| Library | Role |
|---|---|
| `pandas` | Data loading, manipulation, groupby aggregations |
| `numpy` | Numerical ops, cyclic math, log transforms |
| `scikit-learn` | Pipeline, ColumnTransformer, KFold, encoders, scalers, metrics, custom transformer base classes |
| `category_encoders` | `TargetEncoder` for high-cardinality categorical features |
| `lightgbm` | Primary regression model + early stopping callbacks |
| `optuna` | Bayesian hyperparameter search (TPESampler) |
| `pygeohash` | Decode geohash strings to lat/lon coordinates |
| `matplotlib` | Exploratory data analysis plots |
| `Google Colab + Drive` | Cloud execution environment; no local hardware required |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Cyclic sin/cos for time | Prevents midnight discontinuity; preserves circular nature of time-of-day |
| 6 granular peak-hour flags (vs 2) | Each period has a distinct demand profile; fine-grained flags give the model sharper signal |
| Both 15-min and 30-min slots | Redundancy at different granularities; model picks whichever is more predictive per location |
| Geohash prefix hierarchy l3→l5 | Multi-resolution spatial coverage; coarser = macro zone trends, finer = street-level patterns |
| `dist_from_center` | Proxy for urban density; city centre roads typically have higher and more volatile demand |
| geo×peak interaction keys | Same road behaves very differently across time periods; joint key captures that dependency |
| HierarchicalImputer for Temperature & Weather | Spatially and temporally sensible fills; avoids global-mean bias |
| TargetEncoder (smoothing=10) | Handles thousands of unique geo×time combos without cardinality explosion; smoothing avoids overfitting on rare keys |
| Log2FeatureSelector (3-stage) | Removes noise before the model sees data; smaller feature matrix → faster CV, less overfitting |
| log1p target transform | Tames right-skewed demand distribution; stabilises gradient updates |
| Optuna TPE over GridSearchCV | Intelligently explores 10-dimensional parameter space; grid search is infeasible at this scale |
| Preprocessing inside CV folds | Ensures target encoder and imputer never see validation targets during search |
