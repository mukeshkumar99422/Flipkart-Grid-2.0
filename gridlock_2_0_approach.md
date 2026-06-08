# Gridlock 2.0 — Approach, Feature Engineering & Tools

## Problem Statement

The goal is to predict traffic **demand** (a continuous target) at specific road locations and times, given contextual signals such as weather, road type, number of lanes, nearby landmarks, large-vehicle permissions, and geospatial coordinates encoded as geohashes. The task is a tabular regression problem where the quality of feature engineering and hyperparameter tuning are the primary levers.

---

## Overall Approach

The solution follows a clean, modular pipeline:

1. Raw data ingestion (CSV from Google Drive via Google Colab)
2. Custom feature engineering transformers (sklearn-compatible)
3. A ColumnTransformer that handles each feature group differently
4. Log-transform of the target to reduce skew
5. Bayesian hyperparameter tuning with Optuna
6. Final model training on the full dataset with best parameters
7. Inverse-transform predictions and clip negatives before submission

The entire preprocessing is wrapped in a single `sklearn.Pipeline`, making it leak-proof during cross-validation — the pipeline is always fit only on the training fold and applied to the validation fold.

---

## Feature Engineering

### 1. Timestamp Engineering (`TimestampEngineer`)

The raw `timestamp` column contains time-of-day strings (`HH:MM`). The following features are derived:

| Feature | Description |
|---|---|
| `hour`, `minute` | Basic time decomposition |
| `overall_minutes` | Absolute elapsed minutes: `day × 1440 + hour × 60 + minute`; captures long-range time trends across days |
| `day_minutes` | Minutes elapsed within the current day (`hour × 60 + minute`) |
| `day_minutes_sin`, `day_minutes_cos` | Cyclic (sinusoidal) encoding of `day_minutes` — ensures midnight wraps smoothly and the model sees time as circular, not linear |
| `time_slot` | 15-minute bucket within the day (`hour × 4 + minute // 15`), giving 96 discrete slots per day |
| `is_morning_peak` | Binary flag: 1 if hour is 7–9 (rush hour) |
| `is_evening_peak` | Binary flag: 1 if hour is 17–19 (rush hour) |
| `is_night` | Binary flag: 1 if hour is 22–23 or 0–5 |
| `day_time_slot` | Interaction key: `day_str + "_" + time_slot_str`; allows the model to learn unique demand patterns for each day-slot combination |

The `timestamp` and `minute` columns are dropped after transformation.

**Rationale:** Traffic demand is highly periodic. Cyclic encoding of minutes prevents artificial discontinuities (e.g., 23:59 and 00:00 appearing far apart). Peak-hour flags inject domain knowledge directly, and 15-minute granularity captures intra-hour variation that hourly binning misses.

---

### 2. Geohash Engineering (`GeohashEngineer`)

Geohashes encode geographic coordinates as short alphanumeric strings. The engineer:

| Feature | Description |
|---|---|
| `lat`, `lon` | Decoded latitude and longitude from the geohash (cached per unique geohash for speed) |
| `geo_l3`, `geo_l4`, `geo_l5` | Hierarchical geohash prefixes at 3, 4, and 5 characters — coarser substrings represent larger geographic regions |
| `geo5_time_slot` | Interaction: `geo_l5 + "_" + time_slot` — captures demand patterns at a medium-resolution area × time granularity |
| `geo5_hour` | Interaction: `geo_l5 + "_" + hour` |
| `geo5_day` | Interaction: `geo_l5 + "_" + day` |

The raw `geohash` and `geo_l6` (too fine-grained) columns are dropped.

**Rationale:** Geohash prefixes act as spatial aggregation levels. Coarser prefixes (`geo_l3`, `geo_l4`) capture city-zone demand trends; finer ones (`geo_l5`) capture street-level patterns. Interaction keys (e.g., `geo5_time_slot`) let the model learn that the same area has very different demand at 8 AM vs. 8 PM, which a purely additive treatment of location and time cannot express.

---

### 3. Hierarchical Temperature Imputation (`HierarchicalTempImputer`)

Temperature contains missing values. Rather than a simple global or column median fill, imputation is done hierarchically:

1. Fill missing temperatures using the **median Temperature for the same `geo5_time_slot`** (same location cluster, same time slot)
2. Any remaining gaps are filled using the **`geo5_hour`** median
3. Then by **`geo5_day`** median
4. Finally, any still-missing values are filled with the **global median**

This ensures that imputed temperatures are geographically and temporally sensible, not just a one-size-fits-all value. After imputation, a `RobustScaler` (uses median and IQR, not mean and std) is applied to handle outlier temperatures without distortion.

---

## Encoding Strategy by Feature Type

| Feature Group | Columns | Method | Reason |
|---|---|---|---|
| Temperature | `Temperature` + geo-time keys | Hierarchical median impute → RobustScaler | Outlier-robust scaling; spatially aware imputation |
| Low-cardinality categoricals | `RoadType`, `Weather` | SimpleImputer (mode) → OneHotEncoder (drop first) | Few unique values; OHE is safe and interpretable |
| Binary ordinals | `LargeVehicles`, `Landmarks` | OrdinalEncoder with explicit category order | These are intrinsically ordered binary features |
| High-cardinality interaction keys | `geo5_time_slot`, `geo5_hour`, `geo5_day`, `geo_l5`, `geo_l4`, `geo_l3`, `day_time_slot` | TargetEncoder (smoothing=15, min_samples_leaf=10) | Thousands of unique values; target encoding collapses them to a single numeric signal without cardinality explosion; smoothing prevents overfitting on rare combinations |

---

## Target Transformation

The target `demand` is right-skewed. A `log1p` transform (`y_log = np.log1p(y)`) is applied before training. Predictions are back-transformed using `expm1` and clipped to zero to remove any negative predictions.

**Effect:** Reduces the influence of extreme demand spikes on the loss function, stabilises gradient updates, and improves R² on the actual scale.

---

## Model

**LightGBM (`LGBMRegressor`)** is chosen over XGBoost for the following reasons:

- 3–5× faster training on tabular data
- Histogram-based splitting handles high-cardinality features efficiently
- Native support for categorical feature types (though target encoding is used here for full control)
- Comparable or better accuracy on traffic tabular datasets

---

## Hyperparameter Tuning

**Optuna** with the **TPE (Tree-structured Parzen Estimator)** sampler is used for Bayesian optimisation over 100 trials. The objective function performs 3-fold cross-validation and maximises mean R² on the log-transformed target.

Parameters tuned:

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

Early stopping (patience = 50 rounds) is applied inside each fold to avoid over-fitting during the search. Preprocessing is re-fit inside each fold to avoid data leakage.

---

## Cross-Validation Strategy

- **During tuning:** 3-fold KFold (faster iteration across 100 Optuna trials)
- **Pipeline discipline:** `preproc_pipeline.fit_transform` is always called on the training fold only; `transform` is called on the validation fold. This prevents target-encoder and imputer leakage.
- After tuning, the best parameters are used to train a single final model on the full training dataset.

---

## Tools & Libraries

| Tool / Library | Role |
|---|---|
| **pandas** | Data loading, exploration, manipulation |
| **numpy** | Numerical operations, cyclic encoding math, log transforms |
| **scikit-learn** | Pipeline, ColumnTransformer, KFold, OrdinalEncoder, OneHotEncoder, SimpleImputer, RobustScaler, StandardScaler, BaseEstimator/TransformerMixin (for custom transformers), R² metric |
| **category_encoders** | `TargetEncoder` for high-cardinality categorical features |
| **LightGBM (`lgb`)** | Primary regression model (`LGBMRegressor`), early stopping callbacks |
| **Optuna** | Bayesian hyperparameter optimisation (TPESampler, 100 trials) |
| **pygeohash** | Decoding geohash strings to latitude/longitude |
| **matplotlib** | Exploratory data visualisation |
| **Google Colab + Drive** | Cloud execution environment and data storage (no local GPU/hardware required) |

---

## Key Design Decisions Summary

| Decision | Rationale |
|---|---|
| LightGBM over XGBoost | Faster, equally accurate or better on tabular traffic data |
| Optuna Bayesian search over GridSearchCV | Explores a high-dimensional space (8+ parameters) intelligently; grid search is computationally infeasible at this scale |
| log1p target transform | Demand is right-skewed; transform reduces outlier influence and improves gradient stability |
| Hierarchical temperature imputation | Spatially and temporally aware; avoids the bias of a global fill |
| RobustScaler for temperature | Handles outlier temperatures better than StandardScaler |
| Cyclic sin/cos encoding for time | Prevents the model from seeing midnight as discontinuous; preserves the circular nature of time-of-day |
| Peak-hour binary flags | Direct domain knowledge injection; known to have large impact on traffic demand |
| Geohash prefix hierarchy (l3–l5) | Multi-resolution spatial representation; coarser captures macro trends, finer captures local patterns |
| TargetEncoder with smoothing=15 | Encodes high-cardinality geo×time interactions without cardinality explosion; smoothing prevents overfitting on rare combos |
| Preprocessing inside CV folds | Prevents leakage from target encoder and imputer into the validation signal |
