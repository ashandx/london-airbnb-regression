# london-airbnb-regression

Predicting nightly Airbnb listing price from a London snapshot — the kind of automated
valuation problem a short-term rental platform, property manager, or pricing tool solves
in production: given a listing's attributes, estimate what it should charge. Unlike the
Ames Housing set, this is raw scraped data rather than competition-cleaned, so most of the
week went into cleaning and deciding what a "feature" even means before any model touched
it.

## Data

`data/listings.csv.gz` is [Inside Airbnb](https://insideairbnb.com/london/)'s London
snapshot (CC BY 4.0), scraped 2026-06-19 to 2026-06-29: 92,638 listings, 90 columns, 63.6MB
in memory. It's committed to the repo as-is (50.3MB compressed) rather than gitignored —
over GitHub's 50MB soft-warning threshold but under the 100MB hard cap, so a push warning
is expected and not an error.

## What's been done (Monday–Friday)

**EDA (Mon–Wed):** 13 columns are entirely null in this snapshot and dropped outright.
`price` (the target) is a string column (`"$234.50"` style) and only 67.2% populated —
the other ~30k listings have no label and can't be used for supervised training, so
they're excluded rather than imputed. `log1p(price)` is close to normal and centred
around 5.2 (back-transforming to ~£180, the actual median), confirming it as the right
target transform. The 99th percentile price is £1,373 against a max of £527,524 — the
top 0.1% of listings are 84x the 99.9th-percentile value, so extreme values are capped at
the 99th percentile before logging rather than left to distort a linear model.
`neighbourhood_cleansed` shows a clean, sensible central-London premium (City of London
£325, Westminster £307, down to Sutton £85.80 — roughly 3.8x spread), and `room_type` is
heavily imbalanced (Entire home/apt 66.3%, Private room 33.3%, Shared/Hotel room under
0.25% each). Review volume and score barely move price (all correlations under 0.15) —
read as a trust/occupancy signal, not a pricing lever. `availability_365` looks like
noise as a raw linear correlation (0.05) but a fully-blocked calendar (`== 0`, 28.6% of
listings) sits apart from every other availability bucket, so it's engineered as a binary
flag instead of trusted as a raw number.

**Feature engineering (Thu):** `amenities` is a stringified list, parsed into a count
(mean 28.8) and a handful of presence flags. `has_air_conditioning` (0.236) is the
strongest of these — AC is genuinely rare in UK housing stock, so its presence reads as a
premium-property signal rather than a baseline expectation. One flag
(`has_free_parking`) came back as a constant zero, not a low correlation — the string
`"Free parking"` doesn't appear anywhere in the actual amenities data, a genuine naming
mismatch caught by checking real value counts rather than assumed. `accommodates`,
`beds`, `bedrooms`, and `bathrooms` are all strongly collinear (0.51–0.83 pairwise), so
`beds`/`bedrooms` are dropped in favour of two ratio features
(`beds_per_accommodate`, `bath_per_bedroom`) built the same way `TotalPorchSF` was built
on Ames. Of the seven review subscores, only `review_scores_location` (0.139) clears a
meaningful bar — the overall `review_scores_rating` composite is skipped in favour of
that single subscore.

**Preprocessing pipeline (Fri):** target is `log_price_capped` —
`log1p(price.clip(upper=P99))`. Final feature set: 8 numeric, 5 binary, and 2 categorical
columns (`room_type`, `neighbourhood_cleansed`, 4 and 33 categories respectively) feeding
a `ColumnTransformer` (median-impute + `StandardScaler` for numeric, most-frequent-impute
for binary, most-frequent-impute + `OneHotEncoder(handle_unknown="ignore")` for
categorical). Train/test split (80/20, `random_state=42`) happens before the
preprocessor is fit, so no test-set statistics leak into training. The transformed
training matrix is `(49792, 50)` — two-thirds of that width comes from
`neighbourhood_cleansed`'s 33 one-hot columns alone.

## What Saturday adds

Cross-validated comparison of four models on the transformed features:
LinearRegression baseline, Ridge, Lasso, and RandomForestRegressor, each scored on
log-RMSE and R² using `StratifiedKFold` on `qcut`-binned price deciles rather than plain
KFold — the same evaluation upgrade the Ames project's Milestone 3 session introduced,
applied here from the start rather than retrofitted later.

## Setup

```bash
conda create -n london-airbnb-regression python=3.11
conda activate london-airbnb-regression
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter
```

## Run

```bash
jupyter notebook notebooks/london_airbnb_regression.ipynb
```

Run all cells top to bottom.

## Known limitations

- The ~30k listings with no `price` are dropped for training, not imputed — no amount of
  imputation fixes a missing target without fabricating labels. Estimating price for
  those listings would be a separate inference problem, not part of this pipeline.
- The 99th-percentile price cap (£1,373) is a first-pass choice based on the observed
  quantile jump, not an outlier-detection method tuned or validated against alternatives.
- `beds_per_accommodate` and `bath_per_bedroom` both correlate negatively with price
  (-0.22, -0.22), but that's partly because larger multi-bedroom properties naturally
  have lower per-room ratios than studios despite costing more in absolute terms — the
  ratios are partly re-encoding property size, not literally "more beds/baths = cheaper."
- Amenity presence flags depend on exact string matching against however hosts entered
  amenity names (`has_free_parking` returned an all-zero column for exactly this reason)
  — the current flag list was checked against the top-30 most common amenities, not
  exhaustively validated against every naming variant in the data.
- The Random Forest preprocessor (once built) will likely reuse the same
  `StandardScaler`'d feature matrix as the linear models, even though trees don't need
  scaling — a deliberate simplification to keep every model comparable on the same
  feature matrix, matching the same call made on the Ames project.
