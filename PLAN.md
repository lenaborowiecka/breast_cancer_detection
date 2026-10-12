# PLAN.md — Breast cancer detection, end to end

Build a supervised binary classifier (malignant = 1, benign = 0) on the UCI Breast Cancer Wisconsin (Diagnostic)
dataset. Use scikit-learn `Pipeline`s for every step from raw features to prediction, and finish with a Streamlit
app that loads the saved pipeline. Most of the effort goes into **data cleaning, processing and transformation**.
Modelling and the app come after that.

Work through the features **in order**. Each one is small and finishes in a working state where `just check`
passes. Tick the box when the "Done when" criteria are met. The rules in `CLAUDE.md` (metrics, no leakage,
persistence, code style) apply to every feature.

---

## Data profile (from inspecting `data/raw/`)

Source: `https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip`, extracted to
`data/raw/wdbc.data` (headerless CSV, 569 × 32) and `data/raw/wdbc.names`.
SHA-256 of `wdbc.data`: `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`.

The numbers below come from profiling the full file, so they are only descriptive. All modelling decisions must
be re-derived from the **training split only**.

### Structure and integrity: the data is already clean

- Column order: `id`, `diagnosis`, then 10 measurements × {mean (cols 3–12), SE (13–22), worst (23–32)}. Within
  each block the order is radius, texture, perimeter, area, smoothness, compactness, concavity, concave points,
  symmetry, fractal dimension.
- Pandas infers the right types (`id` int64, `diagnosis` string, 30 × float64). There are no nulls, no `?`
  placeholders, no duplicate IDs and no duplicated feature rows. Every value is ≥ 0.
- `id` ranges from 8670 to 911320502. It is an identifier and carries no information; drop it.
- Diagnosis: 357 B / 212 M (37.3 % malignant).
- **The file is not shuffled.** The first 19 rows are all `M`, and row position correlates with the target
  (r = −0.28). Every split and CV fold must shuffle.

So "cleaning" here mostly means **verifying** that the data is clean, assigning names, and encoding the target.
The real work is in validation rules that the app can reuse, and in transformations.

### Domain consistency (makes good validation rules)

| Rule | Observed on full data |
|---|---|
| `X_worst >= X_mean` for all 10 measurements | holds for 100 % of rows |
| `area_mean / (π · radius_mean²)` | 0.917 – 1.064 (median 0.981) |
| `perimeter_mean / (2π · radius_mean)` | 0.983 – 1.133 (median 1.030) |
| concavity == 0 ⇔ concave_points == 0 (for mean, SE and worst) | holds. The 13 zero rows are zero in **all six** concavity columns |

- **The 13 zero-concavity rows are all benign.** They are real values (smooth contours), not missing data, so do
  not impute them.
- The `.names` file defines compactness as `perimeter²/area − 1`, but it is computed per nucleus and then averaged,
  so recomputing it from the mean columns only reaches r = 0.965. Don't use it as a validation rule, and don't add
  it as an engineered feature (it's redundant).

### Distributions and outliers

- **Scales differ enormously:** the ratio of the largest column std to the smallest is about 215,000 (area in
  the hundreds, fractal_dimension_se around 0.001). Scaling is mandatory for logistic regression, SVM and kNN.
- **Skew:** 22 of 30 features have skew > 0.9. The worst are the SE features: area_se 5.4, concavity_se 5.1,
  fractal_dimension_se 3.9, perimeter_se 3.4, radius_se 3.1 (kurtosis up to 49).
- **`log1p` does not fix the small-valued features** (concavity_se goes from 5.1 to 4.2, fractal_dimension_se from
  3.9 to 3.9), because `log1p(x) ≈ x` when x ≪ 1. **Yeo-Johnson** brings every feature to |skew| < 0.25 and
  handles the zeros. Box-Cox and plain `log` fail on the zeros.
- **Outliers (IQR 1.5):** 171 of 569 rows (30 %) have at least one. They are mostly malignant (114 M vs 57 B), and
  for the size features (radius, perimeter, area and their SE / worst) **100 % of outliers are malignant**.
  Removing outliers would delete the signal and bias the data towards benign, so **keep every row**. 14 rows have a
  |z| > 5 (max 12, on concavity_se). Yeo-Johnson or `RobustScaler` handle these.

### Redundancy

- 21 feature pairs have |r| > 0.9, and 15 have |r| > 0.95. Radius, perimeter and area are close to deterministic
  of each other (radius ~ perimeter r = 0.998; radius ~ area Spearman 0.9996). VIF in the mean group: radius 1607,
  perimeter 1956, area 56, compactness 23, concave_points 22.
- Other notable pairs: concavity ~ concave_points (0.92), mean ~ worst for size and texture (0.91 – 0.97).
- PCA on standardized data: 7 components explain 90 % of the variance, 10 explain 95 %.

### Signal (univariate ROC-AUC against malignant)

- Strongest: perimeter_worst 0.976, radius_worst 0.970, area_worst 0.970, concave_points_worst 0.967,
  concave_points_mean 0.964.
- Close to uninformative on their own: symmetry_se 0.44, smoothness_se 0.47, fractal_dimension_mean 0.48,
  texture_se 0.51. These are the candidates for removal in the ablation (F9).
- The `.names` file says the classes are **linearly separable** with all 30 features, and that one plane on
  (area_worst, smoothness_worst, texture_mean) reached 97.5 % accuracy.

### Quick baselines (5-fold stratified CV on the full data, for reference only)

| Pipeline | Recall (M) | PR-AUC | ROC-AUC | Accuracy |
|---|---|---|---|---|
| StandardScaler + LogisticRegression | 0.944 | 0.994 | 0.995 | 0.974 |
| Yeo-Johnson + LogisticRegression | 0.953 | 0.996 | 0.997 | 0.975 |
| 3 features (paper) + LogisticRegression | 0.925 | 0.992 | 0.992 | 0.963 |
| RandomForest (default) | 0.934 | 0.987 | 0.989 | 0.954 |

Implications:
1. Linear models are already very strong. The headroom is in **malignant recall**, which the threshold controls,
   not in ranking, where ROC-AUC is already about 0.995.
2. A 20 % test set has about 114 rows, of which about 43 are malignant, so **one missed cancer moves recall by about
   2.3 points**. Use repeated stratified CV for selection, and report the test metrics with bootstrap confidence
   intervals.
3. `sklearn.datasets.load_breast_cancer` holds the same data, but its target is **inverted** (0 = malignant).
   Any test fixture built from it must remap the target.

---

## Phase 0 — Foundations

### F0. Project setup
- [ ] Rename the package from `src/src/` to `src/breast_cancer_detection/`. Update the `[project.scripts]` entry
  (`breast-cancer = "breast_cancer_detection.cli:app"`), `[tool.hatch.build.targets.wheel] packages`,
  `tests/test_src.py`, `docs/api.md` and `__main__.py`.
- [ ] Add runtime deps: `uv add scikit-learn pandas pyarrow joblib matplotlib seaborn streamlit`.
- [ ] Add a notebook group: `uv add --group notebooks jupyterlab ipykernel statsmodels` (statsmodels is for VIF in
  the EDA).
- [ ] Create `notebooks/`, `models/`, `reports/figures/`, `data/processed/` and `app/`, each with a `.gitkeep`.
- [ ] Update `.gitignore`: ignore `data/processed/`, `models/*.joblib` and `.ipynb_checkpoints/`. Commit
  `data/raw/` (124 KB, so the project is reproducible offline and the checksum is pinned).

**Done when:** `uv sync` works, `uv run python -c "import breast_cancer_detection"` succeeds, and `just check`
passes.

### F1. Config module — `config.py`
- [ ] Paths: `PROJECT_ROOT`, `RAW_DATA_PATH`, `PROCESSED_DIR`, `MODELS_DIR`, `REPORTS_DIR`, all built with
  `pathlib` relative to the project root.
- [ ] Constants: `RANDOM_STATE = 42`, `TEST_SIZE = 0.2`, `CV_FOLDS = 5`, `CV_REPEATS = 10`,
  `TARGET = "diagnosis"`, `ID_COL = "id"`, `RAW_DATA_URL`, `RAW_DATA_SHA256`, and `LABELS = {0: "benign", 1:
  "malignant"}`.
- [ ] Feature names, generated rather than typed out: `MEASUREMENTS` (10 names, in file order) × `STATS =
  ("mean", "se", "worst")` → `FEATURES` (30 snake_case names such as `radius_mean`, `concave_points_worst`, in
  file column order). Also `FEATURE_GROUPS = {"mean": [...], "se": [...], "worst": [...]}`.
- [ ] Validation tolerances, taken from the data profile with some margin: `AREA_RATIO_RANGE = (0.85, 1.15)` and
  `PERIMETER_RATIO_RANGE = (0.90, 1.25)`.

**Done when:** a test asserts that `len(FEATURES) == 30`, the names are unique, and the order matches the file
layout (`FEATURES[0] == "radius_mean"`, `FEATURES[10] == "radius_se"`, `FEATURES[20] == "radius_worst"`).

---

## Phase 1 — Data ingestion, validation and cleaning

### F2. Raw data ingestion — `data.py`
- [ ] `download_raw(force: bool = False) -> Path`: if `data/raw/wdbc.data` is missing, download the zip and
  extract **only** `wdbc.data` and `wdbc.names` (reject paths containing `/` or `..`). Verify the SHA-256 and raise
  an error on mismatch. Does nothing if the file is already present.
- [ ] `load_raw(path=RAW_DATA_PATH) -> pd.DataFrame`: read with `header=None` and
  `names=[ID_COL, TARGET, *FEATURES]`. Set explicit dtypes (`id` int64, `diagnosis` string, features float64) and
  `na_values=["?"]`, so that any future corruption shows up as NaN and fails validation.
- [ ] CLI command `breast-cancer download`.

**Done when:** `load_raw()` returns a 569 × 32 frame with the right names and dtypes. A test reads a small fixture
CSV written to `tmp_path`, with no network access. A second test checks the checksum.

### F3. Validation — `validation.py`
Two layers: **structural** checks, which must hold or the run aborts, and **domain** checks, which hold on this
dataset and are reused later to flag implausible input in the app.

- [ ] Structural checks, `validate_schema(df)`:
  - the expected columns are present, in the expected order, with the expected dtypes;
  - `diagnosis` is in `{"M", "B"}`;
  - there are no nulls in any column;
  - all features are finite and `>= 0`;
  - `id` is unique, and there are no fully duplicated feature rows.
- [ ] Domain checks, `check_domain_rules(X) -> pd.DataFrame`. Returns one boolean column per rule, per row, and
  takes **features only**, so the app can call it:
  - `X_worst >= X_mean` for each of the 10 measurements;
  - `area_mean / (π · radius_mean²)` lies within `AREA_RATIO_RANGE`, with the same check for `_worst`;
  - `perimeter_mean / (2π · radius_mean)` lies within `PERIMETER_RATIO_RANGE`;
  - zero coupling: `concavity_x == 0` ⇔ `concave_points_x == 0`, for x in mean / SE / worst.
- [ ] `validate_raw(df)` runs both layers, raises `DataValidationError` if any structural check fails, and writes
  `reports/data_quality.json` (row count, class counts, zero-concavity count, and a pass count per rule).

**Done when:** every rule passes on the real data, and each rule has a test that fails on a deliberately broken
fixture (null, negative value, bad label, duplicate ID, worst < mean, impossible area/radius ratio, concavity 0
with concave points > 0).

### F4. Cleaning — `cleaning.py`
Because the raw data is already clean, this step is thin and **pure** (no I/O):
- [ ] `clean(df) -> pd.DataFrame`:
  - drop `id`;
  - strip and uppercase `diagnosis`, then map `M → 1`, `B → 0` as int8. Any unmapped value is an error, never a
    silent NaN;
  - drop exact duplicate rows and log the count (expected: 0);
  - **keep** the 13 zero-concavity rows unchanged;
  - **keep** every outlier row, because they are mostly malignant and are the signal (see the profile).
  - Do **not** shuffle here. Shuffling happens in the split (F6).
- [ ] `make_dataset()`: runs `load_raw` → `validate_raw` → `clean`, then writes
  `data/processed/wdbc_clean.parquet`.
- [ ] CLI command `breast-cancer prepare`.

**Done when:** the parquet file has 569 rows, 30 feature columns, and `diagnosis` ∈ {0, 1} with 212 positives.
Tests cover the label mapping (including a bad label raising), ID removal, and the preservation of zeros and
outliers.

### F5. Exploratory data analysis — `notebooks/01_eda.ipynb`
The notebook loads data **through the package** (`from breast_cancer_detection.data import ...`), never with copied
loading code. Save each figure to `reports/figures/`. Sections:
- [ ] **Overview:** shape, dtypes, `describe()`, nulls, duplicates, and the validation report from F3.
- [ ] **Row order:** plot the diagnosis against row index to show the file is ordered, which justifies shuffling.
- [ ] **Target:** class balance (37 % malignant) and what it implies (stratify; use recall and PR-AUC rather than
  accuracy).
- [ ] **Univariate:** histograms by class for each of the 3 groups. A skew / kurtosis table that flags |skew| > 1.
- [ ] **Zeros:** the 13 zero-concavity rows are all benign, and are zero in all six concavity columns together.
- [ ] **Outliers:** IQR outlier counts per feature, with the share that is malignant. This shows that removing
  outliers would remove malignant cases.
- [ ] **Signal:** univariate ROC-AUC per feature, ranked, plus boxplots by class for the top 6 and the bottom 4.
- [ ] **Redundancy:** a clustered correlation heatmap, the list of pairs with |r| > 0.9, VIF per group, and
  scatter plots of radius vs perimeter vs area.
- [ ] **Transformations:** a skew table for raw vs `log1p` vs Yeo-Johnson, showing that `log1p` fails on the small
  SE features. Before/after histograms for area_se, concavity_se and fractal_dimension_se.
- [ ] **Structure:** a scree plot from PCA, and a PC1 vs PC2 scatter coloured by class.
- [ ] **Conclusions:** concrete decisions that feed F7–F9, for example "use Yeo-Johnson on all features", "test
  dropping perimeter and area", "test a correlation filter at 0.95", and "test removing the 4 features with AUC
  ≈ 0.5".

**Done when:** the notebook runs top to bottom
(`uv run --group notebooks jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb`), and its conclusions
match or update the data profile above.

### F6. Train/test split — `data.py`
- [ ] `split_data(df) -> (X_train, X_test, y_train, y_test)`: `train_test_split(..., stratify=y, shuffle=True,
  test_size=TEST_SIZE, random_state=RANDOM_STATE)`. This gives about 455 training and 114 test rows, of which about
  170 and 42 are malignant.
- [ ] Save `data/processed/train.parquet` and `test.parquet`, and add a `load_split()` helper.
- [ ] `make_cv()` returns `RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS,
  random_state=RANDOM_STATE)`. Use it everywhere CV is needed.

**Done when:** a test checks that the split is stratified (the malignant ratio differs by less than 2 percentage
points), reproducible, and that train and test share no rows. All 13 zero-concavity rows exist somewhere in the
data, but the split is not forced to place them.

---

## Phase 2 — Processing and transformation (sklearn pipeline)

All transformers follow the scikit-learn API: they learn everything in `fit` from training data only, and return
named feature output (`set_output(transform="pandas")` / `get_feature_names_out`). This keeps feature names flowing
through to the app and to the explanations.

### F7. Custom transformers — `transformers.py`
- [ ] `FeatureSelector(drop: list[str] | None = None, keep: list[str] | None = None)`: a stateless, domain-driven
  selection, used for example to drop `perimeter_*` and `area_*` (radius already captures them) or the
  low-signal SE features. This lets the CV search compare fixed feature sets.
- [ ] `CorrelationFilter(threshold=0.95)`: in `fit`, compute pairwise |r| on training data. For each pair above
  the threshold, drop the feature with the lower univariate correlation to `y`. Store `features_to_drop_`.
  Expected behaviour at 0.95: one of each radius / perimeter / area triple is kept per stat.
- [ ] `SkewCorrector(threshold=0.75)`: a Yeo-Johnson `PowerTransformer(standardize=False)` applied only to the
  features whose training skew exceeds the threshold. The others pass through. **No `log1p` option**: the profile
  shows it doesn't fix the small-valued features.
- [ ] Shared: `get_feature_names_out`, `set_output` support, and passes the relevant
  `sklearn.utils.estimator_checks` checks.

**Done when:** unit tests show that each transformer learns its state only in `fit` (fitting on a subset and
transforming other data gives no refit), keeps names, handles the zero-concavity rows, and that
`CorrelationFilter` drops perimeter or radius from the near-collinear pair.

### F8. Preprocessing pipeline — `features.py`
- [ ] `build_preprocessor(params) -> Pipeline` with the steps `select` (FeatureSelector) → `skew`
  (SkewCorrector) → `corr` (CorrelationFilter) → `scale` (`StandardScaler` or `RobustScaler`). Each step can be
  set to `"passthrough"` so the CV search can toggle it.
  - Order matters: correct skew **before** the correlation filter, because Pearson correlation on heavily skewed
    data is dominated by the outliers.
  - The pipeline is applied to all 30 numeric features. No `ColumnTransformer` is needed, because there are no
    categorical or missing values.
- [ ] `build_pipeline(model, params) -> Pipeline`: `("preprocess", build_preprocessor(...)), ("model", model)`.
- [ ] `describe_transformed(pipeline, X)`: returns the shape, the output feature names, and the skew per output
  column, so you can inspect what the pipeline produces.

**Done when:** `build_pipeline(LogisticRegression()).fit(X_train, y_train)` works with every step on and with every
step set to passthrough. A test confirms that transforming the test data does not change the fitted state.

### F9. Preprocessing ablation — `notebooks/02_preprocessing.ipynb`
Use `make_cv()` on **train only**, with a fixed `LogisticRegression(max_iter=5000)`. Variants:
- [ ] A: `StandardScaler` only (baseline).
- [ ] B: A + `SkewCorrector`.
- [ ] C: B + `CorrelationFilter(0.95)`, and also at 0.90.
- [ ] D: B + `FeatureSelector` dropping `perimeter_*` and `area_*` (24 features).
- [ ] E: B + dropping the low-AUC features (`symmetry_se`, `smoothness_se`, `fractal_dimension_mean`,
  `texture_se`).
- [ ] F: `RobustScaler` in place of `StandardScaler`.
- [ ] Report the mean ± std of recall, PR-AUC, ROC-AUC and log-loss for each variant, plus the number of output
  features. Keep a step only if it improves the scores, or if it reduces the feature count without a significant
  loss (a difference smaller than 1 std counts as a tie, and the simpler variant wins).
- [ ] Write the winning preprocessing parameters into `config.py`.

**Done when:** the chosen configuration is fixed in config, and a results table backs each decision.

---

## Phase 3 — Modelling and evaluation

### F10. Baseline model — `train.py`
- [ ] The F9-winning preprocessor + `LogisticRegression`, evaluated with `cross_validate(cv=make_cv())` using the
  metrics required by CLAUDE.md.
- [ ] CLI command `breast-cancer train --model logreg`.

**Done when:** the CV scores print and are saved to `reports/cv_baseline.json`. They should be close to the data
profile: recall around 0.95, PR-AUC around 0.995.

### F11. Model comparison and tuning
- [ ] Candidates: logistic regression (L1/L2/ElasticNet, `C`, `class_weight` None or balanced), linear SVM and RBF
  SVM (`probability=True` or a calibrated wrapper), `HistGradientBoostingClassifier`, random forest, and kNN.
- [ ] Run `RandomizedSearchCV` over the **whole pipeline**, so preprocessing parameters (for example
  `preprocess__corr__threshold` and `preprocess__select__drop`) are tuned together with the model.
  `scoring={"ap": "average_precision", "recall": "recall", "roc_auc": "roc_auc"}`, `refit="ap"`.
- [ ] Add a **reduced-input candidate**: logistic regression on a small feature subset, starting from the paper's
  (area_worst, smoothness_worst, texture_mean) or from L1-selected features. Fewer inputs make the app much easier
  to use. Report the trade-off against the full model.
- [ ] Save a comparison table to `reports/model_comparison.csv`. Choose the simplest model within 1 std of the
  best PR-AUC. Expect the linear models to win, because the data is close to linearly separable.

**Done when:** one model (and its feature set) is chosen, with the reasons written down in
`reports/model_selection.md`.

### F12. Threshold tuning and final evaluation — `evaluate.py`
- [ ] Choose the decision threshold on **training data only**, using `TunedThresholdClassifierCV` or out-of-fold
  `cross_val_predict` probabilities. Aim for malignant recall ≥ 0.97 with the best precision available. Check
  calibration (Brier score, reliability curve), and wrap the model in `CalibratedClassifierCV` if the probability
  shown in the app would otherwise be misleading.
- [ ] Refit on all training data and evaluate **once** on the test set. Report the confusion matrix, the
  classification report, ROC and PR curves, and the calibration curve, with figures saved to `reports/figures/`.
- [ ] Add **bootstrap 95 % confidence intervals** (1000 resamples) for test recall, precision and PR-AUC, because
  there are only about 42 malignant test cases.
- [ ] Compute permutation importance on the test set, grouped by measurement, since correlated features share
  importance.

**Done when:** `reports/test_metrics.json` (point estimates and CIs) and the figures exist.

### F13. Persistence — `train.py`
- [ ] Save the fitted pipeline to `models/model.joblib`, and write `models/metadata.json` with: the **input**
  feature order (the raw columns the pipeline expects, which may be fewer than 30 for a reduced model), the
  labels, the threshold, the CV and test metrics with CIs, the training timestamp, the versions of sklearn, pandas
  and Python, the raw-data SHA-256, and the per-input-feature training min / median / max / p1 / p99 (used for the
  app's defaults and range warnings).
- [ ] CLI command `breast-cancer train --final` runs the whole flow, from the prepared data through to the saved
  artifacts.

**Done when:** a save → load round-trip test in `tmp_path` gives identical predictions.

---

## Phase 4 — Inference and Streamlit app

### F14. Prediction module — `predict.py`
- [ ] `load_model(models_dir) -> (pipeline, metadata)`: raises a clear error if the artifacts are missing, and
  warns if the saved sklearn version differs from the installed one.
- [ ] `validate_input(df, metadata) -> (X, warnings)`:
  - **Errors:** a missing column, a non-numeric value, a negative value, or a NaN.
  - **Warnings:** a value outside the training [p1, p99] range, or a failed domain rule from `check_domain_rules`
    (for example worst < mean). Warnings don't block the prediction, but they are shown to the user.
  - Ignores extra columns, such as `id` or `diagnosis` in an uploaded CSV, and reorders the columns to match the
    metadata.
- [ ] `predict(df, pipeline, metadata) -> pd.DataFrame` returns `probability_malignant`, a `prediction` based on
  the stored threshold, and a `label` column.
- [ ] CLI command `breast-cancer predict input.csv`.

**Done when:** tests cover valid input, a missing column, a non-numeric value, shuffled columns, an out-of-range
warning, and a domain-rule warning, using a model trained in a fixture rather than the real artifact.

### F15. Streamlit app, single prediction — `app/streamlit_app.py`
- [ ] Load the model and metadata with `@st.cache_resource`. If they're missing, show the message "run
  `breast-cancer train --final`".
- [ ] Add a sidebar disclaimer: an educational project, not a medical device, not for diagnosis.
- [ ] One `st.number_input` per **input feature in the metadata**, grouped in Mean / SE / Worst expanders (if a
  reduced model is chosen, this is just a short form). Defaults are the training medians, and the step size suits
  each feature's scale.
- [ ] Output: the predicted label, the malignancy probability, the threshold used, and any validation warnings.

**Done when:** `uv run streamlit run app/streamlit_app.py` produces a prediction from the default inputs, and
entering worst < mean shows a warning.

### F16. Streamlit app, batch prediction and explanations
- [ ] Batch mode: upload a CSV (the raw `wdbc.data` layout with a header, or any column order), validate it, show
  per-row errors and warnings, show a results table, and offer the results as a download.
- [ ] Add "Load example" buttons that fill the form with a typical benign and a typical malignant **test-set**
  sample.
- [ ] Explanations: the grouped permutation importance chart. For a linear model, the per-feature contributions
  (coefficient × transformed value) for the current input.
- [ ] Show the model card: the model type, the test metrics with CIs, and the training date, all from the metadata.

**Done when:** an uploaded CSV with shuffled columns works, and one with a missing column shows a clear error.

---

## Phase 5 — Wrap-up

### F17. Docs, quality and release hygiene
- [ ] Update the README with the project summary, the data profile highlights, a quickstart
  (`download → prepare → train → app`), and a results table.
- [ ] Update `docs/usage.md` and `docs/api.md`, and add an entry to `CHANGELOG/unreleased.md`.
- [ ] Add a `just pipeline` recipe that runs download, prepare and train end to end, and a `just app` recipe.
- [ ] CI is green: ruff, ty, and pytest on 3.12–3.14, with coverage at or above the threshold.

**Done when:** a fresh clone can follow the README to a running app.

---

## Feature dependency summary

```
F0 → F1 → F2 → F3 → F4 → F5 (EDA) → F6 → F7 → F8 → F9 → F10 → F11 → F12 → F13 → F14 → F15 → F16 → F17
```
