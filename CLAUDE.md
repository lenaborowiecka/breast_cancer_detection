# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project goal

End-to-end **supervised binary classification** of breast tumours as malignant or benign, using the
[UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
The final deliverable is a **Streamlit app** that loads a trained, persisted model and serves predictions.

Pipeline: data ingestion → validation/EDA → preprocessing → training + model selection → evaluation → model
persistence → Streamlit inference app.

## Dataset facts

- 569 samples, no missing values. Class balance: 357 benign (B), 212 malignant (M) — mildly imbalanced.
- Columns: `ID` (drop it, never a feature), `Diagnosis` (target, `M`/`B`), and 30 real-valued features.
- The 30 features are 10 cell-nucleus measurements (radius, texture, perimeter, area, smoothness, compactness,
  concavity, concave points, symmetry, fractal dimension), each given as **mean**, **standard error** and
  **worst** (mean of the three largest values).
- Many features are highly correlated (radius/perimeter/area especially) — relevant for linear models and
  feature importance interpretation.
- Load with `ucimlrepo` (`fetch_ucirepo(id=17)`). `sklearn.datasets.load_breast_cancer` is the same data and
  works offline — use it in tests so they never hit the network.

## Conventions for the ML work

- **Target encoding:** malignant = `1` (positive class), benign = `0`. Keep this consistent everywhere.
- **No leakage:** do the train/test split first (stratified, fixed `random_state=42`), and put all
  preprocessing (scaling, feature selection) inside a single `sklearn.pipeline.Pipeline` fitted only on
  training data. Tune with stratified K-fold CV on the training set; touch the test set once, at the end.
- **Metrics:** this is a medical screening problem, so false negatives (missed malignancies) are the costly
  error. Report recall (sensitivity) for malignant, precision, F1, ROC-AUC, PR-AUC and the confusion matrix
  — not accuracy alone. If the decision threshold is tuned away from 0.5, store it with the model.
- **Models:** start with a logistic regression baseline, then compare tree ensembles / SVM etc. Prefer the
  simpler model when scores are close.
- **Reproducibility:** fix random seeds; all paths and hyperparameters live in one config module, not
  scattered literals.
- **Persistence:** save the whole fitted pipeline with `joblib` to `models/model.joblib`, plus
  `models/metadata.json` containing feature names *in training order*, class labels, threshold, test metrics,
  training date, and the scikit-learn version. The app must use the feature order from the metadata.
- Notebooks (`notebooks/`) are for exploration only; anything the app or training depends on must live in the
  package and be importable and tested.

## Streamlit app

- Lives in `app/streamlit_app.py`; run with `uv run streamlit run app/streamlit_app.py`.
- Load the model once with `@st.cache_resource`. If `models/model.joblib` is missing, show a clear message
  telling the user to run training first — do not train inside the app.
- Inputs: 30 numeric fields grouped by mean / SE / worst (sensible defaults, e.g. training medians, and
  min/max from the data), plus optional CSV upload for batch prediction with column validation.
- Output: predicted class and malignancy probability; optionally which features drove the prediction.
- Inference logic (input validation, building the feature frame, calling the model) belongs in the package,
  not in the Streamlit script, so it can be unit-tested.
- Always show a disclaimer: educational project, not a medical device, not for diagnosis.

## Repository layout

Scaffolded from `audreyfeldroy/cookiecutter-pypackage`. **Note:** the template generated the package as
`src/src/` (import name `src`). Before adding ML code, rename it to `src/breast_cancer_detection/` and update
`pyproject.toml` (`[project.scripts]`, `[tool.hatch.build.targets.wheel] packages`), the tests, and `docs/`.

Target layout:

```
src/breast_cancer_detection/
  config.py       # paths, seeds, hyperparameters
  data.py         # download/load dataset, cache to data/raw/
  features.py     # preprocessing / pipeline construction
  train.py        # training, CV, model selection, persistence
  evaluate.py     # metrics, plots, report
  predict.py      # load model + metadata, validate input, predict
  cli.py          # Typer CLI: `download`, `train`, `evaluate`, `predict`
app/streamlit_app.py
notebooks/        # EDA only
data/             # gitignored
models/           # gitignored artifacts (model.joblib, metadata.json)
reports/          # metrics and figures
tests/
```

Add `data/`, `models/*.joblib` and notebook checkpoints to `.gitignore`. Add runtime deps
(`scikit-learn`, `pandas`, `ucimlrepo`, `joblib`, `streamlit`, plotting libs) with `uv add`; dev-only tools
with `uv add --group <group>`.

## Commands

Tooling is `uv` + `just`. Python `>=3.12`; the justfile uses 3.14 for checks.

```bash
uv sync                      # install deps (dev group included by default)
just fix                     # ruff format + ruff check --fix
just check                   # format check, lint, ty type-check, pytest  (run before finishing a task)
just test tests/test_x.py    # run specific tests
just coverage                # coverage across 3.12/3.13/3.14 (fail_under = 50)
just docs-serve              # zensical docs on :8000
uv run streamlit run app/streamlit_app.py
```

CI (`.github/workflows/ci.yml`) runs ruff, `ty check` and pytest on 3.12–3.14; keep it green.

## Code style

- Ruff, line length 120, rules E/W/F/I/B/UP. Type hints on all public functions (checked by `ty`).
- Docstrings on public functions (they feed `mkdocstrings` API docs).
- Tests: use small fixtures / `load_breast_cancer`, never network calls or the real `models/` artifact. Cover
  data loading/schema, the pipeline fitting end-to-end on a tiny sample, save/load round-trip, and the
  prediction function's input validation.
- Record user-facing changes in `CHANGELOG/unreleased.md`.
