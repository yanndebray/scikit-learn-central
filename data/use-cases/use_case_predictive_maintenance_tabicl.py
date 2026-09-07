# %% [markdown]
#
# # Turbofan Remaining Useful Life: gradient boosting vs. a tabular foundation model.
#
# This notebook is an alternative take on **Aircraft Engine Predictive
# Maintenance**. The [first version](use_case_predictive_maintenance.ipynb)
# compares three classical scikit-learn regressors on NASA's C-MAPSS data. Here
# we keep the winner of that comparison — **`HistGradientBoostingRegressor`** —
# and put it head to head with **[TabICL](https://github.com/soda-inria/tabicl)**,
# a *tabular foundation model* that predicts by **in-context learning**: instead
# of fitting parameters to your table, it loads a pre-trained transformer and
# reads the training rows as context at prediction time.
#
# ## Environment setup
#
# TabICL runs on **PyTorch** and downloads its checkpoint from the Hugging Face
# Hub on first use, so — unlike the other use cases in this catalog — **this
# notebook does not run in JupyterLite** (the in-browser Pyodide kernel has no
# torch). Run it in a local Python environment:
#
# ```bash
# pip install skore skrub tabicl matplotlib
# ```
#
# The cells below degrade gracefully: if `tabicl` is missing, the gradient
# boosting baseline still runs and the TabICL comparisons are skipped.

# %%
# %pip install skore skrub
# TabICL pulls in PyTorch, which the JupyterLite kernel cannot install. Locally, run:
# # %pip install tabicl

# %%
from importlib.util import find_spec

HAS_TABICL = find_spec("tabicl") is not None
print("TabICL available:", HAS_TABICL)

# %% [markdown]
#
# ## Data loading
#
# The bundled CSVs are subset **FD001** of NASA's C-MAPSS turbofan degradation
# simulator: 100 engines run from a healthy start to failure, one row per
# operational **cycle**, with three operating settings and 21 noisy sensor
# channels. The test set truncates each engine before failure and ships the true
# Remaining Useful Life (RUL) at that truncation point.

# %%
import pandas as pd

train = pd.read_csv("datasets/cmapss_fd001_train.csv")
test = pd.read_csv("datasets/cmapss_fd001_test.csv")
rul = pd.read_csv("datasets/cmapss_fd001_rul.csv")

print(f"train: {train.shape}, test: {test.shape}, engines (train): {train['unit'].nunique()}")
train.head()

# %% [markdown]
#
# ## Target and features
#
# Same preparation as the first notebook, so the two are directly comparable:
# RUL is `max_cycle(unit) - cycle` **clipped at 125 cycles** (far from failure,
# every sensor still looks healthy, so the extra range is mostly noise); columns
# that never move under FD001's single operating condition are dropped; and each
# surviving sensor gains per-engine rolling means and standard deviations over
# the last 5 and 30 cycles.

# %%
RUL_CAP = 125

max_cycle = train.groupby("unit")["cycle"].transform("max")
train["rul"] = (max_cycle - train["cycle"]).clip(upper=RUL_CAP)

test_last = test.groupby("unit")["cycle"].transform("max")
test = test.merge(rul, on="unit", how="left")
test["rul"] = (test["rul"] + (test_last - test["cycle"])).clip(upper=RUL_CAP)

sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
setting_cols = [c for c in train.columns if c.startswith("op_setting_")]
constant_cols = [c for c in sensor_cols + setting_cols if train[c].nunique() == 1]
active_sensors = [c for c in sensor_cols if c not in constant_cols]


def add_rolling_features(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    df = df.sort_values(["unit", "cycle"]).copy()
    grouped = df.groupby("unit")[sensors]
    parts = [df]
    for window in (5, 30):
        roll = grouped.rolling(window=window, min_periods=1)
        parts.append(roll.mean().reset_index(level=0, drop=True).add_suffix(f"_mean_{window}"))
        parts.append(
            roll.std().fillna(0.0).reset_index(level=0, drop=True).add_suffix(f"_std_{window}")
        )
    return pd.concat(parts, axis=1)


train_feat = add_rolling_features(train, active_sensors)
test_feat = add_rolling_features(test, active_sensors)

drop_cols = ["unit", "cycle", "rul"] + constant_cols
X_full = train_feat.drop(columns=drop_cols)
y_full = train_feat["rul"]
groups_full = train_feat["unit"]
print(f"X_full: {X_full.shape} ({len(active_sensors)} active sensors → rolling features)")

# %% [markdown]
#
# ## Budgeting the in-context learner
#
# A gradient boosted tree compresses the training set into split thresholds
# once, then predicts in microseconds. TabICL does the opposite: `fit` is nearly
# free (it only stores and normalizes the table), and **every prediction attends
# over the whole training set**, so inference cost grows with
# *context rows × query rows × columns*.
#
# Consecutive cycles of one engine are nearly identical anyway, so we keep
# **one cycle in eight** — roughly 25 snapshots per engine instead of 200.
#
# This is not only a concession to the budget. Feeding TabICL all 20,631 rows
# instead of these 2,536 makes it measurably *worse* (RMSE around 14.5 against
# 12.3 on the same held-out engines) while costing eight times more: near
# duplicate rows add redundancy, not information, and TabICLv2 was pre-trained
# on tables of 300 to 60,000 samples, so a context stuffed with copies is off
# its home turf. The tree is indifferent to the same trim. Row subsampling is a
# real hyperparameter for an in-context learner, not a shortcut.

# %%
CYCLE_STRIDE = 8

subsample = train_feat["cycle"] % CYCLE_STRIDE == 0
X = X_full[subsample]
y = y_full[subsample]
groups = groups_full[subsample]
print(f"X: {X.shape} (from {X_full.shape[0]} rows, stride {CYCLE_STRIDE})")

# %% [markdown]
#
# One more practical note: we pin TabICL to `device="cpu"`. The model also runs
# on CUDA and Apple MPS, but a few thousand context rows over ~80 columns
# already overruns the memory budget of a laptop GPU, and the offloading that
# kicks in to rescue it is an order of magnitude *slower* than plain CPU
# inference. On a datacentre GPU, drop the argument and let TabICL pick the
# device itself.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

import skrub

estimators = {"hist_gbt": skrub.tabular_pipeline(HistGradientBoostingRegressor(random_state=42))}

if HAS_TABICL:
    from tabicl import TabICLRegressor

    estimators["tabicl"] = TabICLRegressor(n_estimators=4, device="cpu", random_state=42)

list(estimators)

# %% [markdown]
#
# TabICL needs no preprocessing wrapper: it is scikit-learn compliant and
# normalizes each column internally, which is part of the pitch — no imputation,
# no scaling, no hyperparameter search. `n_estimators=4` averages four
# feature-shuffled passes instead of the default eight: on this data that runs
# in half the time for a percent or two of RMSE, which is the trade you want
# while exploring.
#
# ## Comparing both models with `skore.evaluate`
#
# **`skore.evaluate`** takes a **dict of named estimators** plus a **`splitter`**
# and returns a single comparison report — metrics, checks and timings
# aggregated across models and folds.
#
# Cycles from one engine are highly correlated, so an engine leaking across the
# split would inflate the scores. We pre-compute `GroupKFold` splits keyed on
# `unit` and hand them to `skore.evaluate` as the splitter, keeping each engine
# entirely inside one fold.

# %%
from sklearn.model_selection import GroupKFold

import skore

splits = list(GroupKFold(n_splits=4).split(X, y, groups))

# A dict needs at least two entries to build a comparison, so fall back to the
# single baseline report when TabICL is unavailable (e.g. under JupyterLite).
to_evaluate = estimators if len(estimators) > 1 else next(iter(estimators.values()))

cv_report = skore.evaluate(to_evaluate, X, y, splitter=splits)
cv_report

# %% [markdown]
#
# The same diagnostic **checks** run on every cross-validated model, so
# over- or underfitting is flagged per estimator.
#
# We pass `fast_mode=True` here, and that is not a detail: some checks (the
# "useless features" one, for instance) rank features by **permutation
# importance**, which re-predicts the whole validation set once per shuffled
# column. With ~80 columns that is fine for a tree and pathological for an
# in-context learner — hundreds of full-context passes. `fast_mode=True` skips
# the checks marked slow and keeps the cheap ones.

# %%
cv_report.checks.summarize(fast_mode=True)

# %% [markdown]
#
# **Metrics** are aggregated across folds, which is what we need to rank the two
# approaches on RMSE and R² — and, just as important here, to see the cost side
# of the trade-off in the fit and predict times.

# %%
cv_report.metrics.summarize().frame()

# %% [markdown]
#
# Three things stand out, and only one of them was predictable:
#
# - **TabICL is the more accurate model here**, by a clear margin: RMSE around
#   13.7 against 16.5, R² 0.89 against 0.84 — with no tuning, no feature
#   selection and no preprocessing.
# - **It is also far steadier.** Its RMSE varies by ±0.3 across folds where the
#   tree swings by ±0.9, which matters when the fold *is* a group of engines you
#   have never seen.
# - **It pays for that at inference.** Look at `predict_time`: seconds per fold
#   against milliseconds for the tree, well over a hundredfold. `fit_time` is
#   the reverse, and almost meaningless — TabICL's `fit` only stores the table.
#
# (Ignore the `mape` row: RUL reaches 0 at failure, so a percentage error
# divides by zero. RMSE and MAE are the metrics to read for this target.)
#
# ## How many broken engines do you need?
#
# The comparison above says which model to serve; it does not say how much data
# each one needs to get there. Run-to-failure data is expensive — every training
# engine is one machine somebody let break — so the useful question is where
# each model lands with a fleet of 5 engines instead of 80.
#
# We hold out 20 engines, then sweep the number of engines the models may learn
# from. A single draw of "the first 5 engines" is noisy enough to invent
# crossovers that are not there, so each budget is repeated over three random
# draws and we report the spread.

# %%
import numpy as np
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error

units = np.array(sorted(groups.unique()))
holdout_units, pool_units = units[:20], units[20:]

holdout = groups.isin(holdout_units)
X_holdout, y_holdout = X[holdout], y[holdout]

ENGINE_BUDGETS = [5, 10, 20, 40, len(pool_units)]
curve = []

for seed in (0, 1, 2):
    order = np.random.default_rng(seed).permutation(pool_units)
    for budget in ENGINE_BUDGETS:
        mask = groups.isin(order[:budget])
        for name, estimator in estimators.items():
            model = clone(estimator).fit(X[mask], y[mask])
            rmse = root_mean_squared_error(y_holdout, model.predict(X_holdout))
            curve.append(
                {"seed": seed, "engines": budget, "rows": int(mask.sum()),
                 "model": name, "rmse": rmse}
            )
    print(f"draw {seed} done", flush=True)

curve = pd.DataFrame(curve)
curve.pivot_table(index="engines", columns="model", values="rmse", aggfunc="mean")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
for name, part in curve.groupby("model"):
    stats = part.groupby("engines")["rmse"].agg(["mean", "min", "max"])
    line, = ax.plot(stats.index, stats["mean"], marker="o", label=name)
    ax.fill_between(stats.index, stats["min"], stats["max"], alpha=0.15, color=line.get_color())
ax.set_xscale("log")
ax.set_xticks(ENGINE_BUDGETS)
ax.set_xticklabels(ENGINE_BUDGETS)
ax.set_xlabel("Training engines (log scale)")
ax.set_ylabel("RMSE on 20 held-out engines (cycles)")
ax.set_title("Data efficiency: RMSE vs. number of run-to-failure engines")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()

# %% [markdown]
#
# TabICL is ahead at every budget, and the gap is widest in the middle — around
# 20 to 40 engines (roughly 500 to 1 000 rows) it saves about two cycles of
# RMSE over the tree. What it does *not* do is rescue the bottom of the curve:
# with 5 engines, about 120 rows, the two models sit inside each other's spread
# both are poor. That is the honest version of the "foundation models win on
# small data" claim — TabICLv2's pre-training starts at 300 samples, and below
# that you are outside the range it was built for. The prior helps once you have
# a few hundred informative rows; it does not conjure a degradation model out of
# five engines.
#
# ## Predicted vs. true RUL trajectory
#
# The plot engineers actually ask for: follow one held-out **test** engine
# toward failure and overlay each model's prediction on the ground truth. These
# are unseen units, not held-out cycles of the training engines.

# %%
X_test = test_feat.drop(columns=drop_cols)
test_engine = test_feat["unit"] == 1
X_engine = X_test[test_engine]
cycles = test_feat.loc[test_engine, "cycle"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cycles, test_feat.loc[test_engine, "rul"], label="True RUL", linewidth=2, color="black")

for name, estimator in estimators.items():
    model = clone(estimator).fit(X, y)
    ax.plot(cycles, model.predict(X_engine), linestyle="--", label=f"{name} prediction")

ax.axhline(RUL_CAP, color="grey", linestyle=":", label=f"Cap = {RUL_CAP}")
ax.set_xlabel("Cycle")
ax.set_ylabel("Remaining Useful Life (cycles)")
ax.set_title("Engine #1 - predicted vs. true RUL")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
#
# ## Takeaways
#
# - **The foundation model won on accuracy, and it was not close.** Around 13.7
#   RMSE against 16.5, steadier across folds, with zero tuning and zero feature
#   selection. If you are used to reaching for gradient boosting by reflex on
#   tabular data, it is worth spending one cell finding out what the pre-trained
#   model does on your table.
# - **The tree won on economics, by two orders of magnitude.** Milliseconds
#   against seconds per prediction batch, on CPU, with no checkpoint to ship.
#   For fleet-wide nightly scoring that can settle the argument on its own —
#   accuracy is not the only axis.
# - **Cost scales with the context, not the fit.** TabICL's `fit` is free and
#   its `predict` grows with the training set — the inverse of every classical
#   estimator. Budget for it: subsample rows, trim columns, lower
#   `n_estimators`, or set `kv_cache=True` when scoring repeatedly against a
#   fixed training set.
# - **Redundant rows actively hurt it.** Eight times more near-duplicate cycles
#   made TabICL worse *and* slower. Deciding which rows go into the context is
#   part of using one of these models; there is no "just throw the table at it".
# - **Tooling built for estimators still applies — with care.** Any
#   scikit-learn compliant model drops into the same `skore.evaluate`
#   comparison, with the same grouped splits, checks and timings. But
#   diagnostics that re-predict in a loop — permutation importance, learning
#   curves — are priced per prediction, and predictions are no longer free.
