# %% [markdown]
#
# # Turbofan Remaining Useful Life with skore, skrub, and scikit-learn.
#
# ## Environment setup
#
# We need to install some extra dependencies for this notebook if needed (when
# running jupyterlite).

# %%
# %pip install skore skrub ipywidgets

# %% [markdown]
#
# ## Data loading
#
# The bundled CSVs come from **NASA's C-MAPSS turbofan engine degradation
# simulator** (subset **FD001**: sea-level operating condition, single fault
# mode — HPC degradation).
#
# Each row is one operational **cycle** of one **engine** (`unit`). The training
# set follows every engine from a healthy start to the cycle on which it fails;
# the test set truncates each engine at some point before failure and ships a
# separate file with the true Remaining Useful Life (RUL) at that truncation.
#
# Columns:
#
# - `unit`, `cycle` — engine id and 1-indexed cycle number.
# - `op_setting_1..3` — three operating conditions (altitude, Mach, TRA).
# - `sensor_01..21` — 21 simulator outputs (temperatures, pressures, spool
#   speeds, bleeds, fuel-air ratio, …) with realistic sensor noise.
#
# The files are vendored under `datasets/` so this notebook runs offline in
# JupyterLite.

# %%
import pandas as pd

train = pd.read_csv("datasets/cmapss_fd001_train.csv")
test = pd.read_csv("datasets/cmapss_fd001_test.csv")
rul = pd.read_csv("datasets/cmapss_fd001_rul.csv")

print(f"train: {train.shape}, test: {test.shape}, engines (train): {train['unit'].nunique()}")
train.head()

# %% [markdown]
#
# ## Target: Remaining Useful Life
#
# In the training set we know each engine's failure cycle, so RUL at any cycle
# is just `max_cycle(unit) - cycle`. Following the standard C-MAPSS convention,
# we **clip the target at 125 cycles**: an engine that is still very far from
# failure looks healthy on every sensor, so trying to distinguish "300 cycles
# left" from "200 cycles left" is mostly noise. Clipping focuses the model on
# the degradation region where the signal lives.

# %%
RUL_CAP = 125

max_cycle = train.groupby("unit")["cycle"].transform("max")
train["rul"] = (max_cycle - train["cycle"]).clip(upper=RUL_CAP)

# For the test set the true RUL is given at the *last* observed cycle of each
# engine, so we extend it backwards along each unit's cycle history.
test_last = test.groupby("unit")["cycle"].transform("max")
test = test.merge(rul, on="unit", how="left")
test["rul"] = (test["rul"] + (test_last - test["cycle"])).clip(upper=RUL_CAP)

train[["unit", "cycle", "rul"]].head()

# %% [markdown]
#
# ## Feature engineering
#
# Several FD001 sensors are constants under the single operating condition and
# carry no information, so we drop them. We then add a small set of
# **per-engine rolling statistics** over the last 5 and 30 cycles for every
# remaining sensor — a lightweight tabular stand-in for the windowed features a
# time-series model would learn.

# %%
sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
constant_cols = [c for c in sensor_cols if train[c].nunique() == 1]
active_sensors = [c for c in sensor_cols if c not in constant_cols]
print(f"dropped {len(constant_cols)} constant sensors, kept {len(active_sensors)}")


def add_rolling_features(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    df = df.sort_values(["unit", "cycle"]).copy()
    grouped = df.groupby("unit")[sensors]
    for window in (5, 30):
        roll = grouped.rolling(window=window, min_periods=1)
        mean = roll.mean().reset_index(level=0, drop=True).add_suffix(f"_mean_{window}")
        std = roll.std().fillna(0.0).reset_index(level=0, drop=True).add_suffix(f"_std_{window}")
        df = pd.concat([df, mean, std], axis=1)
    return df


train_feat = add_rolling_features(train, active_sensors)
test_feat = add_rolling_features(test, active_sensors)

drop_cols = ["unit", "cycle", "rul"] + constant_cols
X = train_feat.drop(columns=drop_cols)
y = train_feat["rul"]
groups = train_feat["unit"]
print(f"X: {X.shape}")

# %% [markdown]
#
# ## Comparing three regressors with `skore.evaluate`
#
# **`skore.evaluate`** takes a **dict of named estimators** and a **`splitter`**,
# runs cross-validation for every entry, and returns one **comparison report**
# (metrics and checks aggregated across models and folds) — no hand-written
# split loops.
#
# Because cycles from the same engine are highly correlated, leaking part of an
# engine into both train and validation would inflate scores. We pre-compute
# splits from `GroupKFold` keyed on `unit` and pass them as the `splitter`, so
# each engine lives entirely in one fold.
#
# Each pipeline is wrapped with **`skrub.tabular_pipeline(...)`** for consistent
# preprocessing (imputation + scaling for the linear model, passthrough for the
# trees) so the comparison reflects the estimators, not the preprocessing.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import skore
import skrub

splits = list(GroupKFold(n_splits=5).split(X, y, groups))

cv_report = skore.evaluate(
    {
        "ridge": skrub.tabular_pipeline(Ridge()),
        "hist_gbt": skrub.tabular_pipeline(HistGradientBoostingRegressor(random_state=42)),
        "forest": skrub.tabular_pipeline(RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)),
    },
    X,
    y,
    splitter=splits,
)
cv_report

# %% [markdown]
#
# The comparison report runs the same diagnostic **checks** on every
# cross-validated model. Summarize them to see issues such as over- or
# underfitting flagged per estimator.

# %%
cv_report.checks.summarize()

# %% [markdown]
#
# **Metrics** are aggregated across folds and models so you can rank approaches
# (for example RMSE or R²) without writing your own aggregation code.

# %%
cv_report.metrics.summarize().frame()

# %% [markdown]
#
# Histogram gradient boosting typically wins this comparison: the linear model
# underfits the non-linear degradation curve, the random forest matches it on
# accuracy but is heavier, and the histogram tree trains in seconds even on the
# full FD001 cycle table.
#
# ## Predicted vs true RUL trajectory
#
# A picture engineers expect to see: for one engine in the held-out test set,
# plot the predicted RUL alongside the ground-truth RUL as the engine ages
# toward failure.

# %%
import matplotlib.pyplot as plt

best = HistGradientBoostingRegressor(random_state=42).fit(X, y)
X_test = test_feat.drop(columns=drop_cols)
test_feat["rul_pred"] = best.predict(X_test)

engine = test_feat[test_feat["unit"] == 1]
plt.figure(figsize=(8, 4))
plt.plot(engine["cycle"], engine["rul"], label="True RUL", linewidth=2)
plt.plot(engine["cycle"], engine["rul_pred"], label="Predicted RUL", linestyle="--")
plt.axhline(RUL_CAP, color="grey", linestyle=":", label=f"Cap = {RUL_CAP}")
plt.xlabel("Cycle")
plt.ylabel("Remaining Useful Life (cycles)")
plt.title("Engine #1 — predicted vs. true RUL")
plt.legend()
plt.grid(True)
plt.show()
