"""Same LightGBM settings as the baseline, with within-breath history added.

New columns, computed inside each breath_id only:
- u_in lagged 1, 2, and 3 steps (missing early steps filled with 0)
- cumulative sum of u_in
- step index 0-79
R and C are passed to LightGBM as categories. The locked validation breaths are not used for training.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from validation import inspiratory_mae, load_breath_ids

ROOT = Path(__file__).resolve().parent
RAW = ["R", "C", "time_step", "u_in", "u_out"]
EXTRA = ["u_in_lag1", "u_in_lag2", "u_in_lag3", "u_in_cumsum", "step"]
FEATURES = RAW + EXTRA
RESULT_PATH = ROOT / "experiments" / "features.json"


def add_breath_history(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["breath_id", "time_step"], kind="mergesort")
    grouped = frame.groupby("breath_id", sort=False)["u_in"]
    frame["u_in_lag1"] = grouped.shift(1).fillna(0.0)
    frame["u_in_lag2"] = grouped.shift(2).fillna(0.0)
    frame["u_in_lag3"] = grouped.shift(3).fillna(0.0)
    frame["u_in_cumsum"] = grouped.cumsum()
    frame["step"] = frame.groupby("breath_id", sort=False).cumcount()
    frame["R"] = frame["R"].astype("category")
    frame["C"] = frame["C"].astype("category")
    return frame


def slice_mae(y_true, y_pred, mask) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "train.csv", usecols=["breath_id", *RAW, "pressure"])
    frame = add_breath_history(frame)
    val_ids = set(load_breath_ids("val").tolist())
    is_val = frame["breath_id"].isin(val_ids)
    train = frame.loc[~is_val]
    valid = frame.loc[is_val]

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    started = time.perf_counter()
    model.fit(train[FEATURES], train["pressure"], categorical_feature=["R", "C"])
    pred = model.predict(valid[FEATURES])
    elapsed = time.perf_counter() - started

    y = valid["pressure"].to_numpy()
    u_out = valid["u_out"].to_numpy()
    inspiratory = u_out == 0
    mae = inspiratory_mae(y, pred, u_out)
    step = valid["step"].to_numpy()
    early_mae = slice_mae(y, pred, inspiratory & (step < 40))
    late_mae = slice_mae(y, pred, inspiratory & (step >= 40))
    r_values = valid["R"].astype(str).to_numpy()
    c_values = valid["C"].astype(str).to_numpy()
    by_r = {
        level: round(slice_mae(y, pred, inspiratory & (r_values == level)), 4)
        for level in sorted(set(r_values), key=float)
    }
    by_c = {
        level: round(slice_mae(y, pred, inspiratory & (c_values == level)), 4)
        for level in sorted(set(c_values), key=float)
    }

    result = {
        "model": "lightgbm",
        "change": "within-breath u_in lags 1-3, u_in cumsum, step index; R and C categorical",
        "baseline_val_mae": 3.998002671200387,
        "features": FEATURES,
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 63,
        "objective": "regression_l1",
        "val_mae": mae,
        "val_mae_early_steps_0_39": early_mae,
        "val_mae_late_steps_40_79": None if np.isnan(late_mae) else late_mae,
        "note": "No validation row with u_out == 0 has step >= 40, so the late-step MAE is undefined. Every scored row is in the first half of the breath.",
        "val_mae_by_R": by_r,
        "val_mae_by_C": by_c,
        "runtime_seconds": round(elapsed, 1),
    }
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
