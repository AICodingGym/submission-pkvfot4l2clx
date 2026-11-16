"""Same LightGBM settings and breath split, with a longer within-breath u_in history.

The previous model used lags 1-3. This one uses lags 1-20, which covers half of the
inspiratory phase (every scored row is in the first 40 steps), plus the one-step change
in u_in. Cumulative sum, step index, and categorical R and C stay as before.
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
LAGS = list(range(1, 21))
EXTRA = [f"u_in_lag{k}" for k in LAGS] + ["u_in_diff1", "u_in_cumsum", "step"]
FEATURES = RAW + EXTRA
RESULT_PATH = ROOT / "experiments" / "longer_history.json"
PREVIOUS_MAE = 1.3745848510811398


def add_breath_history(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["breath_id", "time_step"], kind="mergesort")
    counts = frame.groupby("breath_id", sort=False).size()
    if not bool((counts.to_numpy() == 80).all()):
        raise RuntimeError("expected every breath to have 80 rows")

    u_in = frame["u_in"].to_numpy(dtype=np.float32).reshape(-1, 80)
    n_breaths = u_in.shape[0]
    for lag in LAGS:
        shifted = np.zeros_like(u_in)
        shifted[:, lag:] = u_in[:, :-lag]
        frame[f"u_in_lag{lag}"] = shifted.reshape(-1)
    frame["u_in_diff1"] = frame["u_in"].to_numpy(dtype=np.float32) - frame["u_in_lag1"].to_numpy()
    frame["u_in_cumsum"] = np.cumsum(u_in, axis=1).reshape(-1)
    frame["step"] = np.tile(np.arange(80, dtype=np.int16), n_breaths)
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
    is_val = frame["breath_id"].isin(set(load_breath_ids("val").tolist()))
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
    inspiratory = valid["u_out"].to_numpy() == 0
    mae = inspiratory_mae(y, pred, valid["u_out"])
    r_values = valid["R"].astype(str).to_numpy()
    c_values = valid["C"].astype(str).to_numpy()
    importance = {
        name: int(score)
        for name, score in sorted(zip(FEATURES, model.feature_importances_), key=lambda item: -item[1])
    }

    result = {
        "model": "lightgbm",
        "change": "extend u_in lags from 1-3 to 1-20 and add the one-step change u_in_diff1",
        "previous_val_mae": PREVIOUS_MAE,
        "val_mae": mae,
        "mae_change": mae - PREVIOUS_MAE,
        "val_mae_by_R": {
            level: round(slice_mae(y, pred, inspiratory & (r_values == level)), 4)
            for level in sorted(set(r_values), key=float)
        },
        "val_mae_by_C": {
            level: round(slice_mae(y, pred, inspiratory & (c_values == level)), 4)
            for level in sorted(set(c_values), key=float)
        },
        "feature_importance_split": importance,
        "n_estimators": 200,
        "runtime_seconds": round(elapsed, 1),
    }
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
