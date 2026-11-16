"""LightGBM baseline on the raw columns, scored on the locked breath holdout.

Features are only the columns already in the file: R, C, time_step, u_in, u_out.
No lags or other engineered features. The validation breath ids are not used for training.
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
FEATURES = ["R", "C", "time_step", "u_in", "u_out"]
RESULT_PATH = ROOT / "experiments" / "baseline.json"


def main() -> None:
    columns = ["breath_id", *FEATURES, "pressure"]
    frame = pd.read_csv(ROOT / "data" / "train.csv", usecols=columns)
    val_ids = load_breath_ids("val")
    is_val = frame["breath_id"].isin(val_ids)
    train = frame.loc[~is_val]
    valid = frame.loc[is_val]

    # A constant prediction shows what the raw-column model has to beat.
    inspiratory = train["u_out"].to_numpy() == 0
    constant = float(train.loc[inspiratory, "pressure"].mean())
    constant_pred = np.full(len(valid), constant)
    constant_mae = inspiratory_mae(valid["pressure"], constant_pred, valid["u_out"])

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
    model.fit(train[FEATURES], train["pressure"])
    pred = model.predict(valid[FEATURES])
    elapsed = time.perf_counter() - started
    mae = inspiratory_mae(valid["pressure"], pred, valid["u_out"])

    result = {
        "model": "lightgbm",
        "features": FEATURES,
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 63,
        "objective": "regression_l1",
        "constant_inspiratory_pressure": constant,
        "constant_val_mae": constant_mae,
        "val_mae": mae,
        "runtime_seconds": round(elapsed, 1),
        "n_train_rows": int(len(train)),
        "n_val_rows": int(len(valid)),
    }
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
