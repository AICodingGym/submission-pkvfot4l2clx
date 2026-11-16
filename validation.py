"""Fixed breath-level holdout and the inspiratory MAE used for every experiment.

The split is by breath_id, not by row. Rows from the same breath stay together.
The metric matches the leaderboard: mean absolute error on rows where u_out == 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
VAL_FRACTION = 0.20
ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "splits"
VAL_IDS_PATH = SPLIT_DIR / "val_breath_ids.csv"
TRAIN_IDS_PATH = SPLIT_DIR / "train_breath_ids.csv"
INFO_PATH = SPLIT_DIR / "split_info.json"


def inspiratory_mae(y_true, y_pred, u_out) -> float:
    """Mean absolute error on inspiratory rows only (u_out == 0)."""
    mask = np.asarray(u_out) == 0
    if not np.any(mask):
        raise ValueError("no inspiratory rows to score")
    error = np.abs(np.asarray(y_true, dtype=float)[mask] - np.asarray(y_pred, dtype=float)[mask])
    return float(error.mean())


def make_breath_split(breath_ids, val_fraction: float = VAL_FRACTION, seed: int = SEED):
    """Return sorted train and validation breath ids. The same seed always rebuilds this split."""
    breaths = np.unique(np.asarray(breath_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(breaths))
    shuffled = breaths[order]
    n_val = int(round(len(breaths) * val_fraction))
    val_ids = np.sort(shuffled[:n_val])
    train_ids = np.sort(shuffled[n_val:])
    return train_ids, val_ids


def load_breath_ids(kind: str) -> np.ndarray:
    path = TRAIN_IDS_PATH if kind == "train" else VAL_IDS_PATH
    return pd.read_csv(path)["breath_id"].to_numpy()


def main() -> None:
    usecols = ["breath_id", "u_out"]
    frame = pd.read_csv(ROOT / "data" / "train.csv", usecols=usecols)
    train_ids, val_ids = make_breath_split(frame["breath_id"])

    overlap = np.intersect1d(train_ids, val_ids)
    if overlap.size:
        raise RuntimeError(f"split overlap: {overlap.size} breaths")

    SPLIT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"breath_id": train_ids}).to_csv(TRAIN_IDS_PATH, index=False)
    pd.DataFrame({"breath_id": val_ids}).to_csv(VAL_IDS_PATH, index=False)

    is_val = frame["breath_id"].isin(val_ids)
    info = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "n_breaths": int(train_ids.size + val_ids.size),
        "n_train_breaths": int(train_ids.size),
        "n_val_breaths": int(val_ids.size),
        "n_train_rows": int((~is_val).sum()),
        "n_val_rows": int(is_val.sum()),
        "n_val_inspiratory_rows": int(((is_val) & (frame["u_out"] == 0)).sum()),
        "metric": "mean absolute error where u_out == 0",
        "rows_per_breath_min": int(frame.groupby("breath_id").size().min()),
        "rows_per_breath_max": int(frame.groupby("breath_id").size().max()),
    }
    INFO_PATH.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
