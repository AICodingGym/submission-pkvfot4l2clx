"""Build the trial 9 prediction file and recheck the saved weights on the holdout.

Predictions are written in the original test-file id order. The saved feature
mean and standard deviation are reused, so this matches training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_bilstm_4x256_features import (
    ALL_FEATURES,
    WEIGHTS_PATH,
    BiLSTM4x256,
    add_features,
)
from experiment_forward_lstm import BATCH_SIZE, ROOT, load_breaths
from experiment_forward_lstm import FEATURES as RAW_FEATURES
from validation import inspiratory_mae, load_breath_ids

OUT_DIR = ROOT / "submission_trial9"


def load_model(device: torch.device):
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    model = BiLSTM4x256(len(ALL_FEATURES)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, mean, std, checkpoint


def predict(model, features: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    scaled = (features - mean) / std
    chunks = []
    with torch.no_grad():
        tensor = torch.tensor(scaled, device=device)
        for start in range(0, tensor.shape[0], BATCH_SIZE):
            chunks.append(model(tensor[start : start + BATCH_SIZE]).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def breaths_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["breath_id", "time_step"], kind="mergesort")
    ids = ordered["breath_id"].to_numpy().reshape(-1, 80)
    if not np.all(ids[:, :1] == ids):
        raise RuntimeError("rows are not grouped into 80-step breaths")
    raw = ordered[RAW_FEATURES].to_numpy(np.float32).reshape(-1, 80, len(RAW_FEATURES))
    return ordered, add_features(raw)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, checkpoint = load_model(device)

    breath_ids, raw, y, u_out = load_breaths()
    features = add_features(raw)
    is_val = np.isin(breath_ids, load_breath_ids("val"))
    val_pred = predict(model, features[is_val], mean, std, device)
    val_mae = inspiratory_mae(y[is_val], val_pred, u_out[is_val])
    inspiratory = u_out[is_val] == 0
    r_values = features[is_val][:, :, 3].reshape(-1)
    c_values = features[is_val][:, :, 4].reshape(-1)
    y_flat = y[is_val].reshape(-1)
    pred_flat = val_pred.reshape(-1)
    insp_flat = inspiratory.reshape(-1)

    def group_mae(values: np.ndarray) -> dict[str, float]:
        scores = {}
        for level in sorted(set(values[insp_flat].tolist())):
            mask = insp_flat & (values == level)
            scores[str(level)] = float(np.abs(y_flat[mask] - pred_flat[mask]).mean())
        return scores

    test = pd.read_csv(ROOT / "data" / "test.csv")
    sample_ids = pd.read_csv(ROOT / "data" / "sample_submission.csv", usecols=["id"])["id"].to_numpy()
    if not np.array_equal(sample_ids, test["id"].to_numpy()):
        raise RuntimeError("test ids do not match sample_submission order")
    ordered, test_features = breaths_from_frame(test)
    test_pred = predict(model, test_features, mean, std, device).reshape(-1)
    if not np.isfinite(test_pred).all():
        raise RuntimeError("non-finite test prediction")
    submission = pd.DataFrame({"id": ordered["id"].to_numpy(), "pressure": test_pred})
    submission = submission.sort_values("id", kind="mergesort")
    if not np.array_equal(submission["id"].to_numpy(), sample_ids):
        raise RuntimeError("submission ids are not in sample order")

    OUT_DIR.mkdir(exist_ok=True)
    submission_path = OUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    summary = {
        "model": "trial 9 bidirectional LSTM, 4 layers, hidden 256",
        "weights": str(WEIGHTS_PATH),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_best_val_mae": float(checkpoint["best_val_mae"]),
        "recomputed_val_mae": val_mae,
        "val_mae_by_R": group_mae(r_values),
        "val_mae_by_C": group_mae(c_values),
        "n_test_rows": int(len(submission)),
        "pressure_min": float(submission["pressure"].min()),
        "pressure_max": float(submission["pressure"].max()),
        "submission": str(submission_path),
    }
    (OUT_DIR / "submission_check.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
