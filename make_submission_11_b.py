"""Build the averaged trial 11 and model B prediction file.

Each row is the mean of the trial 11 stitch and model B. The trial 11 stitch
uses the specialist on R=50 and trial 10 on every other row. Model C is left
out. Predictions are written in the original test-file id order.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from experiment_bilstm_4x256_features import ALL_FEATURES, BiLSTM4x256, add_features
from experiment_bilstm_4x256_r50 import WEIGHTS_PATH as TRIAL10_WEIGHTS
from experiment_bilstm_r50_specialist import WEIGHTS_PATH as SPECIALIST_WEIGHTS
from experiment_forward_lstm import BATCH_SIZE, ROOT, load_breaths
from experiment_forward_lstm import FEATURES as RAW_FEATURES
from experiment_model_b import WEIGHTS_PATH as MODEL_B_WEIGHTS
from validation import inspiratory_mae, load_breath_ids

OUT_DIR = ROOT / "submission_11_b"
TRIAL11_VAL_MAE = 0.21992300559001868


def load_model(path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    model = BiLSTM4x256(len(ALL_FEATURES)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, mean, std


def predict(model, features, mean, std, device) -> np.ndarray:
    scaled = (features - mean) / std
    chunks = []
    with torch.no_grad():
        tensor = torch.tensor(scaled, device=device)
        for start in range(0, tensor.shape[0], BATCH_SIZE):
            chunks.append(model(tensor[start : start + BATCH_SIZE]).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def trial11_stitch(trial10, specialist, resistance) -> np.ndarray:
    stitched = trial10.copy()
    stitched[np.abs(resistance - 50) < 0.1] = specialist[np.abs(resistance - 50) < 0.1]
    return stitched


def breaths_from_frame(frame: pd.DataFrame):
    ordered = frame.sort_values(["breath_id", "time_step"], kind="mergesort")
    ids = ordered["breath_id"].to_numpy().reshape(-1, 80)
    if not np.all(ids[:, :1] == ids):
        raise RuntimeError("rows are not grouped into 80-step breaths")
    raw = ordered[RAW_FEATURES].to_numpy(np.float32).reshape(-1, 80, len(RAW_FEATURES))
    return ordered, add_features(raw)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial10, mean10, std10 = load_model(TRIAL10_WEIGHTS, device)
    specialist, mean_s, std_s = load_model(SPECIALIST_WEIGHTS, device)
    model_b, mean_b, std_b = load_model(MODEL_B_WEIGHTS, device)

    breath_ids, raw, y, u_out = load_breaths()
    features = add_features(raw)
    is_val = np.isin(breath_ids, load_breath_ids("val"))
    val_features = features[is_val]
    base_val = predict(trial10, val_features, mean10, std10, device)
    specialist_val = predict(specialist, val_features, mean_s, std_s, device)
    model_b_val = predict(model_b, val_features, mean_b, std_b, device)
    averaged_val = (
        trial11_stitch(base_val, specialist_val, val_features[:, :, 3]) + model_b_val
    ) / 2
    val_mae = inspiratory_mae(y[is_val], averaged_val, u_out[is_val])
    if val_mae >= TRIAL11_VAL_MAE:
        raise RuntimeError("average is not better than trial 11")

    inspiratory = u_out[is_val] == 0
    r_values = val_features[:, :, 3].reshape(-1)
    c_values = val_features[:, :, 4].reshape(-1)
    y_flat = y[is_val].reshape(-1)
    pred_flat = averaged_val.reshape(-1)
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
    resistance = test_features[:, :, 3]
    averaged_test = (
        trial11_stitch(
            predict(trial10, test_features, mean10, std10, device),
            predict(specialist, test_features, mean_s, std_s, device),
            resistance,
        )
        + predict(model_b, test_features, mean_b, std_b, device)
    ) / 2
    test_pred = averaged_test.reshape(-1)
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
        "model": "average of the trial 11 stitch and model B",
        "recomputed_val_mae": val_mae,
        "trial11_val_mae": TRIAL11_VAL_MAE,
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
