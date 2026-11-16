"""Score the median of trials 8, 9, and 10 on the locked breath holdout."""

from __future__ import annotations

import json

import numpy as np
import torch

from experiment_bilstm_4x256 import WEIGHTS_PATH as TRIAL8_WEIGHTS
from experiment_bilstm_4x256_features import WEIGHTS_PATH as TRIAL9_WEIGHTS
from experiment_bilstm_4x256_features import BiLSTM4x256, add_features
from experiment_bilstm_4x256_r50 import WEIGHTS_PATH as TRIAL10_WEIGHTS
from experiment_forward_lstm import BATCH_SIZE, load_breaths
from validation import inspiratory_mae, load_breath_ids


def load_model(path, n_features: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = BiLSTM4x256(n_features).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    return model, mean, std


def predict(model, features: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    scaled = torch.tensor((features - mean) / std, device=device)
    chunks = []
    with torch.no_grad():
        for start in range(0, scaled.shape[0], BATCH_SIZE):
            chunks.append(model(scaled[start : start + BATCH_SIZE]).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breath_ids, raw, y, u_out = load_breaths()
    is_val = np.isin(breath_ids, load_breath_ids("val"))
    raw_val = raw[is_val]
    featured_val = add_features(raw)[is_val]
    y_val = y[is_val]
    u_val = u_out[is_val]

    specs = [
        ("trial8", TRIAL8_WEIGHTS, raw_val, 5),
        ("trial9", TRIAL9_WEIGHTS, featured_val, featured_val.shape[-1]),
        ("trial10", TRIAL10_WEIGHTS, featured_val, featured_val.shape[-1]),
    ]
    predictions = {}
    individuals = {}
    for name, path, features, n_features in specs:
        model, mean, std = load_model(path, n_features, device)
        pred = predict(model, features, mean, std, device)
        predictions[name] = pred
        individuals[name] = inspiratory_mae(y_val, pred, u_val)
        del model
        torch.cuda.empty_cache()

    stack = np.stack([predictions["trial8"], predictions["trial9"], predictions["trial10"]], axis=0)
    median_pred = np.median(stack, axis=0)
    mean_9_10 = (predictions["trial9"] + predictions["trial10"]) / 2
    result = {
        "individual_val_mae": individuals,
        "median_8_9_10_val_mae": inspiratory_mae(y_val, median_pred, u_val),
        "mean_9_10_val_mae": inspiratory_mae(y_val, mean_9_10, u_val),
    }
    inspiratory = (u_val == 0).reshape(-1)
    resistance = raw_val[:, :, 3].reshape(-1)
    y_flat = y_val.reshape(-1)
    med_flat = median_pred.reshape(-1)
    by_r = {}
    for level in (5, 20, 50):
        mask = inspiratory & (np.abs(resistance - level) < 0.1)
        by_r[str(level)] = float(np.abs(y_flat[mask] - med_flat[mask]).mean())
    result["median_val_mae_by_R"] = by_r
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
