"""Score model B, model C, and the trial 11 stitch on the locked holdout.

The trial 11 prediction uses the specialist on R=50 rows and trial 10 everywhere else.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from experiment_bilstm_4x256_features import BiLSTM4x256, add_features
from experiment_bilstm_4x256_r50 import WEIGHTS_PATH as TRIAL10_WEIGHTS
from experiment_bilstm_r50_specialist import WEIGHTS_PATH as SPECIALIST_WEIGHTS
from experiment_forward_lstm import BATCH_SIZE, ROOT, load_breaths
from experiment_model_b import WEIGHTS_PATH as MODEL_B_WEIGHTS
from experiment_model_c import WEIGHTS_PATH as MODEL_C_WEIGHTS
from validation import inspiratory_mae, load_breath_ids


def load_model(path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    model = BiLSTM4x256(len(checkpoint["features"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, mean, std


def predict(model, features, mean, std, device) -> np.ndarray:
    scaled = torch.tensor((features - mean) / std, device=device)
    chunks = []
    with torch.no_grad():
        for start in range(0, scaled.shape[0], BATCH_SIZE):
            chunks.append(model(scaled[start : start + BATCH_SIZE]).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def cell_mae(y_true, y_pred, mask) -> float:
    flat_y = y_true.reshape(-1)
    flat_p = y_pred.reshape(-1)
    flat_m = mask.reshape(-1)
    return float(np.abs(flat_y[flat_m] - flat_p[flat_m]).mean())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breath_ids, raw, y, u_out = load_breaths()
    features = add_features(raw)
    is_val = np.isin(breath_ids, load_breath_ids("val"))
    featured = features[is_val]
    y_val = y[is_val]
    u_val = u_out[is_val]
    resistance = featured[:, :, 3]
    compliance = featured[:, :, 4]
    r50 = np.abs(resistance - 50) < 0.1
    hard = r50 & ((np.abs(compliance - 20) < 0.1) | (np.abs(compliance - 50) < 0.1))
    inspiratory = u_val == 0

    loaded = {}
    for name, path in (
        ("trial10", TRIAL10_WEIGHTS),
        ("specialist", SPECIALIST_WEIGHTS),
        ("model_b", MODEL_B_WEIGHTS),
        ("model_c", MODEL_C_WEIGHTS),
    ):
        model, mean, std = load_model(path, device)
        loaded[name] = predict(model, featured, mean, std, device)
        del model
        torch.cuda.empty_cache()

    trial11 = loaded["trial10"].copy()
    trial11[r50] = loaded["specialist"][r50]
    predictions = {"trial11": trial11, "model_b": loaded["model_b"], "model_c": loaded["model_c"]}

    def pack(name, pred):
        return {
            "val_mae": inspiratory_mae(y_val, pred, u_val),
            "hard_val_mae": cell_mae(y_val, pred, inspiratory & hard),
        }

    median_all = np.median(np.stack(list(predictions.values()), axis=0), axis=0)
    median_11_b = np.median(np.stack([predictions["trial11"], predictions["model_b"]], axis=0), axis=0)
    hard_rows = (inspiratory & hard).reshape(-1)
    residual = {
        name: (pred - y_val).reshape(-1)[hard_rows]
        for name, pred in predictions.items()
    }
    result = {
        "individual": {name: pack(name, pred) for name, pred in predictions.items()},
        "median_11_b_c": pack("median", median_all),
        "median_11_b": pack("median", median_11_b),
        "hard_residual_correlation": {
            "trial11_model_b": float(np.corrcoef(residual["trial11"], residual["model_b"])[0, 1]),
            "trial11_model_c": float(np.corrcoef(residual["trial11"], residual["model_c"])[0, 1]),
            "model_b_model_c": float(np.corrcoef(residual["model_b"], residual["model_c"])[0, 1]),
        },
    }
    print(json.dumps(result, indent=2))
    (ROOT / "experiments" / "median_b_c.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
