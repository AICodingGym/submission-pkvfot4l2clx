"""Trial 11: fine-tune trial 10 on R=50 breaths only.

The recorded score stitches the two models. Rows with R=50 use this specialist.
Every other row keeps the trial 10 prediction. The cosine starts at 1e-4 and
falls to 1e-5 over 200 epochs. Trial 10 weights are not overwritten.

The backward pass and the u_in * R / cumsum(u_in)/C pair remain the outside
ideas already flagged for the final report.
"""

from __future__ import annotations

import ctypes
import json
import time

import numpy as np
import torch

from experiment_bilstm_4x256_features import BiLSTM4x256, add_features
from experiment_bilstm_4x256_r50 import WEIGHTS_PATH as TRIAL10_WEIGHTS
from experiment_forward_lstm import BATCH_SIZE, ROOT, load_breaths
from validation import inspiratory_mae, load_breath_ids

RESULT_PATH = ROOT / "experiments" / "bilstm_r50_specialist.json"
WEIGHTS_PATH = ROOT / "experiments" / "bilstm_r50_specialist.pt"
MAX_EPOCH = 200
PEAK_LR = 1e-4
MIN_LR = 1e-5
TRIAL10_VAL_MAE = 0.22224515897219077
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def predict(model, features: torch.Tensor) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], BATCH_SIZE):
            chunks.append(model(features[start : start + BATCH_SIZE]).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def slice_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    flat_y = y_true.reshape(-1)
    flat_p = y_pred.reshape(-1)
    flat_m = mask.reshape(-1)
    return float(np.abs(flat_y[flat_m] - flat_p[flat_m]).mean())


def main() -> None:
    keep_awake()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breath_ids, raw, y, u_out = load_breaths()
    features = add_features(raw)
    is_r50 = np.abs(raw[:, 0, 3] - 50) < 0.1
    is_val = np.isin(breath_ids, load_breath_ids("val"))
    train_mask = is_r50 & ~is_val

    checkpoint = torch.load(TRIAL10_WEIGHTS, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    scaled = (features - mean) / std
    x_train = torch.tensor(scaled[train_mask], device=device)
    y_train = torch.tensor(y[train_mask], device=device)
    u_train = torch.tensor(u_out[train_mask], device=device)
    x_val = torch.tensor(scaled[is_val], device=device)
    y_val = y[is_val]
    u_val = u_out[is_val]
    r50_val = is_r50[is_val]

    frozen = BiLSTM4x256(scaled.shape[-1]).to(device)
    frozen.load_state_dict(checkpoint["state_dict"])
    trial10_val = predict(frozen, x_val)
    del frozen
    torch.cuda.empty_cache()
    baseline = inspiratory_mae(y_val, trial10_val, u_val)
    print(f"trial10_stitched_check {baseline:.4f} r50_train_breaths {int(train_mask.sum())}", flush=True)

    model = BiLSTM4x256(scaled.shape[-1]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=PEAK_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=MIN_LR)
    n_train = x_train.shape[0]
    history = []
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCH + 1):
        model.train()
        order = torch.randperm(n_train, device=device)
        total_abs = 0.0
        total_count = 0
        lr = float(optimizer.param_groups[0]["lr"])
        for start in range(0, n_train, BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            pred = model(x_train[batch])
            mask = u_train[batch] == 0
            error = (pred - y_train[batch]).abs()
            loss = error[mask].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_abs += float(error[mask].sum().detach())
            total_count += int(mask.sum())
        scheduler.step()

        specialist_val = predict(model, x_val)
        stitched = trial10_val.copy()
        stitched[r50_val] = specialist_val[r50_val]
        val_mae = inspiratory_mae(y_val, stitched, u_val)
        r50_mae = slice_mae(y_val, specialist_val, (u_val == 0) & r50_val[:, None])
        train_mae = total_abs / total_count
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_mae_r50": train_mae,
                "stitched_val_mae": val_mae,
                "r50_val_mae": r50_mae,
                "learning_rate": lr,
            }
        )
        print(
            f"epoch {epoch} train_r50 {train_mae:.4f} stitched {val_mae:.4f} "
            f"r50 {r50_mae:.4f} best {best_mae:.4f} lr {lr:.6g}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    last = history[-1]
    result = {
        "model": "r50_specialist_stitched_with_trial10",
        "change": "fine-tune trial 10 on R=50 breaths only; stitch R=50 predictions into trial 10",
        "loaded_trial10_weights": True,
        "peak_learning_rate": PEAK_LR,
        "min_learning_rate": MIN_LR,
        "epochs": MAX_EPOCH,
        "n_r50_train_breaths": int(train_mask.sum()),
        "trial10_val_mae": baseline,
        "best_epoch": best_epoch,
        "best_stitched_val_mae": best_mae,
        "last_stitched_val_mae": last["stitched_val_mae"],
        "last_r50_val_mae": last["r50_val_mae"],
        "last_train_mae_r50": last["train_mae_r50"],
        "mae_change_vs_trial10": best_mae - baseline,
        "target_r50_for_overall_0.20": 0.227,
        "history": history,
        "device": str(device),
        "runtime_seconds": round(elapsed, 1),
        "flag": "backward pass and u_in * R, cumsum(u_in)/C remain the outside ideas from earlier trials",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    torch.save(
        {
            "state_dict": best_state,
            "feature_mean": mean,
            "feature_std": std,
            "features": checkpoint.get("features"),
            "best_epoch": best_epoch,
            "best_stitched_val_mae": best_mae,
        },
        WEIGHTS_PATH,
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_stitched_val_mae": best_mae,
                "last_r50_val_mae": last["r50_val_mae"],
                "mae_change_vs_trial10": result["mae_change_vs_trial10"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
