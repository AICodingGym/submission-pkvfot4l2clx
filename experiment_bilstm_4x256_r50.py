"""Trial 10: resume trial 9 and train harder on R=50 breaths.

Same network, features, split, and batch as trial 9. Training starts from the
trial 9 checkpoint. Inspiratory rows with R=50 count double in the training
loss. The recorded MAE stays unweighted. The cosine starts at 3e-4 and falls
to 1e-5 over 200 epochs. The best validation weights are kept.

The backward pass and the u_in * R / cumsum(u_in)/C pair remain the outside
ideas already flagged for the final report.
"""

from __future__ import annotations

import ctypes
import json
import time

import numpy as np
import torch

from experiment_bilstm_4x256_features import WEIGHTS_PATH as TRIAL9_WEIGHTS
from experiment_bilstm_4x256_features import BiLSTM4x256, add_features
from experiment_forward_lstm import BATCH_SIZE, ROOT, load_breaths
from validation import inspiratory_mae, load_breath_ids

RESULT_PATH = ROOT / "experiments" / "bilstm_4x256_r50.json"
WEIGHTS_PATH = ROOT / "experiments" / "bilstm_4x256_r50.pt"
MAX_EPOCH = 200
PEAK_LR = 3e-4
MIN_LR = 1e-5
R50_LOSS_WEIGHT = 2.0
TRIAL9_VAL_MAE = 0.24449364977244953
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def slice_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def main() -> None:
    keep_awake()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breath_ids, raw, y, u_out = load_breaths()
    features = add_features(raw)
    resistance = raw[:, :, 3]
    is_val = np.isin(breath_ids, load_breath_ids("val"))

    checkpoint = torch.load(TRIAL9_WEIGHTS, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    x_train = (features[~is_val] - mean) / std
    x_val = (features[is_val] - mean) / std
    y_train = torch.tensor(y[~is_val], device=device)
    y_val = torch.tensor(y[is_val], device=device)
    u_train = torch.tensor(u_out[~is_val], device=device)
    u_val = torch.tensor(u_out[is_val], device=device)
    r_train = torch.tensor(resistance[~is_val], device=device)
    r_val = resistance[is_val]
    x_train_t = torch.tensor(x_train, device=device)
    x_val_t = torch.tensor(x_val, device=device)

    model = BiLSTM4x256(x_train_t.shape[-1]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=PEAK_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=MIN_LR)
    n_train = x_train_t.shape[0]
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
            pred = model(x_train_t[batch])
            mask = u_train[batch] == 0
            error = (pred - y_train[batch]).abs()
            weights = torch.where((r_train[batch] - 50).abs() < 0.1, R50_LOSS_WEIGHT, 1.0)
            weighted = error * weights
            loss = weighted[mask].sum() / weights[mask].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_abs += float(error[mask].sum().detach())
            total_count += int(mask.sum())
        scheduler.step()

        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, x_val_t.shape[0], BATCH_SIZE):
                predictions.append(model(x_val_t[start : start + BATCH_SIZE]))
        pred_val = torch.cat(predictions).cpu().numpy()
        y_np = y_val.cpu().numpy()
        u_np = u_val.cpu().numpy()
        val_mae = inspiratory_mae(y_np, pred_val, u_np)
        r50_mask = (u_np == 0) & (np.abs(r_val - 50) < 0.1)
        val_mae_r50 = slice_mae(y_np.reshape(-1), pred_val.reshape(-1), r50_mask.reshape(-1))
        train_mae = total_abs / total_count
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "val_mae_r50": val_mae_r50,
                "learning_rate": lr,
            }
        )
        print(
            f"epoch {epoch} train_mae {train_mae:.4f} val_mae {val_mae:.4f} "
            f"r50 {val_mae_r50:.4f} best {best_mae:.4f} lr {lr:.6g}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    last = history[-1]
    result = {
        "model": "bidirectional_lstm",
        "change": "resume trial 9; double the training loss on inspiratory R=50 rows; cosine from 3e-4 to 1e-5 for 200 epochs",
        "loaded_trial9_weights": True,
        "r50_loss_weight": R50_LOSS_WEIGHT,
        "peak_learning_rate": PEAK_LR,
        "min_learning_rate": MIN_LR,
        "epochs": MAX_EPOCH,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "last_val_mae": last["val_mae"],
        "last_train_mae": last["train_mae"],
        "last_val_mae_r50": last["val_mae_r50"],
        "trial9_val_mae": TRIAL9_VAL_MAE,
        "mae_change_vs_trial9": best_mae - TRIAL9_VAL_MAE,
        "history": history,
        "device": str(device),
        "runtime_seconds": round(elapsed, 1),
        "flag": "backward pass and u_in * R, cumsum(u_in)/C remain the outside ideas from earlier trials",
    }
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    torch.save(
        {
            "state_dict": best_state,
            "feature_mean": mean,
            "feature_std": std,
            "features": checkpoint.get("features"),
            "best_epoch": best_epoch,
            "best_val_mae": best_mae,
        },
        WEIGHTS_PATH,
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_mae": best_mae,
                "last_val_mae": last["val_mae"],
                "last_val_mae_r50": last["val_mae_r50"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
