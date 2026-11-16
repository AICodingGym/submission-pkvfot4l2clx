"""Model B: a new seed of the trial 9 schedule, then the trial 10 R=50 upweight.

Phase 1 matches trial 9: fresh weights, cosine from 1e-3 to 1e-5, 200 epochs.
Phase 2 loads those best weights, doubles the loss on inspiratory R=50 rows,
and uses a fresh Adam with a cosine from 3e-4 to 1e-5 for 200 epochs.
The recorded score is unweighted inspiratory MAE. Trial 9 and trial 10 files
are not overwritten.

The backward pass and the u_in * R / cumsum(u_in)/C pair remain the outside
ideas already flagged for the final report.
"""

from __future__ import annotations

import ctypes
import json
import time

import numpy as np
import torch

from experiment_bilstm_4x256_features import ALL_FEATURES, BiLSTM4x256, add_features
from experiment_forward_lstm import BATCH_SIZE, LEARNING_RATE, ROOT, batched_mae, load_breaths, standardize
from validation import inspiratory_mae, load_breath_ids

SEED = 7
MAX_EPOCH = 200
MIN_LR = 1e-5
PHASE2_PEAK_LR = 3e-4
R50_LOSS_WEIGHT = 2.0
PHASE1_PATH = ROOT / "experiments" / "model_b_phase1.pt"
PHASE1_JSON = ROOT / "experiments" / "model_b_phase1.json"
WEIGHTS_PATH = ROOT / "experiments" / "model_b.pt"
RESULT_PATH = ROOT / "experiments" / "model_b.json"
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def slice_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def save_weights(path, state, mean, std, best_epoch, best_mae, key: str) -> None:
    torch.save(
        {
            "state_dict": state,
            "feature_mean": mean,
            "feature_std": std,
            "features": ALL_FEATURES,
            "best_epoch": best_epoch,
            key: best_mae,
            "seed": SEED,
        },
        path,
    )


def main() -> None:
    keep_awake()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breath_ids, raw, y, u_out = load_breaths()
    features = add_features(raw)
    resistance = raw[:, :, 3]
    is_val = np.isin(breath_ids, load_breath_ids("val"))
    x_train, x_val, mean, std = standardize(features[~is_val], features[is_val])
    y_train = torch.tensor(y[~is_val], device=device)
    y_val = torch.tensor(y[is_val], device=device)
    u_train = torch.tensor(u_out[~is_val], device=device)
    u_val = torch.tensor(u_out[is_val], device=device)
    x_train_t = torch.tensor(x_train, device=device)
    x_val_t = torch.tensor(x_val, device=device)
    n_train = x_train_t.shape[0]

    model = BiLSTM4x256(x_train_t.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=MIN_LR)
    history = []
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    print(f"phase 1 seed {SEED} train_breaths {n_train}", flush=True)

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
            loss = error[mask].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_abs += float(error[mask].sum().detach())
            total_count += int(mask.sum())
        scheduler.step()
        val_mae = batched_mae(model, x_val_t, y_val, u_val, BATCH_SIZE)
        train_mae = total_abs / total_count
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae, "learning_rate": lr})
        print(
            f"phase 1 epoch {epoch} train_mae {train_mae:.4f} val_mae {val_mae:.4f} "
            f"best {best_mae:.4f} lr {lr:.6g}",
            flush=True,
        )

    phase1_seconds = round(time.perf_counter() - started, 1)
    PHASE1_JSON.parent.mkdir(exist_ok=True)
    PHASE1_JSON.write_text(
        json.dumps(
            {
                "model": "model_b_phase1",
                "seed": SEED,
                "best_epoch": best_epoch,
                "best_val_mae": best_mae,
                "runtime_seconds": phase1_seconds,
                "history": history,
            },
            indent=2,
        )
    )
    save_weights(PHASE1_PATH, best_state, mean, std, best_epoch, best_mae, "best_val_mae")
    print(f"phase 1 done best {best_mae:.4f} epoch {best_epoch} seconds {phase1_seconds}", flush=True)

    torch.manual_seed(SEED + 1)
    model.load_state_dict(best_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=PHASE2_PEAK_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=MIN_LR)
    r_train = torch.tensor(resistance[~is_val], device=device)
    r_val = resistance[is_val]
    history2 = []
    best_mae2 = float("inf")
    best_epoch2 = 0
    best_state2 = None
    started2 = time.perf_counter()

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
        if val_mae < best_mae2:
            best_mae2 = val_mae
            best_epoch2 = epoch
            best_state2 = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history2.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "val_mae_r50": val_mae_r50,
                "learning_rate": lr,
            }
        )
        print(
            f"phase 2 epoch {epoch} train_mae {train_mae:.4f} val_mae {val_mae:.4f} "
            f"r50 {val_mae_r50:.4f} best {best_mae2:.4f} lr {lr:.6g}",
            flush=True,
        )

    phase2_seconds = round(time.perf_counter() - started2, 1)
    last = history2[-1]
    result = {
        "model": "model_b",
        "seed": SEED,
        "phase1_best_epoch": best_epoch,
        "phase1_best_val_mae": best_mae,
        "phase1_runtime_seconds": phase1_seconds,
        "phase2_best_epoch": best_epoch2,
        "best_val_mae": best_mae2,
        "last_val_mae": last["val_mae"],
        "last_val_mae_r50": last["val_mae_r50"],
        "last_train_mae": last["train_mae"],
        "phase2_runtime_seconds": phase2_seconds,
        "runtime_seconds": round(phase1_seconds + phase2_seconds, 1),
        "history_phase2": history2,
        "flag": "backward pass and u_in * R, cumsum(u_in)/C remain the outside ideas from earlier trials",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    save_weights(WEIGHTS_PATH, best_state2, mean, std, best_epoch2, best_mae2, "best_val_mae")
    print(
        json.dumps(
            {
                "phase1_best_val_mae": best_mae,
                "phase2_best_epoch": best_epoch2,
                "best_val_mae": best_mae2,
                "last_val_mae_r50": last["val_mae_r50"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
