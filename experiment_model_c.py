"""Model C: a new seed, trained from scratch, with a heavier hard-cell loss.

Same 4-layer bidirectional LSTM, features, split, and 200-epoch cosine from
1e-3 to 1e-5 as trial 9. Inspiratory rows with R=50 and C=20 or C=50 count
four times in the training loss. Every other inspiratory row counts once.
The recorded score is unweighted inspiratory MAE.

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
from experiment_forward_lstm import BATCH_SIZE, LEARNING_RATE, ROOT, load_breaths, standardize
from validation import inspiratory_mae, load_breath_ids

SEED = 11
MAX_EPOCH = 200
MIN_LR = 1e-5
HARD_LOSS_WEIGHT = 4.0
WEIGHTS_PATH = ROOT / "experiments" / "model_c.pt"
RESULT_PATH = ROOT / "experiments" / "model_c.json"
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
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breath_ids, raw, y, u_out = load_breaths()
    features = add_features(raw)
    resistance = raw[:, :, 3]
    compliance = raw[:, :, 4]
    is_val = np.isin(breath_ids, load_breath_ids("val"))
    x_train, x_val, mean, std = standardize(features[~is_val], features[is_val])
    y_train = torch.tensor(y[~is_val], device=device)
    y_val = torch.tensor(y[is_val], device=device)
    u_train = torch.tensor(u_out[~is_val], device=device)
    u_val = torch.tensor(u_out[is_val], device=device)
    r_train = torch.tensor(resistance[~is_val], device=device)
    c_train = torch.tensor(compliance[~is_val], device=device)
    x_train_t = torch.tensor(x_train, device=device)
    x_val_t = torch.tensor(x_val, device=device)
    hard_val = (np.abs(resistance[is_val] - 50) < 0.1) & (
        (np.abs(compliance[is_val] - 20) < 0.1) | (np.abs(compliance[is_val] - 50) < 0.1)
    )
    n_train = x_train_t.shape[0]

    model = BiLSTM4x256(x_train_t.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=MIN_LR)
    history = []
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    print(f"model c seed {SEED} train_breaths {n_train}", flush=True)

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
            hard = ((r_train[batch] - 50).abs() < 0.1) & (
                ((c_train[batch] - 20).abs() < 0.1) | ((c_train[batch] - 50).abs() < 0.1)
            )
            weights = torch.where(hard, HARD_LOSS_WEIGHT, 1.0)
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
        hard_mask = (u_np == 0) & hard_val
        hard_mae = slice_mae(y_np.reshape(-1), pred_val.reshape(-1), hard_mask.reshape(-1))
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
                "hard_val_mae": hard_mae,
                "learning_rate": lr,
            }
        )
        print(
            f"epoch {epoch} train_mae {train_mae:.4f} val_mae {val_mae:.4f} "
            f"hard {hard_mae:.4f} best {best_mae:.4f} lr {lr:.6g}",
            flush=True,
        )

    elapsed = round(time.perf_counter() - started, 1)
    last = history[-1]
    result = {
        "model": "model_c",
        "seed": SEED,
        "hard_loss_weight": HARD_LOSS_WEIGHT,
        "hard_cells": "R=50 and C=20 or C=50",
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "last_val_mae": last["val_mae"],
        "last_hard_val_mae": last["hard_val_mae"],
        "last_train_mae": last["train_mae"],
        "runtime_seconds": elapsed,
        "history": history,
        "flag": "backward pass and u_in * R, cumsum(u_in)/C remain the outside ideas from earlier trials",
    }
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    torch.save(
        {
            "state_dict": best_state,
            "feature_mean": mean,
            "feature_std": std,
            "features": ALL_FEATURES,
            "best_epoch": best_epoch,
            "best_val_mae": best_mae,
            "seed": SEED,
        },
        WEIGHTS_PATH,
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_mae": best_mae,
                "last_hard_val_mae": last["hard_val_mae"],
                "runtime_seconds": elapsed,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
