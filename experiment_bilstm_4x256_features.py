"""Trial 9: trial 8's network, plus engineered features and cosine decay.

Same 4-layer bidirectional LSTM, hidden size 256, batch 512, breath split, and
inspiratory L1 loss. Trial 8 weights are not loaded. The run completes all 200
epochs. The learning rate follows a cosine from 1e-3 down to 1e-5. The best
validation weights are kept.

FLAG FOR THE FINAL REPORT: u_in * R and cumsum(u_in) / C are an outside idea.
The column definitions support them, and they are also a common constructed
feature on this challenge. Credit them alongside the backward pass. The other
new columns come from our own trials.
"""

from __future__ import annotations

import ctypes
import json
import time

import numpy as np
import torch
from torch import nn

from experiment_forward_lstm import (
    BATCH_SIZE,
    FEATURES as RAW_FEATURES,
    LEARNING_RATE,
    ROOT,
    SEED,
    batched_mae,
    load_breaths,
    standardize,
)
from validation import load_breath_ids

RESULT_PATH = ROOT / "experiments" / "bilstm_4x256_features.json"
WEIGHTS_PATH = ROOT / "experiments" / "bilstm_4x256_features.pt"
HIDDEN_SIZE = 256
NUM_LAYERS = 4
MAX_EPOCH = 200
MIN_LR = 1e-5
TRIAL8_VAL_MAE = 0.3687169523814254
EXTRA_FEATURES = [
    "u_in_cumsum",
    "u_in_diff1",
    "u_in_cumsum_insp",
    "steps_until_u_out",
    "u_in_times_R",
    "u_in_cumsum_over_C",
]
ALL_FEATURES = [*RAW_FEATURES, *EXTRA_FEATURES]
FLAGGED_FEATURES = ["u_in_times_R", "u_in_cumsum_over_C"]
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class BiLSTM4x256(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(HIDDEN_SIZE * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence).squeeze(-1)


def keep_awake() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def add_features(x: np.ndarray) -> np.ndarray:
    """x is (breaths, 80, 5) in RAW_FEATURES order."""
    u_in = x[:, :, 0]
    u_out = x[:, :, 1]
    resistance = x[:, :, 3]
    compliance = x[:, :, 4]
    cumsum = np.cumsum(u_in, axis=1)
    diff1 = np.zeros_like(u_in)
    diff1[:, 1:] = u_in[:, 1:] - u_in[:, :-1]
    cumsum_insp = np.cumsum(u_in * (u_out == 0), axis=1)
    flipped = u_out > 0.5
    has_flip = flipped.any(axis=1)
    first_flip = np.argmax(flipped, axis=1)
    first_flip = np.where(has_flip, first_flip, 80).astype(np.float32)
    steps_until = first_flip[:, None] - np.arange(80, dtype=np.float32)[None, :]
    extra = np.stack(
        [
            cumsum,
            diff1,
            cumsum_insp,
            steps_until,
            u_in * resistance,
            cumsum / compliance,
        ],
        axis=-1,
    )
    return np.concatenate([x, extra.astype(np.float32)], axis=-1)


def main() -> None:
    keep_awake()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breath_ids, x, y, u_out = load_breaths()
    x = add_features(x)
    is_val = np.isin(breath_ids, load_breath_ids("val"))
    x_train, x_val, mean, std = standardize(x[~is_val], x[is_val])
    y_train = torch.tensor(y[~is_val], device=device)
    y_val = torch.tensor(y[is_val], device=device)
    u_train = torch.tensor(u_out[~is_val], device=device)
    u_val = torch.tensor(u_out[is_val], device=device)
    x_train_t = torch.tensor(x_train, device=device)
    x_val_t = torch.tensor(x_val, device=device)

    model = BiLSTM4x256(x_train_t.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCH, eta_min=MIN_LR
    )
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
            f"epoch {epoch} train_mae {train_mae:.4f} val_mae {val_mae:.4f} "
            f"best {best_mae:.4f} lr {lr:.6g}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    last = history[-1]
    result = {
        "model": "bidirectional_lstm",
        "flag": "u_in * R and cumsum(u_in) / C are an outside idea; credit them with the backward pass",
        "flagged_features": FLAGGED_FEATURES,
        "bidirectional": True,
        "layers": NUM_LAYERS,
        "hidden_size_per_direction": HIDDEN_SIZE,
        "features": ALL_FEATURES,
        "schedule": "cosine annealing from 1e-3 to 1e-5 over 200 epochs",
        "loaded_trial8_weights": False,
        "batch_size": BATCH_SIZE,
        "max_epoch": MAX_EPOCH,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "last_epoch": last["epoch"],
        "last_val_mae": last["val_mae"],
        "last_train_mae": last["train_mae"],
        "trial8_val_mae": TRIAL8_VAL_MAE,
        "mae_change_vs_trial8": best_mae - TRIAL8_VAL_MAE,
        "history": history,
        "device": str(device),
        "runtime_seconds": round(elapsed, 1),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
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
        },
        WEIGHTS_PATH,
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_mae": best_mae,
                "last_val_mae": last["val_mae"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
