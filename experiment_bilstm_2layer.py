"""Two-layer bidirectional LSTM. The extra layer is the only change from trial 6.

The backward pass remains the outside idea flagged for the final report.
This run does not load trial 6 weights. Hidden size stays 128 per direction,
inputs stay the raw columns, and the split, loss, and 200-epoch rules are unchanged.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import nn

from experiment_forward_lstm import (
    BATCH_SIZE,
    FEATURES,
    HIDDEN_SIZE,
    LEARNING_RATE,
    ROOT,
    SEED,
    batched_mae,
    load_breaths,
    standardize,
)
from validation import load_breath_ids

RESULT_PATH = ROOT / "experiments" / "bilstm_2layer.json"
WEIGHTS_PATH = ROOT / "experiments" / "bilstm_2layer.pt"
MAX_EPOCH = 200
LR_PATIENCE = 5
STOP_PATIENCE = 10
MIN_LR = 1e-5
ONE_LAYER_VAL_MAE = 0.7678641960893317


class BiLSTMTwoLayer(nn.Module):
    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence).squeeze(-1)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breath_ids, x, y, u_out = load_breaths()
    is_val = np.isin(breath_ids, load_breath_ids("val"))
    x_train, x_val, mean, std = standardize(x[~is_val], x[is_val])
    y_train = torch.tensor(y[~is_val], device=device)
    y_val = torch.tensor(y[is_val], device=device)
    u_train = torch.tensor(u_out[~is_val], device=device)
    u_val = torch.tensor(u_out[is_val], device=device)
    x_train_t = torch.tensor(x_train, device=device)
    x_val_t = torch.tensor(x_val, device=device)

    model = BiLSTMTwoLayer(len(FEATURES), HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_train = x_train_t.shape[0]
    history = []
    lr_cuts = []
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    since_best = 0
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCH + 1):
        model.train()
        order = torch.randperm(n_train, device=device)
        total_abs = 0.0
        total_count = 0
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

        val_mae = batched_mae(model, x_val_t, y_val, u_val, BATCH_SIZE)
        train_mae = total_abs / total_count
        lr = float(optimizer.param_groups[0]["lr"])
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            since_best = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            since_best += 1
            if since_best % LR_PATIENCE == 0 and lr > MIN_LR:
                new_lr = max(lr * 0.5, MIN_LR)
                for group in optimizer.param_groups:
                    group["lr"] = new_lr
                lr_cuts.append({"epoch": epoch, "learning_rate": new_lr})
                lr = new_lr
        history.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "learning_rate": lr,
                "epochs_since_best": since_best,
            }
        )
        print(
            f"epoch {epoch} train_mae {train_mae:.4f} val_mae {val_mae:.4f} "
            f"best {best_mae:.4f} lr {lr:.6g} since_best {since_best}",
            flush=True,
        )
        if since_best >= STOP_PATIENCE:
            stopped_early = True
            break

    elapsed = time.perf_counter() - started
    last = history[-1]
    result = {
        "model": "bidirectional_lstm",
        "flag": "backward pass remains the outside idea from trial 6; the new change is a second layer",
        "bidirectional": True,
        "layers": 2,
        "change": "add a second bidirectional layer; hidden size, features, split, and epoch rules unchanged",
        "features": FEATURES,
        "hidden_size_per_direction": HIDDEN_SIZE,
        "loaded_trial6_weights": False,
        "max_epoch": MAX_EPOCH,
        "last_epoch": last["epoch"],
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "last_val_mae": last["val_mae"],
        "last_train_mae": last["train_mae"],
        "one_layer_val_mae": ONE_LAYER_VAL_MAE,
        "mae_change_vs_one_layer": best_mae - ONE_LAYER_VAL_MAE,
        "learning_rate_cuts": lr_cuts,
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
                "last_epoch": last["epoch"],
                "last_val_mae": last["val_mae"],
                "stopped_early": stopped_early,
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
