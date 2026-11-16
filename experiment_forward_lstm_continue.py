"""Continue the forward LSTM from the 10-epoch weights, up to epoch 200.

Same network, split, and raw-column inputs as trial 4. The Adam moments start
fresh because they were not saved. The run keeps the weights with the lowest
validation MAE. If that MAE does not improve for 5 epochs, the learning rate is
halved, down to 1e-5. If it does not improve for 10 epochs, training stops.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from experiment_forward_lstm import (
    BATCH_SIZE,
    FEATURES,
    HIDDEN_SIZE,
    LEARNING_RATE,
    ROOT,
    SEED,
    ForwardLSTM,
    batched_mae,
    load_breaths,
)
from validation import load_breath_ids

START_WEIGHTS = ROOT / "experiments" / "forward_lstm.pt"
START_LOG = ROOT / "experiments" / "forward_lstm.json"
RESULT_PATH = ROOT / "experiments" / "forward_lstm_200.json"
WEIGHTS_PATH = ROOT / "experiments" / "forward_lstm_200.pt"
MAX_EPOCH = 200
LR_PATIENCE = 5
STOP_PATIENCE = 10
MIN_LR = 1e-5
TREE_VAL_MAE = 1.2675111937941614


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = json.loads(START_LOG.read_text())
    best_mae = float(prior["val_mae"])
    best_epoch = int(prior["epochs"])

    breath_ids, x, y, u_out = load_breaths()
    is_val = np.isin(breath_ids, load_breath_ids("val"))
    checkpoint = torch.load(START_WEIGHTS, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    x_train = (x[~is_val] - mean) / std
    x_val = (x[is_val] - mean) / std

    y_train = torch.tensor(y[~is_val], device=device)
    y_val = torch.tensor(y[is_val], device=device)
    u_train = torch.tensor(u_out[~is_val], device=device)
    u_val = torch.tensor(u_out[is_val], device=device)
    x_train_t = torch.tensor(x_train, device=device)
    x_val_t = torch.tensor(x_val, device=device)

    model = ForwardLSTM(len(FEATURES), HIDDEN_SIZE).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_train = x_train_t.shape[0]
    history = []
    lr_cuts = []
    since_best = 0
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(best_epoch + 1, MAX_EPOCH + 1):
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
        improved = val_mae < best_mae
        if improved:
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
        "model": "forward_lstm",
        "bidirectional": False,
        "change": "continue trial 4 weights through epoch 200, with learning-rate cuts and early stopping",
        "continued_from_epoch": int(prior["epochs"]),
        "max_epoch": MAX_EPOCH,
        "last_epoch": last["epoch"],
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "last_val_mae": last["val_mae"],
        "last_train_mae": last["train_mae"],
        "tree_val_mae": TREE_VAL_MAE,
        "mae_change_vs_tree": best_mae - TREE_VAL_MAE,
        "learning_rate_cuts": lr_cuts,
        "lr_patience": LR_PATIENCE,
        "stop_patience": STOP_PATIENCE,
        "min_learning_rate": MIN_LR,
        "history": history,
        "device": str(device),
        "runtime_seconds": round(elapsed, 1),
        "adam_state": "fresh; moments from trial 4 were not saved",
    }
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
