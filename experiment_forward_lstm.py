"""Forward LSTM on the raw columns, scored on the locked breath holdout.

One example is one breath of 80 steps. The network reads steps in time order and
predicts pressure at each step from the current step and earlier ones only.
Inputs are u_in, u_out, time_step, R, and C. Lag columns are not used.
Loss is L1 on inspiratory steps (u_out == 0), which is the leaderboard metric.
The epoch count and the other settings below were fixed before this run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from validation import inspiratory_mae, load_breath_ids

ROOT = Path(__file__).resolve().parent
FEATURES = ["u_in", "u_out", "time_step", "R", "C"]
RESULT_PATH = ROOT / "experiments" / "forward_lstm.json"
WEIGHTS_PATH = ROOT / "experiments" / "forward_lstm.pt"
PREVIOUS_VAL_MAE = 1.2675111937941614

HIDDEN_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
SEED = 42


class ForwardLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence).squeeze(-1)


def load_breaths() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(ROOT / "data" / "train.csv", usecols=["breath_id", *FEATURES, "pressure"])
    frame = frame.sort_values(["breath_id", "time_step"], kind="mergesort")
    breath_ids = frame["breath_id"].to_numpy().reshape(-1, 80)
    if not np.all(breath_ids[:, :1] == breath_ids):
        raise RuntimeError("rows are not grouped into 80-step breaths")
    x = frame[FEATURES].to_numpy(np.float32).reshape(-1, 80, len(FEATURES))
    y = frame["pressure"].to_numpy(np.float32).reshape(-1, 80)
    u_out = frame["u_out"].to_numpy(np.float32).reshape(-1, 80)
    return breath_ids[:, 0], x, y, u_out


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1))
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (other_x - mean) / std, mean, std


def batched_mae(model: nn.Module, x: torch.Tensor, y: torch.Tensor, u_out: torch.Tensor, batch_size: int) -> float:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            predictions.append(model(x[start : start + batch_size]))
    pred = torch.cat(predictions).detach().cpu().numpy()
    return inspiratory_mae(y.detach().cpu().numpy(), pred, u_out.detach().cpu().numpy())


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

    model = ForwardLSTM(len(FEATURES), HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_train = x_train_t.shape[0]
    history = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
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
        history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae})
        print(f"epoch {epoch} train_mae {train_mae:.4f} val_mae {val_mae:.4f}", flush=True)

    elapsed = time.perf_counter() - started
    final = history[-1]
    result = {
        "model": "forward_lstm",
        "bidirectional": False,
        "change": "replace the row-wise tree with a forward LSTM over the 80 steps; raw columns only",
        "features": FEATURES,
        "hidden_size": HIDDEN_SIZE,
        "layers": 1,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "loss": "L1 where u_out == 0",
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "previous_val_mae": PREVIOUS_VAL_MAE,
        "val_mae": final["val_mae"],
        "train_mae": final["train_mae"],
        "mae_change": final["val_mae"] - PREVIOUS_VAL_MAE,
        "history": history,
        "device": str(device),
        "runtime_seconds": round(elapsed, 1),
        "n_train_breaths": int((~is_val).sum()),
        "n_val_breaths": int(is_val.sum()),
    }
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    torch.save({"state_dict": model.state_dict(), "feature_mean": mean, "feature_std": std}, WEIGHTS_PATH)
    print(json.dumps({k: result[k] for k in ("val_mae", "train_mae", "mae_change", "runtime_seconds", "device")}, indent=2))


if __name__ == "__main__":
    main()
