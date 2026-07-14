"""
Triple Barrier Labeling - LSTM Training Pipeline (Nifty 500)

Reimplementation of the model/training setup from:
"Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data:
Evidence from Korean Markets" (Kang & Kim, 2025).

Paper's optimal config: 4-layer LSTM, hidden_size=8, window=100, raw OHLCV
input (5 channels), no dropout, Adam(lr=1e-3), CrossEntropyLoss, model
selection on validation macro-F1.

Adapted here for the larger/more heterogeneous Nifty 500 dataset: hidden_size
and dropout are bumped up (see Config), inputs are per-window normalized
(see normalize_window) since raw prices span penny stocks to ~INR 70,000
shares, and the loss uses balanced class weights.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_DIR = "processed_data"
CHECKPOINT_PATH = "best_model.pth"

INPUT_SIZE = 5       # O, H, L, C, V
HIDDEN_SIZE = 32       # paper's optimal was 8; bumped up for our larger/noisier dataset
NUM_LAYERS = 4        # paper's optimal LSTM depth
NUM_CLASSES = 3        # SL, Time Limit, TP
DROPOUT = 0.2        # regularizes the extra capacity above; only active between LSTM layers

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label remapping so CrossEntropyLoss sees classes {0, 1, 2}.
# Raw labels from the triple-barrier builder: -1 (Stop Loss), 0 (Time Limit), 1 (Take Profit).
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
CLASS_NAMES = ["Stop Loss", "Time Limit", "Take Profit"]


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
def normalize_window(window: np.ndarray) -> np.ndarray:
    """Scales a single (100, 5) OHLCV window so different tickers/price levels
    become comparable inputs.

    Nifty 500 spans penny stocks to ~ INR 70,000 shares; without this, the raw
    price magnitude dominates the signal and the LSTM can't share patterns
    across tickers. Columns are O, H, L, C, V.

    - O/H/L/C -> percent change relative to the window's first close price,
      expressed in percentage points (removes absolute price level, keeps
      the shape of the move). A raw fraction (e.g. 0.01 for a 1% move) is
      too small in magnitude for a 4-layer LSTM to react to at initialization
      -- the signal gets crushed to near-zero before reaching the output, so
      the network never trains. Scaling by 100 puts it on a comparable
      footing to the volume z-score below.
    - Volume  -> log1p, then z-scored using this window's own mean/std
      (keeps it scale-free per-ticker without needing global statistics).
    """
    window = window.astype(np.float32)
    ref_close = window[0, 3]
    eps = 1e-8

    ohlc = (window[:, :4] / (ref_close + eps) - 1.0) * 100.0

    log_vol = np.log1p(window[:, 4])
    vol_mean, vol_std = log_vol.mean(), log_vol.std()
    vol_norm = (log_vol - vol_mean) / (vol_std + eps)

    return np.concatenate([ohlc, vol_norm[:, None]], axis=1)


class TripleBarrierDataset(Dataset):
    """Loads a windowed OHLCV split (.npy) and its triple-barrier labels.

    X is expected to have shape (N, 100, 5); y has shape (N,) with values
    in {-1, 0, 1}, which are remapped here to {0, 1, 2} for CrossEntropyLoss.

    X_train.npy is ~2.3 GB; it is memory-mapped (mmap_mode='r') instead of
    being read fully into RAM, since eagerly loading it can exceed available
    memory. Individual windows are copied into small tensors on access.
    """

    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path)

        # Vectorized remap -1/0/1 -> 0/1/2 via a lookup table shifted by +1.
        remap_table = np.array([LABEL_MAP[-1], LABEL_MAP[0], LABEL_MAP[1]], dtype=np.int64)
        y_remapped = remap_table[y.astype(np.int64) + 1]
        self.y = torch.from_numpy(y_remapped).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        # np.array(..., copy=True) copies the single window out of the memory-mapped
        # file into a writable, contiguous buffer, then normalizes it.
        raw_window = np.array(self.X[idx], copy=True)
        window = torch.from_numpy(normalize_window(raw_window))
        return window, self.y[idx]


def build_dataloaders():
    train_ds = TripleBarrierDataset(
        os.path.join(DATA_DIR, "X_train.npy"), os.path.join(DATA_DIR, "y_train.npy")
    )
    val_ds = TripleBarrierDataset(
        os.path.join(DATA_DIR, "X_val.npy"), os.path.join(DATA_DIR, "y_val.npy")
    )
    test_ds = TripleBarrierDataset(
        os.path.join(DATA_DIR, "X_test.npy"), os.path.join(DATA_DIR, "y_test.npy")
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds


def compute_class_weights(train_ds: "TripleBarrierDataset") -> torch.Tensor:
    """Inverse-frequency ('balanced') class weights from the training labels,
    so the minority classes cost more when misclassified during training."""
    y_train = train_ds.y.numpy()
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(NUM_CLASSES), y=y_train
    )
    return torch.tensor(weights, dtype=torch.float32)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class TripleBarrierLSTM(nn.Module):
    """Deep LSTM on normalized OHLCV windows -> 3-way classification head."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len=100, input_size=5)
        lstm_out, _ = self.lstm(x)          # lstm_out: (batch, seq_len, hidden_size)
        last_step = lstm_out[:, -1, :]      # hidden state at day 100 (final timestep)
        logits = self.fc(last_step)         # (batch, num_classes)
        return logits


# ----------------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    """Runs one pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    torch.set_grad_enabled(is_train)
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        if is_train:
            loss.backward()
            # Clips exploding gradients, which stacked LSTMs are prone to over
            # 100 timesteps; cheap insurance and doesn't affect well-behaved batches.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
    torch.set_grad_enabled(True)

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, accuracy, macro_f1


def train():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, test_loader, train_ds = build_dataloaders()
    print(
        f"Train batches: {len(train_loader)} | "
        f"Val batches: {len(val_loader)} | "
        f"Test batches: {len(test_loader)}"
    )

    class_weights = compute_class_weights(train_ds).to(DEVICE)
    print(f"Class weights (SL, Time Limit, TP): {class_weights.tolist()}")

    model = TripleBarrierLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer=None)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )

        # Champion model saver: keep the checkpoint with the best validation macro-F1.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> New best model saved (val macro F1 = {best_val_f1:.4f})")

    print(f"\nTraining complete. Best validation macro F1: {best_val_f1:.4f}")

    # ------------------------------------------------------------------------
    # Final out-of-sample evaluation using the champion checkpoint.
    # ------------------------------------------------------------------------
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    test_loss, test_acc, test_f1 = run_epoch(model, test_loader, criterion, optimizer=None)

    print("\n===== Final Test Set Performance (best checkpoint) =====")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")


if __name__ == "__main__":
    train()
