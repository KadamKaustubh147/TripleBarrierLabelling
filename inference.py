"""
Inference script for the Triple-Barrier LSTM (Nifty 500).

Two modes:

  1. Live signal (default): loads best_model.pth and predicts on the most
     recent 100-day OHLCV window for one or more tickers, reading straight
     from raw_data/<TICKER>.csv. This is the "what would the model say
     today" use case -- no label is known yet.

       python inference.py --tickers RELIANCE TCS INFY
       python inference.py --all

  2. Batch eval: re-runs the champion checkpoint against a saved split from
     processed_data/ and prints a full classification report + confusion
     matrix (train.py only prints the single macro-F1 number).

       python inference.py --eval test

Reuses TripleBarrierLSTM, normalize_window, TripleBarrierDataset, and the
label constants from train.py so inference preprocessing can never drift
from what the model was actually trained on.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from build_triple_barrier_dataset import FEATURE_COLS, WINDOW_SIZE
from train import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DATA_DIR,
    DEVICE,
    TripleBarrierDataset,
    TripleBarrierLSTM,
    normalize_window,
)


# ----------------------------------------------------------------------------
# Live signal: latest window per ticker
# ----------------------------------------------------------------------------
def load_latest_window(ticker: str, raw_data_dir: Path) -> tuple[np.ndarray, pd.Timestamp]:
    """Reads raw_data/<ticker>.csv and returns its most recent WINDOW_SIZE-day
    OHLCV window, in the same [Open, High, Low, Close, Volume] column order
    the model was trained on."""
    path = raw_data_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no raw data at {path}")

    df = pd.read_csv(path, parse_dates=["Datetime"])
    if df["Datetime"].dt.tz is not None:
        df["Datetime"] = df["Datetime"].dt.tz_localize(None)
    df = df.sort_values("Datetime").reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        raise ValueError(f"only {len(df)} rows of history, need at least {WINDOW_SIZE}")

    window_df = df.iloc[-WINDOW_SIZE:]
    window = window_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    as_of_date = window_df["Datetime"].iloc[-1]
    return window, as_of_date


def predict_window(model: TripleBarrierLSTM, window: np.ndarray) -> tuple[int, np.ndarray]:
    """Runs one normalized (WINDOW_SIZE, 5) window through the model and
    returns the predicted class index plus the full softmax distribution."""
    normalized = normalize_window(window)
    x = torch.from_numpy(normalized).unsqueeze(0).to(DEVICE)  # (1, W, 5)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(probs.argmax()), probs


def run_live_signals(tickers: list[str], raw_data_dir: Path, model: TripleBarrierLSTM) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            window, as_of_date = load_latest_window(ticker, raw_data_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skip] {ticker}: {exc}")
            continue

        pred_class, probs = predict_window(model, window)
        rows.append(
            {
                "ticker": ticker,
                "as_of": as_of_date.date(),
                "prediction": CLASS_NAMES[pred_class],
                "P(Stop Loss)": probs[0],
                "P(Time Limit)": probs[1],
                "P(Take Profit)": probs[2],
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Batch eval: full classification report on a saved split
# ----------------------------------------------------------------------------
def run_batch_eval(split: str, model: TripleBarrierLSTM) -> None:
    x_path = Path(DATA_DIR) / f"X_{split}.npy"
    y_path = Path(DATA_DIR) / f"y_{split}.npy"
    ds = TripleBarrierDataset(str(x_path), str(y_path))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            logits = model(X_batch)
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    print(f"\n===== Classification report ({split} split, {len(ds):,} windows) =====")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(
        confusion_matrix(all_targets, all_preds), index=CLASS_NAMES, columns=CLASS_NAMES
    ))


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", help="Ticker symbols to predict the latest signal for, e.g. RELIANCE TCS")
    parser.add_argument("--all", action="store_true", help="Predict the latest signal for every ticker in raw_data/")
    parser.add_argument("--eval", choices=["train", "val", "test"], help="Batch-evaluate the checkpoint on a saved split instead of live prediction")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--checkpoint", type=Path, default=Path(CHECKPOINT_PATH))
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint} -- run train.py first.")

    model = TripleBarrierLSTM().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (device: {DEVICE})")

    if args.eval:
        run_batch_eval(args.eval, model)
        return

    if args.all:
        tickers = sorted(p.stem for p in args.raw_data_dir.glob("*.csv"))
    elif args.tickers:
        tickers = args.tickers
    else:
        parser.error("specify --tickers TICKER [TICKER ...], --all, or --eval {train,val,test}")
        return

    signals = run_live_signals(tickers, args.raw_data_dir, model)
    if signals.empty:
        print("No predictions produced -- check ticker names / raw_data/ contents.")
        return

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(f"\n===== Latest signal per ticker ({len(signals)} predicted) =====")
    print(signals.to_string(index=False))


if __name__ == "__main__":
    main()
