"""
Triple-Barrier OHLCV dataset builder for Nifty500 daily data.

Reproduces the data pipeline from:
"Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data:
Evidence from Korean Markets"

Pipeline stages (each is an independent, callable function so the script
can be re-run/resumed at any stage):

    1. split_master_csv_by_ticker() -> raw_data/<TICKER>.csv   (disk mgmt)
    2. build_dataset()              -> loops over raw_data/*.csv, slides a
                                        100-day window over each stock,
                                        labels it with the Triple Barrier
                                        Method over the next 29 days, and
                                        buckets the result into
                                        train/val/test by a GLOBAL date cut
                                        (no per-stock split -> no leakage).
    3. main()                        -> orchestrates 1+2, prints shapes and
                                         class balance, optionally persists
                                         the final arrays to processed_data/.

Performance note
-----------------
The triple-barrier scan ("walk 29 days forward, see which barrier is hit
first") is naively an O(rows * horizon) Python loop. Instead we vectorize
it per-stock with numpy.lib.stride_tricks.sliding_window_view: for every
window we need only the *first* index at which High/Low breaches the
upper/lower barrier, which is exactly `argmax` of a boolean array (argmax
returns the index of the first True). This produces results identical to
a day-by-day early-stopping loop but runs as compiled numpy code instead
of a Python for-loop over 1.7M+ rows.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

WINDOW_SIZE = 100          # lookback window length (trading days)
FORWARD_HORIZON = 29       # days looked forward to evaluate the barriers
BARRIER_PCT = 0.09         # +/- 9% triple-barrier threshold
MIN_STOCK_LENGTH = WINDOW_SIZE + FORWARD_HORIZON  # 129 rows minimum

FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Global chronological split boundaries (inclusive), based on the date of
# the LAST day in each 100-day lookback window.
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_START = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-02-28")   # dataset coverage ends "early 2026"

# float32 keeps the final arrays ~4 bytes/value (millions of (100,5)
# windows add up fast); precision loss is negligible for OHLC prices and
# for volume figures below 2**24 (~16.7M shares/day), which covers all but
# the most extreme single-day volume spikes.
FEATURE_DTYPE = np.float32
LABEL_DTYPE = np.int8


# --------------------------------------------------------------------------
# Stage 1: chronological file splitting (disk management)
# --------------------------------------------------------------------------

def _sanitize_filename(ticker: str) -> str:
    """Strip characters that are illegal in Windows filenames."""
    return "".join(c if c not in '<>:"/\\|?*' else "_" for c in ticker)


def split_master_csv_by_ticker(
    master_csv_path: Path,
    raw_data_dir: Path,
    force: bool = False,
) -> list[Path]:
    """
    Read the single master CSV once, split it per-Ticker, sort each
    stock's rows chronologically, and write one CSV per stock into
    `raw_data_dir`. This lets stage 2 stream one small file at a time
    instead of holding the ~150MB+ master file in memory repeatedly.

    Idempotent: if raw_data_dir already contains files and force=False,
    the (expensive) master-file read is skipped entirely.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    existing = list(raw_data_dir.glob("*.csv"))
    if existing and not force:
        print(f"[split] {len(existing)} per-ticker files already exist in "
              f"{raw_data_dir} - skipping re-split (pass force=True to redo).")
        return existing

    print(f"[split] reading master file {master_csv_path} ...")
    t0 = time.time()
    df = pd.read_csv(
        master_csv_path,
        dtype={
            "Ticker": "string",
            "Company": "string",
            "Open": "float64",
            "High": "float64",
            "Low": "float64",
            "Close": "float64",
            "Volume": "float64",
        },
        parse_dates=["Datetime"],
    )
    # Daily bars all share the same intraday timestamp component; drop the
    # timezone offset and normalize to midnight so later date comparisons
    # ("<= TRAIN_END") work purely on calendar dates.
    if df["Datetime"].dt.tz is not None:
        df["Datetime"] = df["Datetime"].dt.tz_localize(None)
    df["Datetime"] = df["Datetime"].dt.normalize()

    print(f"[split] loaded {len(df):,} rows for {df['Ticker'].nunique()} "
          f"tickers in {time.time() - t0:.1f}s")

    written = []
    for ticker, group in df.groupby("Ticker", sort=False, observed=True):
        group = group.sort_values("Datetime")
        out_path = raw_data_dir / f"{_sanitize_filename(str(ticker))}.csv"
        group.to_csv(out_path, index=False)
        written.append(out_path)

    print(f"[split] wrote {len(written)} per-ticker CSVs to {raw_data_dir} "
          f"in {time.time() - t0:.1f}s total")
    return written


# --------------------------------------------------------------------------
# Stage 2: rolling window + vectorized triple-barrier labeling
# --------------------------------------------------------------------------

def _load_stock_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={c: "float64" for c in FEATURE_COLS},
        parse_dates=["Datetime"],
    )
    if df["Datetime"].dt.tz is not None:
        df["Datetime"] = df["Datetime"].dt.tz_localize(None)
    df["Datetime"] = df["Datetime"].dt.normalize()
    # Defensive re-sort: stage 1 already sorts, but stage 2 must never
    # silently mislabel a stock if it's ever run against a foreign file.
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


def compute_windows_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized triple-barrier labeling for a single stock.

    Returns
    -------
    feature_windows : (num_windows, WINDOW_SIZE, 5) float array
        Raw [Open, High, Low, Close, Volume] for each lookback window.
    labels : (num_windows,) int8 array
        1 = upper barrier (take profit) hit first
       -1 = lower barrier (stop loss) hit first
        0 = both hit same day (neutralized) OR neither hit (time limit)
    end_dates : (num_windows,) datetime64 array
        Datetime of the FINAL day of each lookback window - this is what
        decides train/val/test bucket membership.
    """
    n = len(df)
    num_windows = n - WINDOW_SIZE - FORWARD_HORIZON + 1
    if num_windows <= 0:
        empty_feat = np.empty((0, WINDOW_SIZE, 5), dtype=FEATURE_DTYPE)
        empty_lab = np.empty((0,), dtype=LABEL_DTYPE)
        empty_dates = np.empty((0,), dtype="datetime64[ns]")
        return empty_feat, empty_lab, empty_dates

    ohlcv = df[FEATURE_COLS].to_numpy(dtype=np.float64)  # (n, 5)
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    close = ohlcv[:, 3]
    dates = df["Datetime"].to_numpy()

    # --- Input feature windows -------------------------------------------------
    # sliding_window_view(ohlcv, WINDOW_SIZE, axis=0) -> shape (n-W+1, 5, W);
    # swapaxes puts it back into the requested (num_windows, W, 5) layout.
    # This is a zero-copy view; the copy only happens later when we filter
    # by train/val/test boolean mask.
    all_windows = sliding_window_view(ohlcv, WINDOW_SIZE, axis=0).swapaxes(1, 2)
    feature_windows = all_windows[:num_windows]  # drop windows with no full horizon

    # --- Triple barrier ----------------------------------------------------
    # Baseline = Close on the final day of each window.
    baseline = close[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + num_windows]
    upper_barrier = baseline * (1.0 + BARRIER_PCT)
    lower_barrier = baseline * (1.0 - BARRIER_PCT)

    # Forward horizon slices for High/Low, one row of length FORWARD_HORIZON
    # per window, aligned so horizon_high[w, d] = High on day (d+1) after
    # window w's final day.
    horizon_high = sliding_window_view(high[WINDOW_SIZE:], FORWARD_HORIZON)[:num_windows]
    horizon_low = sliding_window_view(low[WINDOW_SIZE:], FORWARD_HORIZON)[:num_windows]

    hit_upper = horizon_high >= upper_barrier[:, None]
    hit_lower = horizon_low <= lower_barrier[:, None]

    any_upper = hit_upper.any(axis=1)
    any_lower = hit_lower.any(axis=1)

    # argmax on a boolean array returns the index of the FIRST True value,
    # which is exactly "first day the barrier is breached". Use a sentinel
    # of FORWARD_HORIZON (an index that can never occur) for "never hit".
    first_upper_day = np.where(any_upper, hit_upper.argmax(axis=1), FORWARD_HORIZON)
    first_lower_day = np.where(any_lower, hit_lower.argmax(axis=1), FORWARD_HORIZON)

    labels = np.zeros(num_windows, dtype=LABEL_DTYPE)
    labels[first_upper_day < first_lower_day] = 1
    labels[first_lower_day < first_upper_day] = -1
    # Equal indices (including the "neither ever hit" sentinel-vs-sentinel
    # case) fall through and stay 0: this covers BOTH the "same-day hit ->
    # neutralize" rule and the "time limit reached -> 0" rule.

    end_dates = dates[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + num_windows]

    return feature_windows.astype(FEATURE_DTYPE), labels, end_dates


def _split_masks(end_dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks bucketing windows by the GLOBAL date of their final day."""
    train_mask = end_dates <= TRAIN_END.to_datetime64()
    val_mask = (end_dates >= VAL_START.to_datetime64()) & (end_dates <= VAL_END.to_datetime64())
    test_mask = (end_dates >= TEST_START.to_datetime64()) & (end_dates <= TEST_END.to_datetime64())
    return train_mask, val_mask, test_mask


def _count_windows_per_split(path: Path) -> tuple[int, int, int]:
    """
    Cheap first-pass count: read ONLY the Datetime column (skip the 4
    OHLCV float columns entirely) to learn how many windows this stock
    contributes to each split, without materializing any (100, 5) arrays.
    """
    dates_df = pd.read_csv(path, usecols=["Datetime"], parse_dates=["Datetime"])
    dates = dates_df["Datetime"]
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    dates = dates.dt.normalize().to_numpy()

    n = len(dates)
    num_windows = n - WINDOW_SIZE - FORWARD_HORIZON + 1
    if num_windows <= 0:
        return 0, 0, 0

    end_dates = dates[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + num_windows]
    train_mask, val_mask, test_mask = _split_masks(end_dates)
    return int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())


def build_dataset(
    raw_data_dir: Path,
    processed_dir: Path,
    progress_every: int = 50,
) -> dict[str, np.ndarray]:
    """
    Loop over every per-ticker CSV in raw_data_dir, slide the 100-day
    window, triple-barrier label it, and bucket into train/val/test by the
    GLOBAL date split (never a per-stock percentage split - that would leak
    future dates from one stock into the "past" of another).

    Two-pass, pre-allocated, disk-backed design
    ---------------------------------------------
    A naive "append each stock's arrays to a Python list, np.concatenate
    at the end" approach needs roughly 2x the final dataset size in RAM at
    the moment of concatenation (the list's chunks plus the newly
    allocated contiguous output). With 500+ stocks and multi-million
    (100, 5) windows that easily exceeds available memory on a modest
    machine. Instead we:
      Pass 1: read only the (tiny) Datetime column per stock to count
              exactly how many windows land in each split.
      Pass 2: pre-allocate X_train/X_val/X_test as disk-backed
              `np.memmap` .npy files at their EXACT final shape and copy
              each stock's windows directly into the right slice. This
              keeps peak *resident* memory bounded to roughly one stock's
              data at a time (the OS pages the memmap to/from disk as
              needed) instead of requiring one giant contiguous RAM
              allocation, which is what caused an outright MemoryError on
              a RAM-constrained machine during development of this
              pipeline. The small integer label arrays (y_*) stay in
              plain RAM since they're only a few MB even at millions of
              rows.
    """
    stock_files = sorted(raw_data_dir.glob("*.csv"))
    print(f"[build] found {len(stock_files)} stock files")

    # --- Pass 1: count windows per split (cheap: Datetime column only) -----
    t0 = time.time()
    per_stock_counts = [_count_windows_per_split(p) for p in stock_files]
    totals = {
        "train": sum(c[0] for c in per_stock_counts),
        "val": sum(c[1] for c in per_stock_counts),
        "test": sum(c[2] for c in per_stock_counts),
    }
    print(f"[build] pass 1 (counting) done in {time.time() - t0:.1f}s: "
          f"train={totals['train']:,} val={totals['val']:,} test={totals['test']:,}")

    # --- Pre-allocate final arrays at exact size (X on disk, y in RAM) -----
    processed_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    cursors = {}
    for name in ("train", "val", "test"):
        x_path = processed_dir / f"X_{name}.npy"
        arrays[f"X_{name}"] = np.lib.format.open_memmap(
            x_path, mode="w+", dtype=FEATURE_DTYPE,
            shape=(totals[name], WINDOW_SIZE, 5),
        )
        arrays[f"y_{name}"] = np.empty((totals[name],), dtype=LABEL_DTYPE)
        cursors[name] = 0

    # --- Pass 2: compute windows/labels per stock, write into place --------
    n_skipped = 0
    t0 = time.time()
    for i, (path, counts) in enumerate(zip(stock_files, per_stock_counts), start=1):
        if sum(counts) == 0:
            n_skipped += 1
            continue

        df = _load_stock_csv(path)
        feature_windows, labels, end_dates = compute_windows_and_labels(df)
        train_mask, val_mask, test_mask = _split_masks(end_dates)

        for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            n_hits = int(mask.sum())
            if n_hits == 0:
                continue
            start = cursors[name]
            arrays[f"X_{name}"][start:start + n_hits] = feature_windows[mask]
            arrays[f"y_{name}"][start:start + n_hits] = labels[mask]
            cursors[name] += n_hits

        if progress_every and i % progress_every == 0:
            elapsed = time.time() - t0
            print(f"[build] {i}/{len(stock_files)} stocks processed "
                  f"({elapsed:.1f}s elapsed)")

    print(f"[build] done: {len(stock_files) - n_skipped} stocks used, "
          f"{n_skipped} skipped (< {MIN_STOCK_LENGTH} rows of history), "
          f"{time.time() - t0:.1f}s total")

    # Sanity check: every pre-allocated slot must have been filled exactly.
    for name in ("train", "val", "test"):
        assert cursors[name] == totals[name], (
            f"{name}: filled {cursors[name]} of {totals[name]} pre-allocated slots"
        )
        arrays[f"X_{name}"].flush()  # ensure all memmap writes hit disk

    return arrays


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
    print(f"\n{name} set:")
    print(f"  X_{name} shape: {X.shape}   y_{name} shape: {y.shape}")
    total = len(y)
    if total == 0:
        print("  (empty)")
        return
    values, counts = np.unique(y, return_counts=True)
    dist = dict(zip(values.tolist(), counts.tolist()))
    for label in (1, 0, -1):
        count = dist.get(label, 0)
        pct = 100.0 * count / total
        tag = {1: "Take Profit (+1)", 0: "Time Limit (0)", -1: "Stop Loss (-1)"}[label]
        print(f"  {tag:<18} {count:>9,} windows  ({pct:5.2f}%)")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", type=Path,
                         default=Path("dataset/nifty500_1d.csv"))
    parser.add_argument("--raw-data-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--force-resplit", action="store_true",
                         help="Re-split the master CSV even if raw_data/ is already populated.")
    parser.add_argument("--save-arrays", action="store_true", default=True,
                         help="Persist final X/y arrays as .npy files in --processed-dir.")
    parser.add_argument("--no-save-arrays", dest="save_arrays", action="store_false")
    args = parser.parse_args()

    # Stage 1: disk management.
    split_master_csv_by_ticker(args.master_csv, args.raw_data_dir, force=args.force_resplit)

    # Stages 2-4: rolling window extraction, triple-barrier labeling,
    # global chronological bucketing, array consolidation. X_* arrays are
    # written directly to --processed-dir as disk-backed memmaps (see
    # build_dataset docstring for why); y_* stay in RAM and are saved below.
    arrays = build_dataset(args.raw_data_dir, args.processed_dir)

    for name in ("train", "val", "test"):
        report_split(name, arrays[f"X_{name}"], arrays[f"y_{name}"])

    if args.save_arrays:
        for name in ("train", "val", "test"):
            np.save(args.processed_dir / f"y_{name}.npy", arrays[f"y_{name}"])
        print(f"\n[save] X_*.npy (memmap) and y_*.npy arrays are in {args.processed_dir}/")
    else:
        # On Windows a memmap must be released (garbage collected) before
        # its backing file can be deleted, otherwise unlink() raises
        # PermissionError because the mmap still holds the file open.
        for name in ("train", "val", "test"):
            del arrays[f"X_{name}"]
        gc.collect()
        for name in ("train", "val", "test"):
            (args.processed_dir / f"X_{name}.npy").unlink(missing_ok=True)
        print(f"\n[cleanup] --no-save-arrays: removed X_*.npy from {args.processed_dir}/")


if __name__ == "__main__":
    main()
