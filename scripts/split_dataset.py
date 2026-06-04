#!/usr/bin/env python3
"""Chronological / time-based deterministic dataset splitter.

Reads: data/processed/dataset.parquet
Writes:
  data/processed/splits/train.parquet
  data/processed/splits/val.parquet
  data/processed/splits/test.parquet
  data/processed/splits/split_report.json

Validations performed:
 - splits are non-empty
 - chronological ordering preserved
 - train.max_timestamp + min_gap_seconds <= val.min_timestamp
 - val.max_timestamp + min_gap_seconds <= test.min_timestamp

Default ratios: train 80%, val 10%, test 10%

Usage:
  python scripts/split_dataset.py --input data/processed/dataset.parquet --out_dir data/processed/splits
"""
from pathlib import Path
import argparse
import json
import sys
import math

import pandas as pd


def read_table(path: Path):
    if path.suffix == '.parquet' and path.exists():
        return pd.read_parquet(path)
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(f"Input file not found: {path}")


def write_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix('.csv'), index=False)


def summarize_split(df: pd.DataFrame):
    if df.empty:
        return {
            'rows': 0,
            'ts_min': None,
            'ts_max': None,
            'unique_users': 0,
            'unique_items': 0,
            'event_counts': {}
        }

    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    users_col = 'visitorid' if 'visitorid' in df.columns else ('visitor_id' if 'visitor_id' in df.columns else None)
    items_col = 'itemid' if 'itemid' in df.columns else ('item_id' if 'item_id' in df.columns else None)

    event_counts = df['event'].value_counts().to_dict() if 'event' in df.columns else {}

    return {
        'rows': int(len(df)),
        'ts_min': str(ts_min),
        'ts_max': str(ts_max),
        'unique_users': int(df[users_col].nunique()) if users_col else None,
        'unique_items': int(df[items_col].nunique()) if items_col else None,
        'event_counts': event_counts
    }


def validate_splits(report: dict, min_gap_seconds: int = 15):
    # basic non-empty
    for name in ['train', 'val', 'test']:
        if report[name]['rows'] == 0:
            return False, f"Split '{name}' is empty"

    # chronological boundaries
    try:
        train_max = pd.to_datetime(report['train']['ts_max'])
        val_min = pd.to_datetime(report['val']['ts_min'])
        val_max = pd.to_datetime(report['val']['ts_max'])
        test_min = pd.to_datetime(report['test']['ts_min'])
    except Exception as e:
        return False, f"Timestamp parse error: {e}"

    if not (train_max < val_min):
        return False, "Train max timestamp is not less than Val min timestamp"

    if (train_max + pd.Timedelta(seconds=min_gap_seconds)) > val_min:
        return False, f"Train/Val gap smaller than {min_gap_seconds} seconds"

    if not (val_max < test_min):
        return False, "Val max timestamp is not less than Test min timestamp"

    if (val_max + pd.Timedelta(seconds=min_gap_seconds)) > test_min:
        return False, f"Val/Test gap smaller than {min_gap_seconds} seconds"

    return True, "OK"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/processed/dataset.parquet')
    p.add_argument('--out_dir', default='data/processed/splits')
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--test_ratio', type=float, default=0.1)
    p.add_argument('--min_gap_seconds', type=int, default=15)
    args = p.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)

    df = read_table(inp)

    if 'timestamp' not in df.columns:
        # try common alternatives
        for alt in ['ts', 'time', 'event_time']:
            if alt in df.columns:
                df.rename(columns={alt: 'timestamp'}, inplace=True)
                break

    # coerce timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    # drop rows without timestamp
    before = len(df)
    df = df[df['timestamp'].notna()]
    after = len(df)
    if after < before:
        print(f"Dropped {before-after} rows without valid timestamp")

    # sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    n = len(df)
    if n == 0:
        print('No rows to split after timestamp parsing. Exiting.')
        sys.exit(1)

    # deterministic split by position preserving chronology
    train_end = int(math.floor(n * args.train_ratio))
    val_end = train_end + int(math.floor(n * args.val_ratio))

    # ensure non-empty
    if train_end <= 0:
        train_end = 1
    if val_end <= train_end:
        val_end = train_end + 1
    if val_end >= n:
        val_end = n - 1

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # write outputs
    write_table(train_df, out_dir / 'train.parquet')
    write_table(val_df, out_dir / 'val.parquet')
    write_table(test_df, out_dir / 'test.parquet')

    report = {
        'train': summarize_split(train_df),
        'val': summarize_split(val_df),
        'test': summarize_split(test_df),
        'total_rows': int(n)
    }

    ok, msg = validate_splits(report, min_gap_seconds=args.min_gap_seconds)

    report['validation'] = {
        'ok': bool(ok),
        'message': str(msg),
        'min_gap_seconds': int(args.min_gap_seconds)
    }

    out_report = out_dir / 'split_report.json'
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"Wrote splits to {out_dir} and report to {out_report}")
    if not ok:
        print('Validation failed:', msg)
        sys.exit(2)


if __name__ == '__main__':
    main()
