#!/usr/bin/env python3
"""Prepare dataset: merge item properties and canonicalize events into a processed dataset.

Writes:
- data/processed/dataset.parquet (or dataset.csv if pyarrow missing)
- data/processed/merge_report.csv

Usage:
  python scripts/prepare_dataset.py --raw data/raw --out data/processed
"""
import argparse
from pathlib import Path
import pandas as pd
import csv
import sys


def load_item_properties(raw_dir: Path):
    parts = list(raw_dir.glob('item_properties*.csv'))
    if not parts:
        return pd.DataFrame()
    dfs = []
    for p in sorted(parts):
        try:
            dfs.append(pd.read_csv(p, dtype=str))
        except Exception:
            dfs.append(pd.read_csv(p, dtype=str, engine='python'))
    items = pd.concat(dfs, ignore_index=True)
    # prefer last timestamp for duplicate itemid
    items = items.sort_values('timestamp').drop_duplicates('itemid', keep='last')
    return items


def stream_events(raw_dir: Path, out_path: Path, items_df: pd.DataFrame):
    events_file = raw_dir / 'events.csv'
    if not events_file.exists():
        print(f"events.csv not found in {raw_dir}")
        return 0, 0

    chunksize = 200000
    total_in = 0
    total_out = 0

    # prepare output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        use_parquet = True
    except Exception:
        use_parquet = False

    writer = None

    for chunk in pd.read_csv(events_file, chunksize=chunksize, dtype=str):
        total_in += len(chunk)
        # normalize columns
        # expected columns: timestamp, visitorid, event, itemid, transactionid
        cols = {c.lower(): c for c in chunk.columns}
        # rename to lower-case canonical
        chunk.columns = [c.lower() for c in chunk.columns]

        # parse timestamp robustly (handles ms or s, numeric strings)
        if 'timestamp' in chunk.columns:
            # coerce to numeric first
            chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
            if chunk['timestamp'].notna().any():
                max_ts = chunk['timestamp'].max()
                # heuristics: values > 1e12 => milliseconds, >1e9 => seconds
                if max_ts > 1e12:
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='ms', errors='coerce')
                else:
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='s', errors='coerce')
            else:
                chunk['timestamp'] = pd.NaT
        else:
            chunk['timestamp'] = pd.NaT

        # drop rows without itemid
        if 'itemid' in chunk.columns:
            chunk = chunk[chunk['itemid'].notna()]
        else:
            chunk['itemid'] = None

        # left-join item metadata if available
        if not items_df.empty:
            chunk = chunk.merge(items_df[['itemid']], on='itemid', how='left')

        total_out += len(chunk)

        # write
        if use_parquet:
            table = pa.Table.from_pandas(chunk)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
        else:
            # append CSV
            header = not out_path.exists()
            chunk.to_csv(out_path, mode='a', index=False, header=header)

    if writer is not None:
        writer.close()

    return total_in, total_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default='data/raw', help='raw data folder')
    p.add_argument('--out', default='data/processed/dataset.parquet', help='output dataset path')
    args = p.parse_args()

    raw_dir = Path(args.raw)
    out_path = Path(args.out)

    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    items_df = load_item_properties(raw_dir)
    if not items_df.empty:
        print(f"Loaded item properties: {len(items_df)} items")
    else:
        print("No item properties found or loaded.")

    total_in, total_out = stream_events(raw_dir, out_path, items_df)

    report_path = out_dir / 'merge_report.csv'
    with report_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['stage', 'count'])
        writer.writerow(['events_rows_in', total_in])
        writer.writerow(['events_rows_out', total_out])
        writer.writerow(['items_loaded', len(items_df)])

    print(f"Wrote merge report to {report_path}")


if __name__ == '__main__':
    main()
