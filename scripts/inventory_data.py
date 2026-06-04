#!/usr/bin/env python3
"""Simple inventory of CSV files under a data folder.

Writes a JSON list with entries: {"path":..., "rows":..., "columns": {col: dtype}, "sample_values": {col: sample}}

Usage:
  python scripts/inventory_data.py --input data/raw/retailrocket --out data/inventory.json
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def inspect_csv(path: Path, sample_rows: int = 100):
    info = {"path": str(path), "rows": 0, "columns": {}, "sample_values": {}}
    try:
        # count rows (safe fallback if file is large)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            rows = sum(1 for _ in f) - 1
            info["rows"] = max(rows, 0)
    except Exception:
        info["rows"] = None

    try:
        df = pd.read_csv(path, nrows=sample_rows)
        for col in df.columns:
            dtype = str(df[col].dtype)
            info["columns"][col] = dtype
            sample = None
            try:
                s = df[col].dropna()
                if len(s) > 0:
                    sample = s.iloc[0]
            except Exception:
                sample = None
            info["sample_values"][col] = None if sample is None else str(sample)
    except Exception:
        # if pandas fails, leave columns empty
        pass

    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    if not inp.exists():
        print(f"Input folder {inp} does not exist — writing empty inventory.")
    else:
        for path in sorted(inp.rglob("*.csv")):
            print(f"Inspecting {path}")
            info = inspect_csv(path)
            results.append(info)

    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote inventory to {out}")


if __name__ == "__main__":
    main()
