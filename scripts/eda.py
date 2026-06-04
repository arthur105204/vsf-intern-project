#!/usr/bin/env python3
"""Simple EDA for processed dataset.

Produces `reports/eda_summary.md` with key statistics: total interactions, unique users, unique items,
event type counts, sparsity, and top categories.

Usage:
  python scripts/eda.py --input data/processed/dataset.parquet --out reports/eda_summary.md
"""
import argparse
from pathlib import Path
import pandas as pd


def load_data(path: Path):
    if path.suffix == '.parquet' and path.exists():
        return pd.read_parquet(path)
    if path.exists():
        return pd.read_csv(path)
    # try dataset.csv in same folder
    csv_path = path.with_suffix('.csv')
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Input data not found at {path}")


def summarize(df: pd.DataFrame):
    out = {}
    out['interactions'] = int(len(df))
    out['unique_users'] = int(df['visitorid'].nunique()) if 'visitorid' in df.columns else int(df['visitor_id'].nunique())
    out['unique_items'] = int(df['itemid'].nunique()) if 'itemid' in df.columns else int(df['item_id'].nunique())
    if 'event' in df.columns:
        out['event_counts'] = df['event'].value_counts().to_dict()
    else:
        out['event_counts'] = {}

    # sparsity: interactions / (users * items)
    u = out['unique_users']
    m = out['unique_items']
    out['sparsity'] = out['interactions'] / (u * m) if u and m else None

    # top categories if available
    if 'category' in df.columns:
        out['top_categories'] = df['category'].value_counts().head(10).to_dict()
    else:
        out['top_categories'] = {}

    # example histories: sample 3 users
    sample_users = df['visitorid'].dropna().unique()[:3] if 'visitorid' in df.columns else []
    samples = {}
    for u in sample_users:
        seq = df[df['visitorid'] == u].sort_values('timestamp').head(10)
        samples[str(u)] = seq[['timestamp', 'event', 'itemid']].to_dict(orient='records')
    out['example_histories'] = samples
    return out


def write_markdown(summary: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.write('# EDA Summary\n\n')
        f.write(f"- total interactions: **{summary['interactions']}**\n")
        f.write(f"- unique users: **{summary['unique_users']}**\n")
        f.write(f"- unique items: **{summary['unique_items']}**\n")
        f.write(f"- sparsity (interactions / users*items): **{summary['sparsity']}**\n\n")
        f.write('## Event counts\n\n')
        for k, v in summary['event_counts'].items():
            f.write(f"- {k}: {v}\n")
        f.write('\n')
        f.write('## Top categories\n\n')
        for k, v in summary['top_categories'].items():
            f.write(f"- {k}: {v}\n")
        f.write('\n')
        f.write('## Example short histories\n\n')
        for u, seq in summary['example_histories'].items():
            f.write(f"### visitor {u}\n")
            for r in seq:
                f.write(f"- {r.get('timestamp')} | {r.get('event')} | {r.get('itemid')}\n")
            f.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/processed/dataset.parquet')
    p.add_argument('--out', default='reports/eda_summary.md')
    args = p.parse_args()

    path = Path(args.input)
    df = load_data(path)
    summary = summarize(df)
    write_markdown(summary, Path(args.out))
    print(f"Wrote EDA summary to {args.out}")


if __name__ == '__main__':
    main()
