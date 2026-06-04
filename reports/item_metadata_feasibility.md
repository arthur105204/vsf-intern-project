# Item Metadata Feasibility

## Goal

Decide whether RetailRocket item metadata is usable for the next modeling gate, especially category-like item features.

## Files inspected

- `data/raw/item_properties_part1.csv`
- `data/raw/item_properties_part2.csv`
- `data/raw/category_tree.csv`
- `data/processed/splits/train.parquet`
- current model vocab from `outputs/experiments/history_tower_weighted/vocab.json`

## Schema summary

### `item_properties_part1.csv` / `item_properties_part2.csv`

Columns:

- `timestamp`
- `itemid`
- `property`
- `value`

Observed metadata shape:

- total rows across both files: `20,275,902`
- unique items with metadata rows: `417,053`
- unique property keys: `1,104`

Important observation:

- `property=categoryid` exists explicitly
- `categoryid` appears `788,214` times
- every item with metadata also appears to have a `categoryid` row in these files

Other property IDs such as `888`, `790`, `6`, `283`, and `202` are present at large scale, but they are opaque numeric property IDs and are less immediately usable than `categoryid`.

### `category_tree.csv`

Columns:

- `categoryid`
- `parentid`

Observed structure:

- category nodes: `1,669`
- root categories: `25`

This is a straightforward category hierarchy and is suitable for deriving parent-category features later if needed.

## Can `item_id` be mapped to category-like features?

Yes.

The clearest join path is:

1. map `itemid -> categoryid` from `item_properties`
2. map `categoryid -> parentid` from `category_tree`

This means item-level category features are feasible.

One caution:

- `23,352` items have more than one distinct `categoryid` value over time
- this is about `5.60%` of items with category metadata

So category assignment is mostly stable, but not perfectly static. If category features are added later, a train-safe snapshot rule should be used instead of blindly taking a future category state.

## Coverage

### Metadata coverage in raw item properties

- items with any metadata rows: `417,053`
- items with `categoryid`: `417,053`
- items with `categoryid` also found in `category_tree`: `416,921`

This means category coverage is effectively the same as metadata coverage for this dataset.

### Coverage for experiment-relevant item sets

| Item set | Total items | With metadata | Coverage | With category in tree | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train split unique items | 212,915 | 169,372 | 79.55% | 169,372 | 79.55% |
| Validation ground-truth unique items | 77,696 | 64,930 | 83.57% | 64,930 | 83.57% |
| Test ground-truth unique items | 77,507 | 64,643 | 83.40% | 64,643 | 83.40% |
| Current model vocab items | 100,648 | 89,065 | 88.49% | 89,065 | 88.49% |

## Interpretation

Metadata coverage is high enough to be useful:

- about `80%` of train items have category metadata
- about `83%` of validation/test ground-truth items have category metadata
- about `88%` of current model-vocab items have category metadata

This is much stronger coverage than the current user-history coverage bottleneck, and it suggests category features could add meaningful item-side generalization.

Also, `categoryid` is much cleaner than the other raw property IDs because:

- it is explicit
- it has broad coverage
- it comes with a usable hierarchy

## Recommendation

**Add category embedding now.**

Reason:

- category mapping is clearly available
- coverage is high enough to matter
- current learned models are still well below the popularity baseline
- history features helped, but not enough
- category is the lowest-risk metadata feature to try before broader property engineering

Suggested next implementation rule:

- derive one train-safe category assignment per item
- start with a simple category embedding on the item tower
- leave broader property parsing for a later gate

## If not now, what would block it?

The only meaningful caution is temporal consistency of `categoryid` for a minority of items. That is not a reason to skip category features; it is only a reason to define a careful snapshoting rule when implementing them.
