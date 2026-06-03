# Dataset — RetailRocket Ecommerce Dataset

## Dataset Source

This project uses the RetailRocket Ecommerce Dataset from Kaggle.

Dataset page: RetailRocket Ecommerce Dataset / Retailrocket Recommender System Dataset.

The raw data is not committed to this repository. It should be downloaded locally from Kaggle.

## Why This Dataset

RetailRocket is suitable for this project because it represents a real-world ecommerce recommendation scenario with implicit feedback.

It contains user-item interaction events such as product views, add-to-cart events, and transactions. This makes it useful for studying:

* implicit feedback recommendation
* sparse user-item interactions
* long-tail item distribution
* popularity bias
* candidate retrieval
* top-K recommendation evaluation

## Expected Raw Files

After downloading and extracting the dataset, the expected files are:

```text
data/raw/retailrocket/events.csv
data/raw/retailrocket/item_properties.csv
data/raw/retailrocket/category_tree.csv
```

## Data Files

### events.csv

Main user-item interaction log.

Expected columns:

```text
timestamp
visitorid
event
itemid
transactionid
```

Example event types:

```text
view
addtocart
transaction
```

This file is the main source for training implicit-feedback recommendation models.

### item_properties.csv

Item metadata/properties.

Expected columns:

```text
timestamp
itemid
property
value
```

This file can be used to enrich the item tower with product metadata.

### category_tree.csv

Category hierarchy.

Expected columns:

```text
categoryid
parentid
```

This file can be used for category-level analysis or item/category feature engineering.

## How to Download

Install Kaggle CLI:

```bash
pip install kaggle
```

Configure Kaggle API credentials:

```text
~/.kaggle/kaggle.json
```

Download the dataset:

```bash
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw/retailrocket --unzip
```

## Data Storage Policy

Raw dataset files are not committed to GitHub.

Reasons:

* The dataset can be large.
* The source dataset is hosted on Kaggle.
* The repository should stay lightweight and reproducible.
* Users should download the dataset from the original source.

The repository only includes:

```text
data/sample/
notebooks/
src/
docs/
configs/
```

## Dataset Review Checklist

During mentor review, this project will show:

* dataset source and download command
* raw file structure
* row counts
* event type distribution
* number of unique visitors
* number of unique items
* sparsity analysis
* long-tail item distribution
* example user interaction histories
* example positive pairs for two-tower training
* train/validation/test split strategy

## Dataset Role in the Project

The dataset is converted into implicit feedback pairs:

```text
visitor_id → item_id
```

Event strength can be weighted as:

```text
view = weak positive
addtocart = medium positive
transaction = strong positive
```

These pairs are used to train a two-tower retrieval model:

```text
User tower: visitor/session behavior → user embedding
Item tower: item ID/metadata/category → item embedding
Similarity: dot product / cosine similarity
Output: top-K recommended items
```
