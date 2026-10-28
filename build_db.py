"""
build_db.py - load the 9 Olist CSVs into a SQLite database.

Used by the notebook (Part 1) and by the SQL agent, so both always see the same schema.

    python build_db.py                      # reads ./data, writes ./olist.db
    python build_db.py --data-dir path/to/csvs --db olist.db

What it does
  1. Reads each CSV with zip-code columns kept as text (so "01001" keeps its leading zero).
  2. Writes each one as a table with a short, readable name (orders, order_items, ...).
  3. Adds indexes on the join keys, so the analysis queries and the agent stay fast.
  4. Creates one helper view, v_order_delivery, with delivery days and the late flag
     already computed (one definition of "late" shared by the notebook and the agent).
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd

# table name in SQLite  ->  CSV file name from Kaggle
TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

# indexes on the columns we join and filter on
INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_orders_order ON orders(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_orders_customer ON orders(customer_id)",
    "CREATE INDEX IF NOT EXISTS ix_items_order ON order_items(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_items_product ON order_items(product_id)",
    "CREATE INDEX IF NOT EXISTS ix_items_seller ON order_items(seller_id)",
    "CREATE INDEX IF NOT EXISTS ix_customers_id ON customers(customer_id)",
    "CREATE INDEX IF NOT EXISTS ix_customers_unique ON customers(customer_unique_id)",
    "CREATE INDEX IF NOT EXISTS ix_payments_order ON payments(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_reviews_order ON reviews(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_products_id ON products(product_id)",
    "CREATE INDEX IF NOT EXISTS ix_sellers_id ON sellers(seller_id)",
    "CREATE INDEX IF NOT EXISTS ix_geo_zip ON geolocation(geolocation_zip_code_prefix)",
]

# One shared definition of delivery time and lateness.
# "Late" compares calendar dates: the estimate is a date (00:00:00), so an order that
# arrives at 3 pm on the promised day counts as on time.
DELIVERY_VIEW = """
CREATE VIEW IF NOT EXISTS v_order_delivery AS
SELECT o.order_id,
       o.customer_id,
       c.customer_unique_id,
       c.customer_state,
       o.order_purchase_timestamp,
       o.order_delivered_customer_date,
       o.order_estimated_delivery_date,
       julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp) AS delivery_days,
       julianday(o.order_estimated_delivery_date) - julianday(o.order_purchase_timestamp) AS promised_days,
       CASE WHEN date(o.order_delivered_customer_date) > date(o.order_estimated_delivery_date)
            THEN 1 ELSE 0 END AS is_late
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
"""


def build_database(data_dir: str | Path = "data", db_path: str | Path = "olist.db",
                   verbose: bool = True) -> Path:
    """Create (or rebuild) the SQLite database from the Olist CSVs. Returns the db path."""
    data_dir, db_path = Path(data_dir), Path(db_path)
    missing = [f for f in TABLES.values() if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing CSVs in {data_dir.resolve()}: {missing}. "
            "Download the Olist dataset from Kaggle and unzip it into this folder."
        )

    if db_path.exists():
        db_path.unlink()  # rebuild from scratch so reruns are reproducible

    start = time.time()
    con = sqlite3.connect(db_path)
    try:
        for table, file_name in TABLES.items():
            # keep zip-code prefixes as strings; everything else lets pandas infer
            df = pd.read_csv(data_dir / file_name,
                             dtype={c: str for c in ("customer_zip_code_prefix",
                                                     "seller_zip_code_prefix",
                                                     "geolocation_zip_code_prefix")})
            df.to_sql(table, con, if_exists="replace", index=False, chunksize=50_000)
            if verbose:
                print(f"  loaded {table:<22} {len(df):>9,} rows")
        for stmt in INDEXES:
            con.execute(stmt)
        con.execute(DELIVERY_VIEW)
        con.commit()
    finally:
        con.close()

    if verbose:
        size_mb = db_path.stat().st_size / 1e6
        print(f"Built {db_path} ({size_mb:,.0f} MB) in {time.time() - start:.1f}s")
    return db_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load the Olist CSVs into SQLite.")
    parser.add_argument("--data-dir", default="data", help="folder with the 9 Olist CSVs")
    parser.add_argument("--db", default="olist.db", help="output SQLite file")
    args = parser.parse_args()
    build_database(args.data_dir, args.db)
