from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# db_table -> parquet_stem  (the 5 tables the API serves from)
_TABLES = {
    "sales": "transactions",
    "customer_segments": "customer_segments",
    "forecast_results": "forecast_results",
    "churn_predictions": "churn_predictions",
    "inventory": "inventory",
}

# chunksize * max_columns must stay below SQLite's 32,766 bind-param cap.
CHUNK = 2000


def get_engine() -> Engine:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    return create_engine(url, pool_pre_ping=True)


def _to_object(df: pd.DataFrame) -> pd.DataFrame:
    """pandas 'string' dtype -> object, so every DB driver accepts it."""
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype) == "string":
            df[c] = df[c].astype(object)
    return df


def load_all(engine: Engine | None = None) -> None:
    engine = engine or get_engine()

    # --- analytical tables the API serves ---
    for table, stem in _TABLES.items():
        df = pd.read_parquet(PROCESSED / f"{stem}.parquet")
        if table == "sales" and "Date" in df.columns:
            df = df.drop(columns=["Date"])          # aux col the API doesn't use
        df = _to_object(df)
        df.to_sql(table, engine, if_exists="replace", index=False,
                  chunksize=CHUNK, method="multi")
        print(f"  loaded {table:<20} {len(df):>8,} rows")

    # --- normalized reference tables (brief-compliance) ---
    seg = pd.read_parquet(PROCESSED / "customer_segments.parquet")
    customers = seg[["CustomerID", "Country", "FirstPurchase", "LastPurchase"]] \
        .rename(columns=str.lower)
    _to_object(customers).to_sql("customers", engine, if_exists="replace",
                                 index=False, chunksize=CHUNK, method="multi")
    print(f"  loaded {'customers':<20} {len(customers):>8,} rows")

    tx = pd.read_parquet(PROCESSED / "transactions.parquet")
    products = (tx.groupby("StockCode")
                  .agg(description=("Description",
                                    lambda s: s.mode().iat[0] if len(s.mode()) else None),
                       avg_price=("Price", "mean"))
                  .reset_index().rename(columns={"StockCode": "stock_code"}))
    _to_object(products).to_sql("products", engine, if_exists="replace",
                                index=False, chunksize=CHUNK, method="multi")
    print(f"  loaded {'products':<20} {len(products):>8,} rows")

    _create_indexes(engine)
    print("Done.")


def _create_indexes(engine: Engine) -> None:
    """Speed up the common filters. Quoted identifiers keep CamelCase intact."""
    stmts = [
        'CREATE INDEX IF NOT EXISTS idx_sales_customer ON sales ("CustomerID")',
        'CREATE INDEX IF NOT EXISTS idx_sales_date     ON sales ("InvoiceDate")',
        'CREATE INDEX IF NOT EXISTS idx_sales_stock    ON sales ("StockCode")',
        'CREATE INDEX IF NOT EXISTS idx_churn_customer ON churn_predictions (customer_id)',
    ]
    with engine.begin() as conn:
        for s in stmts:
            try:
                conn.execute(text(s))
            except Exception as e:      # non-fatal (e.g. engine quirks)
                print(f"  (index skipped: {e})")


def check(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print(f"Connection OK -> {engine.url.render_as_string(hide_password=True)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load", action="store_true")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    eng = get_engine()
    if a.check:
        check(eng)
    if a.load:
        load_all(eng)
    if not (a.check or a.load):
        print("Nothing to do. Pass --load and/or --check.")