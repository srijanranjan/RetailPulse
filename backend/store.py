from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

_SOURCES: dict[str, tuple[str, str, list[str]]] = {
    "transactions": ("sales", "transactions", ["InvoiceDate"]),
    "segments":     ("customer_segments", "customer_segments",
                     ["FirstPurchase", "LastPurchase"]),
    "forecast":     ("forecast_results", "forecast_results", ["ds"]),
    "churn":        ("churn_predictions", "churn_predictions", []),
    "inventory":    ("inventory", "inventory", []),
}


@lru_cache(maxsize=1)
def engine() -> Engine | None:
    """Return a SQLAlchemy engine if DATABASE_URL is set, else None."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    if url.startswith("postgres://"):          # Render's legacy scheme
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    return create_engine(url, pool_pre_ping=True)


def data_source() -> str:
    eng = engine()
    return eng.dialect.name if eng is not None else "parquet"


@lru_cache(maxsize=None)
def _load(entity: str) -> pd.DataFrame:
    db_table, stem, date_cols = _SOURCES[entity]
    eng = engine()
    if eng is not None:
        df = pd.read_sql_table(db_table, eng)
    else:
        df = pd.read_parquet(PROCESSED / f"{stem}.parquet")
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df

def transactions() -> pd.DataFrame:
    return _load("transactions")


def segments() -> pd.DataFrame:
    return _load("segments")


def forecast() -> pd.DataFrame:
    return _load("forecast")


def churn() -> pd.DataFrame:
    return _load("churn")


def inventory() -> pd.DataFrame:
    return _load("inventory")

def _pg() -> Engine | None:
    eng = engine()
    return eng if (eng is not None and eng.dialect.name == "postgresql") else None


def kpis() -> dict:
    inv = inventory()
    low = int((inv["alert"] == "LOW_STOCK").sum())
    over = int((inv["alert"] == "OVERSTOCK").sum())
    eng = _pg()
    if eng is not None:
        sql = text('''
            SELECT COALESCE(SUM("TotalPrice"), 0) AS rev,
                   COUNT(DISTINCT "Invoice")      AS orders,
                   COUNT(DISTINCT "CustomerID")   AS customers,
                   COUNT(DISTINCT "StockCode")    AS products,
                   MIN("InvoiceDate")             AS dfrom,
                   MAX("InvoiceDate")             AS dto
            FROM sales
        ''')
        with eng.connect() as c:
            r = c.execute(sql).mappings().one()
        return {
            "total_revenue": round(float(r["rev"]), 2),
            "total_orders": int(r["orders"]),
            "total_customers": int(r["customers"]),
            "total_products": int(r["products"]),
            "low_stock_alerts": low,
            "overstock_alerts": over,
            "date_from": str(pd.to_datetime(r["dfrom"]).date()),
            "date_to": str(pd.to_datetime(r["dto"]).date()),
            "data_source": data_source(),
        }
    tx = transactions()
    return {
        "total_revenue": round(float(tx["TotalPrice"].sum()), 2),
        "total_orders": int(tx["Invoice"].nunique()),
        "total_customers": int(tx["CustomerID"].nunique()),
        "total_products": int(tx["StockCode"].nunique()),
        "low_stock_alerts": low,
        "overstock_alerts": over,
        "date_from": str(tx["InvoiceDate"].min().date()),
        "date_to": str(tx["InvoiceDate"].max().date()),
        "data_source": data_source(),
    }


def monthly_sales() -> list[dict]:
    eng = _pg()
    if eng is not None:
        sql = text('''
            SELECT to_char(date_trunc('month', "InvoiceDate"), 'YYYY-MM') AS month,
                   SUM("TotalPrice") AS revenue
            FROM sales GROUP BY 1 ORDER BY 1
        ''')
        with eng.connect() as c:
            rows = c.execute(sql).mappings().all()
        return [{"month": r["month"], "revenue": round(float(r["revenue"]), 2)}
                for r in rows]
    tx = transactions()
    m = (tx.set_index("InvoiceDate")["TotalPrice"].resample("MS").sum().reset_index())
    m["InvoiceDate"] = m["InvoiceDate"].dt.strftime("%Y-%m")
    return m.rename(columns={"InvoiceDate": "month", "TotalPrice": "revenue"}) \
            .to_dict(orient="records")


def top_products(limit: int) -> list[dict]:
    eng = _pg()
    if eng is not None:
        sql = text('''
            SELECT "Description" AS product, SUM("TotalPrice") AS revenue
            FROM sales GROUP BY "Description"
            ORDER BY revenue DESC LIMIT :n
        ''')
        with eng.connect() as c:
            rows = c.execute(sql, {"n": limit}).mappings().all()
        return [{"product": r["product"], "revenue": round(float(r["revenue"]), 2)}
                for r in rows]
    tx = transactions()
    top = tx.groupby("Description")["TotalPrice"].sum().nlargest(limit).round(2).reset_index()
    return top.rename(columns={"Description": "product", "TotalPrice": "revenue"}) \
              .to_dict(orient="records")


def sales_by_country(limit: int) -> list[dict]:
    eng = _pg()
    if eng is not None:
        sql = text('''
            SELECT "Country" AS country, SUM("TotalPrice") AS revenue
            FROM sales GROUP BY "Country"
            ORDER BY revenue DESC LIMIT :n
        ''')
        with eng.connect() as c:
            rows = c.execute(sql, {"n": limit}).mappings().all()
        return [{"country": r["country"], "revenue": round(float(r["revenue"]), 2)}
                for r in rows]
    tx = transactions()
    c = tx.groupby("Country")["TotalPrice"].sum().nlargest(limit).round(2).reset_index()
    return c.rename(columns={"Country": "country", "TotalPrice": "revenue"}) \
            .to_dict(orient="records")

@lru_cache(maxsize=None)
def model(name: str):
    """Load a saved model artifact by stem name, cached."""
    return joblib.load(MODELS / f"{name}.joblib")


def refresh_cache() -> None:
    """Clear cached tables (call after reloading the database)."""
    _load.cache_clear()
    engine.cache_clear()