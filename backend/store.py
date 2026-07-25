from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine
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


@lru_cache(maxsize=None)
def model(name: str):
    """Load a saved model artifact by stem name, cached."""
    return joblib.load(MODELS / f"{name}.joblib")


def refresh_cache() -> None:
    """Clear cached tables (call after reloading the database)."""
    _load.cache_clear()
    engine.cache_clear()