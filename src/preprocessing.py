"""
RetailPulse — Data Preprocessing
================================
Loads the raw Online Retail II data, cleans it, and produces the analytical
tables used across the platform (sales, RFM, daily-demand series).

Design goals
------------
* Deterministic, logged cleaning (every drop is counted).
* Two logical outputs from one clean base:
    - `transactions`  : line-item level, valid sales only, customer attributed.
    - `returns`       : cancellation / return lines (Invoice starts with 'C').
* Derived tables:
    - `daily_sales`   : revenue & quantity per day (for Prophet forecasting).
    - `rfm`           : Recency / Frequency / Monetary per customer.

Run directly:  python -m src.preprocessing
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_XLSX_DEFAULT = RAW_DIR / "online_retail_II.xlsx"   # put the workbook here
RAW_PARQUET = RAW_DIR / "online_retail_II.parquet"     # cache, created on first run
PROCESSED = ROOT / "data" / "processed"

# Stock codes that are administrative / service items, not sellable products.
SERVICE_CODES = {
    "POST", "DOT", "C2", "M", "D", "S", "CRUK", "PADS",
    "BANK CHARGES", "ADJUST", "ADJUST2", "AMAZONFEE",
}
SERVICE_PREFIXES = ("TEST", "GIFT", "gift_", "B", "BANK")


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_raw(xlsx_path: str = RAW_XLSX_DEFAULT, use_cache: bool = True) -> pd.DataFrame:
    """Load both sheets of the workbook (cached to Parquet for speed)."""
    if use_cache and RAW_PARQUET.exists():
        return pd.read_parquet(RAW_PARQUET)

    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    for c in ["Invoice", "StockCode", "Description", "Country"]:
        df[c] = df[c].astype("string")
    RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PARQUET, index=False)
    return df


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #
def _is_service(code: pd.Series) -> pd.Series:
    up = code.str.upper().str.strip()
    is_set = up.isin({c.upper() for c in SERVICE_CODES})
    is_pref = up.str.startswith(tuple(p.upper() for p in SERVICE_PREFIXES))
    return is_set | is_pref


def clean(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, returns). Logs every filtering step."""
    log = []
    n0 = len(df)

    df = df.copy()
    df["Invoice"] = df["Invoice"].str.strip()
    df["StockCode"] = df["StockCode"].str.strip()
    df["Description"] = df["Description"].str.strip()

    # Rename for convenience
    df = df.rename(columns={"Customer ID": "CustomerID"})

    # Drop exact duplicate lines
    before = len(df)
    df = df.drop_duplicates()
    log.append(("drop exact duplicates", before - len(df)))

    # Split returns / cancellations (Invoice starting with 'C')
    is_return = df["Invoice"].str.startswith("C", na=False)
    returns = df[is_return].copy()
    df = df[~is_return].copy()
    log.append(("split out returns/cancellations", int(is_return.sum())))

    # Remove service / admin stock codes from product sales
    before = len(df)
    df = df[~_is_service(df["StockCode"])]
    log.append(("remove service/admin stock codes", before - len(df)))

    # Valid quantity & price
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    log.append(("remove qty<=0 or price<=0", before - len(df)))

    # Missing customer id -> cannot attribute for RFM/churn
    before = len(df)
    df = df.dropna(subset=["CustomerID"])
    log.append(("drop missing CustomerID", before - len(df)))

    # Types & derived columns
    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Date"] = df["InvoiceDate"].dt.date

    if verbose:
        print(f"Rows in:  {n0:,}")
        for step, n in log:
            print(f"  - {step:38s}: -{n:,}")
        print(f"Rows out (transactions): {len(df):,}  "
              f"({len(df)/n0*100:.1f}% retained)")
        print(f"Returns table:           {len(returns):,}")

    return df.reset_index(drop=True), returns.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Derived tables
# --------------------------------------------------------------------------- #
def build_daily_sales(tx: pd.DataFrame) -> pd.DataFrame:
    """Daily revenue & quantity — the series Prophet will forecast."""
    daily = (
        tx.groupby("Date")
        .agg(revenue=("TotalPrice", "sum"),
             quantity=("Quantity", "sum"),
             orders=("Invoice", "nunique"))
        .reset_index()
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    # Reindex to a continuous daily calendar (fill gaps with 0)
    full = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    daily = (daily.set_index("Date").reindex(full)
             .rename_axis("Date").reset_index())
    daily[["revenue", "quantity", "orders"]] = (
        daily[["revenue", "quantity", "orders"]].fillna(0))
    return daily


def build_rfm(tx: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recency / Frequency / Monetary table, one row per customer."""
    if snapshot is None:
        # Day after the last transaction — standard RFM convention
        snapshot = tx["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        tx.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max"),
            Country=("Country", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )
    rfm["Tenure"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm["AvgOrderValue"] = (rfm["Monetary"] / rfm["Frequency"]).round(2)
    rfm.attrs["snapshot"] = str(snapshot)
    return rfm


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(xlsx_path: str = RAW_XLSX_DEFAULT) -> dict[str, pd.DataFrame]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = load_raw(xlsx_path)
    tx, returns = clean(raw)
    daily = build_daily_sales(tx)
    rfm = build_rfm(tx)

    tx.to_parquet(PROCESSED / "transactions.parquet", index=False)
    returns.to_parquet(PROCESSED / "returns.parquet", index=False)
    daily.to_parquet(PROCESSED / "daily_sales.parquet", index=False)
    rfm.to_parquet(PROCESSED / "rfm.parquet", index=False)
    # Small CSV samples for quick inspection / database seeding
    tx.head(5000).to_csv(PROCESSED / "transactions_sample.csv", index=False)
    rfm.to_csv(PROCESSED / "rfm.csv", index=False)

    print("\nSaved to data/processed/:")
    for name, d in {"transactions": tx, "returns": returns,
                    "daily_sales": daily, "rfm": rfm}.items():
        print(f"  {name:14s} {d.shape}")
    return {"transactions": tx, "returns": returns, "daily_sales": daily, "rfm": rfm}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default=RAW_XLSX_DEFAULT)
    run(p.parse_args().xlsx)