from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

Z_SCORE = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}   # service level → z


def demand_stats(tx: pd.DataFrame, top_n: int | None = 200) -> pd.DataFrame:
    """Average & std of DAILY demand per product over the active period."""
    daily = (tx.groupby(["StockCode", tx["InvoiceDate"].dt.date])["Quantity"]
             .sum().reset_index(name="qty"))
    stats = (daily.groupby("StockCode")["qty"]
             .agg(avg_daily_demand="mean", demand_std_daily="std",
                  active_days="size").reset_index())
    stats["demand_std_daily"] = stats["demand_std_daily"].fillna(0)
    desc = (tx.groupby("StockCode")["Description"]
            .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
            .reset_index())
    stats = stats.merge(desc, on="StockCode")
    if top_n:                                  # focus on the movers
        stats = stats.nlargest(top_n, "avg_daily_demand").reset_index(drop=True)
    return stats


def _simulate_stock(stats: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Placeholder current stock ~ a few weeks of demand, with noise.
    Replace with your real inventory levels in production."""
    rng = np.random.default_rng(seed)
    base = stats["avg_daily_demand"] * rng.uniform(3, 40, len(stats))
    return np.round(base).astype(int)


def recommend(tx: pd.DataFrame, lead_time: int = 7, review_period: int = 7,
              service_level: float = 0.95, top_n: int | None = 200,
              current_stock: pd.Series | None = None) -> pd.DataFrame:
    z = Z_SCORE.get(service_level, 1.65)
    s = demand_stats(tx, top_n=top_n)

    s["current_stock"] = (current_stock if current_stock is not None
                          else _simulate_stock(s))

    s["safety_stock"] = np.ceil(z * s["demand_std_daily"] * np.sqrt(lead_time))
    s["reorder_point"] = np.ceil(s["avg_daily_demand"] * lead_time + s["safety_stock"])
    review_demand = s["avg_daily_demand"] * review_period
    s["reorder_qty"] = np.maximum(
        0, np.ceil(s["reorder_point"] + review_demand - s["current_stock"])).astype(int)

    days_left = np.where(s["avg_daily_demand"] > 0,
                         s["current_stock"] / s["avg_daily_demand"], np.inf)
    today = pd.Timestamp(tx["InvoiceDate"].max().date())
    s["stockout_date"] = [
        (today + pd.Timedelta(days=float(d))).date() if np.isfinite(d) else None
        for d in days_left]

    def _alert(row):
        if row["current_stock"] <= row["reorder_point"]:
            return "LOW_STOCK"
        if row["current_stock"] > row["reorder_point"] + 3 * review_demand[row.name]:
            return "OVERSTOCK"
        return "OK"
    s["alert"] = s.apply(_alert, axis=1)

    cols = ["StockCode", "Description", "current_stock", "avg_daily_demand",
            "safety_stock", "reorder_point", "reorder_qty",
            "stockout_date", "alert"]
    return s[cols].round(2)


def run(**kw):
    tx = pd.read_parquet(PROCESSED / "transactions.parquet")
    rec = recommend(tx, **kw)
    rec.to_parquet(PROCESSED / "inventory.parquet", index=False)
    print("Alert counts:", rec["alert"].value_counts().to_dict())
    print("\nSample low-stock items needing reorder:")
    print(rec[rec.alert == "LOW_STOCK"].nlargest(8, "reorder_qty")
          [["Description", "current_stock", "reorder_point",
            "reorder_qty", "stockout_date"]].to_string(index=False))
    return rec


if __name__ == "__main__":
    run()