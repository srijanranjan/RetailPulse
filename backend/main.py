from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import store

app = FastAPI(title="RetailPulse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "RetailPulse API", "status": "ok", "docs": "/docs"}


@app.get("/dashboard")
def dashboard():
    tx = store.transactions()
    inv = store.inventory()
    return {
        "total_revenue": round(float(tx["TotalPrice"].sum()), 2),
        "total_orders": int(tx["Invoice"].nunique()),
        "total_customers": int(tx["CustomerID"].nunique()),
        "total_products": int(tx["StockCode"].nunique()),
        "low_stock_alerts": int((inv["alert"] == "LOW_STOCK").sum()),
        "overstock_alerts": int((inv["alert"] == "OVERSTOCK").sum()),
        "date_from": str(tx["InvoiceDate"].min().date()),
        "date_to": str(tx["InvoiceDate"].max().date()),
        "data_source": store.data_source(),
    }


@app.get("/sales/monthly")
def sales_monthly():
    tx = store.transactions()
    m = (tx.set_index("InvoiceDate")["TotalPrice"].resample("MS").sum()
         .reset_index())
    m["InvoiceDate"] = m["InvoiceDate"].dt.strftime("%Y-%m")
    return m.rename(columns={"InvoiceDate": "month", "TotalPrice": "revenue"}) \
            .to_dict(orient="records")


@app.get("/sales/top_products")
def top_products(limit: int = Query(10, ge=1, le=100)):
    tx = store.transactions()
    top = (tx.groupby("Description")["TotalPrice"].sum()
           .nlargest(limit).round(2).reset_index())
    return top.rename(columns={"Description": "product", "TotalPrice": "revenue"}) \
              .to_dict(orient="records")


@app.get("/sales/by_country")
def sales_by_country(limit: int = Query(10, ge=1, le=50)):
    tx = store.transactions()
    c = (tx.groupby("Country")["TotalPrice"].sum()
         .nlargest(limit).round(2).reset_index())
    return c.rename(columns={"Country": "country", "TotalPrice": "revenue"}) \
            .to_dict(orient="records")


@app.get("/forecast")
def get_forecast(horizon: int = Query(30, ge=1, le=90)):
    fc = store.forecast().copy()
    fc["ds"] = pd.to_datetime(fc["ds"]).dt.strftime("%Y-%m-%d")
    future = fc.tail(horizon)
    return {
        "history_tail": fc.iloc[-(horizon + 60):-horizon][["ds", "yhat"]]
                          .to_dict(orient="records"),
        "forecast": future[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                          .round(2).to_dict(orient="records"),
        "forecast_total": round(float(future["yhat"].sum()), 2),
    }


@app.get("/segments")
def get_segments():
    seg = store.segments()
    summary = (seg.groupby("segment")
               .agg(customers=("CustomerID", "size"),
                    avg_recency=("Recency", "mean"),
                    avg_frequency=("Frequency", "mean"),
                    avg_monetary=("Monetary", "mean"),
                    total_revenue=("Monetary", "sum"))
               .round(2).reset_index())
    return summary.to_dict(orient="records")


@app.get("/customers")
def get_customers(segment: str | None = None,
                  limit: int = Query(100, ge=1, le=1000)):
    seg = store.segments()
    if segment:
        seg = seg[seg["segment"] == segment]
        if seg.empty:
            raise HTTPException(404, f"No customers in segment '{segment}'")
    cols = ["CustomerID", "Recency", "Frequency", "Monetary",
            "segment", "Country"]
    out = seg[cols].nlargest(limit, "Monetary").round(2)
    return out.to_dict(orient="records")


@app.get("/inventory")
def get_inventory(alert: str | None = Query(None,
                  description="Filter: LOW_STOCK, OVERSTOCK, OK")):
    inv = store.inventory().copy()
    if alert:
        inv = inv[inv["alert"] == alert.upper()]
    inv["stockout_date"] = inv["stockout_date"].astype(str)
    return inv.to_dict(orient="records")

class ChurnRequest(BaseModel):
    recency: float = Field(..., examples=[30])
    frequency: float = Field(..., examples=[5])
    monetary: float = Field(..., description="log1p(total spend)", examples=[8.5])
    avg_order_value: float = Field(..., examples=[350])
    n_products: float = Field(..., examples=[40])
    total_qty: float = Field(..., examples=[500])
    tenure: float = Field(..., examples=[200])


def _risk(p: float) -> str:
    return "High" if p >= 0.66 else "Medium" if p >= 0.33 else "Low"


@lru_cache(maxsize=1)
def _churn_explainer():
    """Rebuild the SHAP explainer from the model once (never unpickle it)."""
    return shap.TreeExplainer(store.model("churn")["model"])


@app.post("/predict_churn")
def predict_churn(req: ChurnRequest):
    bundle = store.model("churn")
    mdl, cols = bundle["model"], bundle["features"]

    X = pd.DataFrame([[getattr(req, c) for c in cols]], columns=cols)
    proba = float(mdl.predict_proba(X)[:, 1][0])

    dm = xgb.DMatrix(X, feature_names=cols)
    contribs = mdl.get_booster().predict(dm, pred_contribs=True)[0, :-1]
    contributions = (pd.Series(contribs, index=cols)
                     .sort_values(key=abs, ascending=False))
    top = [{"feature": f, "shap_value": round(float(v), 4)}
           for f, v in contributions.head(5).items()]

    return {
        "churn_probability": round(proba, 4),
        "risk_level": _risk(proba),
        "top_factors": top,
    }


@app.get("/predict_churn/{customer_id}")
def churn_lookup(customer_id: int):
    """Return the precomputed churn score for an existing customer."""
    ch = store.churn()
    row = ch[ch["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(404, f"Customer {customer_id} not found")
    r = row.iloc[0]
    return {
        "customer_id": customer_id,
        "churn_probability": float(r["churn_probability"]),
        "risk_level": str(r["risk_level"]),
    }