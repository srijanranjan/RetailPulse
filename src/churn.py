from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb          # <- was: import shap
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def build_features(tx: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int = 90):
    """Return (X, y, customer_ids) using a leakage-free temporal split."""
    obs = tx[tx["InvoiceDate"] <= cutoff]
    fut = tx[(tx["InvoiceDate"] > cutoff) &
             (tx["InvoiceDate"] <= cutoff + pd.Timedelta(days=horizon_days))]

    feats = (obs.groupby("CustomerID")
             .agg(recency=("InvoiceDate", lambda s: (cutoff - s.max()).days),
                  frequency=("Invoice", "nunique"),
                  monetary=("TotalPrice", "sum"),
                  avg_order_value=("TotalPrice", "mean"),
                  n_products=("StockCode", "nunique"),
                  total_qty=("Quantity", "sum"),
                  tenure=("InvoiceDate", lambda s: (s.max() - s.min()).days))
             .reset_index())
    feats["monetary"] = np.log1p(feats["monetary"])   # tame the skew

    active_future = set(fut["CustomerID"].unique())
    feats["churn"] = (~feats["CustomerID"].isin(active_future)).astype(int)

    cols = ["recency", "frequency", "monetary", "avg_order_value",
            "n_products", "total_qty", "tenure"]
    return feats[cols], feats["churn"], feats["CustomerID"], cols


def _risk(p: float) -> str:
    return "High" if p >= 0.66 else "Medium" if p >= 0.33 else "Low"


def run(horizon_days: int = 90):
    MODELS.mkdir(exist_ok=True)
    tx = pd.read_parquet(PROCESSED / "transactions.parquet")
    cutoff = tx["InvoiceDate"].max() - pd.Timedelta(days=horizon_days)

    X, y, cust_ids, cols = build_features(tx, cutoff, horizon_days)
    print(f"Cutoff {cutoff.date()} | customers={len(X):,} | churn rate={y.mean():.1%}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    )
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    print(f"Test ROC-AUC: {auc:.3f}")
    print(classification_report(yte, (proba >= 0.5).astype(int),
                                target_names=["retained", "churned"], digits=3))

    # Global feature importance via XGBoost's native TreeSHAP (pred_contribs).
    # This avoids the shap<->xgboost version fragility; values are identical.
    dm = xgb.DMatrix(Xte, feature_names=cols)
    contribs = model.get_booster().predict(dm, pred_contribs=True)
    importance = (pd.Series(np.abs(contribs[:, :-1]).mean(0), index=cols)
                  .sort_values(ascending=False))
    print("Top churn drivers (mean |SHAP|):")
    print(importance.round(3).to_string())

    # Score every customer & persist
    all_proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({"customer_id": cust_ids,
                        "churn_probability": all_proba.round(4)})
    out["risk_level"] = out["churn_probability"].apply(_risk)
    out.to_parquet(PROCESSED / "churn_predictions.parquet", index=False)

    joblib.dump({"model": model, "features": cols}, MODELS / "churn.joblib")
    print("Risk distribution:", out["risk_level"].value_counts().to_dict())
    return out, auc


if __name__ == "__main__":
    run()