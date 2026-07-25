
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

FEATURES = ["Recency", "Frequency", "Monetary"]
SEGMENT_NAMES = ["At-Risk", "New", "Regular", "Loyal", "VIP"]  # worst → best


def _prep(rfm: pd.DataFrame) -> np.ndarray:
    """Log-transform skewed features then standardise."""
    X = rfm[FEATURES].copy()
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])
    # Recency: smaller is better; keep as-is (scaler handles range)
    return X.values


def train(rfm: pd.DataFrame, k: int = 5, random_state: int = 42):
    X = _prep(rfm)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(Xs)
    rfm = rfm.copy()
    rfm["cluster"] = km.labels_

    # --- map clusters to business labels by an RFM "value score" ---
    # Higher score = better customer. Recency inverted (recent = good).
    prof = rfm.groupby("cluster")[FEATURES].mean()
    score = (-prof["Recency"].rank()          # recent → high
             + prof["Frequency"].rank()       # frequent → high
             + prof["Monetary"].rank())       # big spender → high
    ordered = score.sort_values().index.tolist()   # worst → best
    names = SEGMENT_NAMES[:k] if k == 5 else [f"Segment {i}" for i in range(k)]
    cluster_to_name = {c: n for c, n in zip(ordered, names)}
    rfm["segment"] = rfm["cluster"].map(cluster_to_name)

    return rfm, km, scaler, cluster_to_name


def run(k: int = 5):
    MODELS.mkdir(exist_ok=True)
    rfm = pd.read_parquet(PROCESSED / "rfm.parquet")
    seg, km, scaler, mapping = train(rfm, k=k)

    joblib.dump({"model": km, "scaler": scaler, "mapping": mapping,
                 "features": FEATURES}, MODELS / "segmentation.joblib")
    seg.to_parquet(PROCESSED / "customer_segments.parquet", index=False)

    print(f"Inertia: {km.inertia_:,.0f}")
    summary = (seg.groupby("segment")
               .agg(customers=("CustomerID", "size"),
                    recency=("Recency", "mean"),
                    frequency=("Frequency", "mean"),
                    monetary=("Monetary", "mean"),
                    total_revenue=("Monetary", "sum"))
               .round(1)
               .reindex([n for n in SEGMENT_NAMES if n in seg.segment.unique()]))
    summary["rev_share_%"] = (summary["total_revenue"] /
                              summary["total_revenue"].sum() * 100).round(1)
    print(summary.to_string())
    return seg, summary


if __name__ == "__main__":
    run()