from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def _series(metric: str = "revenue") -> pd.DataFrame:
    daily = pd.read_parquet(PROCESSED / "daily_sales.parquet")
    df = daily.rename(columns={"Date": "ds", metric: "y"})[["ds", "y"]]
    return df


def _metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true > 0                       # avoid div-by-zero on closed days
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": round(mae, 1), "RMSE": round(rmse, 1), "MAPE_%": round(mape, 1)}


def _make_model() -> Prophet:
    return Prophet(
        weekly_seasonality=True,     # captures the Saturday closure
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.90,
    )


def evaluate(metric: str = "revenue", horizon: int = 30) -> dict:
    df = _series(metric)
    train, test = df.iloc[:-horizon], df.iloc[-horizon:]
    m = _make_model().fit(train)
    fc = m.predict(test[["ds"]])
    scores = _metrics(test["y"].values, fc["yhat"].values)
    print(f"[{metric}] holdout={horizon}d  ->  {scores}")
    return scores


def run(metric: str = "revenue", horizon: int = 30):
    MODELS.mkdir(exist_ok=True)
    scores = evaluate(metric, horizon)

    # Refit on the full series and forecast forward
    df = _series(metric)
    m = _make_model().fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="D")
    fc = m.predict(future)

    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out["yhat"] = out["yhat"].clip(lower=0)          # revenue can't be negative
    out["yhat_lower"] = out["yhat_lower"].clip(lower=0)
    out["metric"] = metric
    out.to_parquet(PROCESSED / "forecast_results.parquet", index=False)
    joblib.dump(m, MODELS / f"forecast_{metric}.joblib")

    fut = out.tail(horizon)
    print(f"Next {horizon}d forecast total {metric}: "
          f"{fut['yhat'].sum():,.0f}  "
          f"(90% CI {fut['yhat_lower'].sum():,.0f} – {fut['yhat_upper'].sum():,.0f})")
    return out, scores


if __name__ == "__main__":
    run()