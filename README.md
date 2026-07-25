# 🛒 RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

An end-to-end retail analytics platform built on the **UCI Online Retail II**
dataset. It cleans ~1M transactions, forecasts demand, segments customers,
predicts churn, and recommends inventory actions — served by a **FastAPI**
backend and an interactive **Streamlit** dashboard.

<!-- Replace these with your live URLs once deployed -->
**Live dashboard:** [https://retailpulse-zi-d-io.streamlit.app](https://retailpulse-zi-d-io.streamlit.app/)
**Live API docs:** [https://retailpulse-api.onrender.com/docs](https://retailpulse-api-l3q9.onrender.com/docs)

---

## 📸 Dashboard

### Home — KPI overview
![Dashboard home](docs/screenshots/dashboard.png)

### Sales Analytics — revenue trends, top products, markets
![Sales Analytics](docs/screenshots/sales_analytics.png)

### Demand Forecast — 30-day Prophet forecast with 90% confidence bands
![Demand Forecast](docs/screenshots/demand_forecast.png)

### Customer Analytics — RFM segments & live churn prediction
![Customer Analytics](docs/screenshots/customer_analytics.png)

### Inventory — reorder recommendations & stock alerts
![Inventory](docs/screenshots/inventory.png)

---

## ✨ Features

| Module | What it does | Model / Method |
|--------|--------------|----------------|
| **Sales Analytics** | Revenue trends, top products, market breakdown | Pandas aggregation |
| **Demand Forecasting** | 30-day revenue forecast with confidence intervals | Prophet |
| **Customer Segmentation** | VIP / Loyal / Regular / New / At-Risk segments | KMeans on RFM |
| **Churn Prediction** | Per-customer churn probability + explanations | XGBoost + TreeSHAP |
| **Inventory Recommendation** | Reorder qty, stockout dates, stock alerts | Rule-based reorder-point |

---

## 📊 Dataset

UK-based online gift retailer, **2009-12-01 → 2011-12-09** (~1.07M line items).
Columns: `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`.

After cleaning (dedup, remove cancellations/service codes/invalid rows, drop
rows without a Customer ID): **776,579 valid sales lines (72.8% retained)**,
**5,852 customers**, **£17.07M** total revenue.

> Source: UCI Machine Learning Repository — Online Retail II.
> the workbook is at `data/raw/online_retail_II.xlsx`.

---

## 📈 Results

- **Segmentation** — VIP customers are just **890 people (15%) but drive ~70% of revenue**; At-Risk (1,077) haven't purchased in ~520 days on average.
- **Forecast** — Prophet reaches **~25.8% MAPE** on a 30-day holdout; it captures the retailer's near-zero-sales Saturdays via weekly seasonality.
- **Churn** — XGBoost scores **ROC-AUC 0.787** on a leakage-free temporal split; recency is the dominant driver (TreeSHAP).
- **Inventory** — flags low-stock / overstock across the top movers with reorder quantities and estimated stockout dates.

---

## 🏗️ Architecture

```text
                         GitHub
                            │
              ┌─────────────┴──────────────┐
              │                            │
     Render (FastAPI API)         Streamlit Cloud (Dashboard)
              │                            │
   data/processed/*.parquet   ── HTTP ──►  reads JSON from API
        or Render Postgres
              │
        models/*.joblib
```

Data flow:
```text
Excel  →  preprocessing  →  processed tables (Parquet)  →  models
                                     │                        │
                                     └──────────┬─────────────┘
                                                ▼
                                       FastAPI (store.py)
                                                ▼
                                      Streamlit dashboard
```

---

## 🧰 Tech Stack

**Python 3.11** · pandas · NumPy · scikit-learn · Prophet · XGBoost · TreeSHAP ·
FastAPI · Uvicorn · SQLAlchemy · PostgreSQL · Streamlit · Plotly · Render ·
Streamlit Cloud.

---

## 📁 Project Structure

```text
RetailPulse/
├── data/
│   ├── raw/                 online_retail_II.xlsx  (not committed)
│   └── processed/           cleaned tables + model outputs (Parquet)
├── models/                  trained models (.joblib)
├── notebooks/
│   └── 01_EDA.ipynb
├── src/
│   ├── preprocessing.py     clean + feature engineering (RFM, daily sales)
│   ├── segmentation.py      KMeans customer segments
│   ├── forecasting.py       Prophet demand forecast
│   ├── churn.py             XGBoost churn + TreeSHAP
│   └── inventory.py         reorder-point engine
├── backend/
│   ├── main.py              FastAPI app + endpoints
│   ├── store.py             data access (Postgres or Parquet)
│   └── database.py          load processed data into Postgres
├── frontend/
│   ├── Home.py              dashboard home (KPIs)
│   ├── api.py               API client
│   └── pages/               Sales · Forecast · Customers · Inventory
├── requirements.txt         dashboard deps (Streamlit Cloud)
├── requirements-api.txt     backend deps (Render)
├── requirements-dev.txt     full local env (run + train)
├── render.yaml              Render Blueprint
└── runtime.txt              Python version pin
```

---

## 🚀 Local Setup

**1. Install (full local environment):**
```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

**2. Build data + train models** (run once, in order):
```bash
python -m src.preprocessing         # → transactions, returns, daily_sales, rfm
python -m src.segmentation          # → customer_segments + model
python -m src.forecasting           # → forecast_results + model
python -m src.churn                 # → churn_predictions + model
python -m src.inventory             # → inventory
```

**3. Run the app** (two terminals):
```bash
# terminal 1 — backend
uvicorn backend.main:app --reload --port 8000     # docs: http://localhost:8000/docs

# terminal 2 — dashboard
streamlit run frontend/Home.py                    # http://localhost:8501
```

The API serves from `data/processed/*.parquet` by default. To use PostgreSQL:
```bash
export DATABASE_URL="postgresql://user:pass@host:5432/db"
python -m backend.database --check
python -m backend.database --load
```
When `DATABASE_URL` is set, `/dashboard` reports `"data_source": "postgresql"`.

---

## ☁️ Deployment

### API on Render
1. Push the repo to GitHub.
2. Render → **New +** → **Blueprint** → select the repo (reads `render.yaml`).
3. Apply. Render installs `requirements-api.txt` and runs
   `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
4. Verify at `https://<service>.onrender.com/docs`.

### Dashboard on Streamlit Cloud
1. share.streamlit.io → **New app** → repo, main file `frontend/Home.py`.
2. **Advanced settings → Secrets:**
```toml
   API_URL = "https://<service>.onrender.com"
```
3. Deploy (installs the root `requirements.txt`).

> Render's free tier sleeps after ~15 min idle; the first request wakes it in
> 30–60s, so the dashboard may need one refresh after inactivity.

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/dashboard` | KPI summary (revenue, orders, customers, alerts) |
| GET | `/sales/monthly` | Monthly revenue series |
| GET | `/sales/top_products?limit=` | Top products by revenue |
| GET | `/sales/by_country?limit=` | Revenue by country |
| GET | `/forecast?horizon=` | Forecast + confidence interval |
| GET | `/segments` | Segment summary statistics |
| GET | `/customers?segment=&limit=` | Customers in a segment |
| GET | `/inventory?alert=` | Inventory recommendations (filterable) |
| POST | `/predict_churn` | Live churn prediction + SHAP factors |
| GET | `/predict_churn/{customer_id}` | Precomputed churn score for a customer |

Interactive docs at `/docs`.

---

## 📝 Notes & Limitations

- **Inventory stock levels are simulated** — the dataset has no stock column, so
  `current_stock` is generated for the demo. The reorder logic is real; swap in a
  live stock feed by replacing `_simulate_stock` in `src/inventory.py`.
- **Churn labeling** uses a leakage-free temporal split (features up to a cutoff;
  churn = no purchase in the following 90 days), so recency isn't circular.
- **Saturday effect** — the retailer processes almost no weekend orders; this real
  pattern is modeled by Prophet's weekly seasonality.
- **SHAP** values are computed with XGBoost's native TreeSHAP (`pred_contribs`) to
  avoid version fragility in the standalone `shap` library.

---

## 📄 License

Released under the MIT License. Dataset © its original authors (UCI ML Repository).
