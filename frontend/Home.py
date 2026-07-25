
import streamlit as st

from api import get, health, API_URL

st.set_page_config(page_title="RetailPulse", page_icon="🛒", layout="wide")

st.title("🛒 RetailPulse")
st.caption("AI-powered customer analytics & demand forecasting")

if not health():
    st.error(f"Cannot reach the API at {API_URL}. "
             "Start it with `uvicorn backend.main:app --port 8000` "
             "or set the API_URL environment variable.")
    st.stop()

kpi = get("/dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"£{kpi['total_revenue']:,.0f}")
c2.metric("Orders", f"{kpi['total_orders']:,}")
c3.metric("Customers", f"{kpi['total_customers']:,}")
c4.metric("Products", f"{kpi['total_products']:,}")

c5, c6, c7 = st.columns(3)
c5.metric("⚠️ Low-stock items", kpi["low_stock_alerts"])
c6.metric("📦 Overstock items", kpi["overstock_alerts"])
c7.metric("Data range", f"{kpi['date_from']} → {kpi['date_to']}")

st.divider()
st.subheader("Explore")
st.markdown(
    "- **📈 Sales Analytics** — revenue trends, top products, markets\n"
    "- **🔮 Demand Forecast** — 30-day Prophet forecast with confidence bands\n"
    "- **👥 Customer Analytics** — RFM segments & live churn prediction\n"
    "- **📦 Inventory** — reorder recommendations & stock alerts\n\n"
    "Use the sidebar to navigate."
)