import pandas as pd
import plotly.express as px
import streamlit as st

from api import get

st.set_page_config(page_title="Inventory", page_icon="📦", layout="wide")
st.title("📦 Inventory Recommendations")
st.caption("Reorder points from historical demand. `current_stock` is simulated "
           "in the demo — swap in your real stock feed for production use.")

inv = pd.DataFrame(get("/inventory"))

counts = inv["alert"].value_counts()
c1, c2, c3 = st.columns(3)
c1.metric("🔴 Low stock", int(counts.get("LOW_STOCK", 0)))
c2.metric("🟠 Overstock", int(counts.get("OVERSTOCK", 0)))
c3.metric("🟢 OK", int(counts.get("OK", 0)))

flt = st.radio("Filter", ["All", "LOW_STOCK", "OVERSTOCK", "OK"], horizontal=True)
view = inv if flt == "All" else inv[inv["alert"] == flt]

urgent = view[view["alert"] == "LOW_STOCK"].nlargest(15, "reorder_qty")
if not urgent.empty:
    fig = px.bar(urgent[::-1], x="reorder_qty", y="Description", orientation="h",
                 title="Largest reorder quantities (low-stock items)",
                 color="reorder_qty", color_continuous_scale="Reds")
    fig.update_layout(yaxis_title="", xaxis_title="Reorder qty", height=500,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    view.sort_values("reorder_qty", ascending=False),
    use_container_width=True,
    column_config={
        "current_stock": st.column_config.NumberColumn("Current stock"),
        "reorder_point": st.column_config.NumberColumn("Reorder point"),
        "reorder_qty": st.column_config.NumberColumn("Reorder qty"),
        "stockout_date": st.column_config.TextColumn("Est. stockout"),
    },
)