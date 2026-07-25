import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api import get

st.set_page_config(page_title="Demand Forecast", page_icon="🔮", layout="wide")
st.title("🔮 Demand Forecast")
st.caption("Prophet forecast of daily revenue with a 90% confidence interval.")

horizon = st.slider("Forecast horizon (days)", 7, 90, 30, step=1)
data = get("/forecast", horizon=horizon)

hist = pd.DataFrame(data["history_tail"])
fc = pd.DataFrame(data["forecast"])
for d in (hist, fc):
    if not d.empty:
        d["ds"] = pd.to_datetime(d["ds"])

fig = go.Figure()

fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines",
                         line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines",
                         line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(37,99,235,0.15)", name="90% interval"))

fig.add_trace(go.Scatter(x=hist["ds"], y=hist["yhat"], mode="lines",
                         name="Recent (fitted)", line=dict(color="#94a3b8")))
fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines",
                         name="Forecast", line=dict(color="#2563eb", width=2)))
fig.update_layout(title=f"Next {horizon} days", yaxis_title="Revenue (£)",
                  xaxis_title="", height=500, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.metric(f"Forecasted revenue (next {horizon} days)",
          f"£{data['forecast_total']:,.0f}")

with st.expander("Forecast table"):
    st.dataframe(fc.assign(ds=fc["ds"].dt.date), use_container_width=True)