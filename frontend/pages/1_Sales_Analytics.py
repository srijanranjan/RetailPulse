import pandas as pd
import plotly.express as px
import streamlit as st

from api import get

st.set_page_config(page_title="Sales Analytics", page_icon="📈", layout="wide")
st.title("📈 Sales Analytics")

monthly = pd.DataFrame(get("/sales/monthly"))
fig = px.line(monthly, x="month", y="revenue", markers=True,
              title="Monthly Revenue (£)")
fig.update_layout(yaxis_title="Revenue (£)", xaxis_title="")
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)


with col1:
    n = st.slider("Top products", 5, 25, 10)
    top = pd.DataFrame(get("/sales/top_products", limit=n))
    fig = px.bar(top[::-1], x="revenue", y="product", orientation="h",
                 title=f"Top {n} Products by Revenue")
    fig.update_layout(yaxis_title="", xaxis_title="Revenue (£)", height=500)
    st.plotly_chart(fig, use_container_width=True)


with col2:
    m = st.slider("Top countries", 5, 20, 10)
    countries = pd.DataFrame(get("/sales/by_country", limit=m))
    fig = px.bar(countries[::-1], x="revenue", y="country", orientation="h",
                 title=f"Top {m} Countries by Revenue", color="revenue",
                 color_continuous_scale="Blues")
    fig.update_layout(yaxis_title="", xaxis_title="Revenue (£)", height=500,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)