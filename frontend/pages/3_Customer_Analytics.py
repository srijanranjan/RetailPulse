import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from api import get, post

st.set_page_config(page_title="Customer Analytics", page_icon="👥", layout="wide")
st.title("👥 Customer Analytics")

tab1, tab2, tab3 = st.tabs(["Segments", "Customer explorer", "Churn predictor"])


with tab1:
    seg = pd.DataFrame(get("/segments"))
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(seg, names="segment", values="customers",
                     title="Customers per segment", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(seg.sort_values("total_revenue"), x="total_revenue",
                     y="segment", orientation="h",
                     title="Revenue by segment (£)", color="segment")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg, use_container_width=True)


with tab2:
    segments = ["VIP", "Loyal", "Regular", "New", "At-Risk"]
    choice = st.selectbox("Segment", segments)
    custs = pd.DataFrame(get("/customers", segment=choice, limit=200))
    fig = px.scatter(custs, x="Recency", y="Monetary", size="Frequency",
                     color="Frequency", hover_data=["CustomerID", "Country"],
                     title=f"{choice} customers — Recency vs Monetary",
                     color_continuous_scale="Viridis")
    fig.update_layout(yaxis_title="Monetary (£)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(custs, use_container_width=True)


with tab3:
    st.markdown("Enter a customer profile to get a live churn prediction "
                "with the SHAP factors driving it.")
    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Recency (days since last order)", 0, 800, 120)
    frequency = c2.number_input("Frequency (orders)", 1, 400, 3)
    total_spend = c3.number_input("Total spend (£)", 1.0, 1e6, 800.0)
    aov = c1.number_input("Avg order value (£)", 1.0, 1e5, 250.0)
    n_products = c2.number_input("Distinct products", 1, 5000, 20)
    total_qty = c3.number_input("Total quantity", 1, 100000, 200)
    tenure = c1.number_input("Tenure (days)", 0, 800, 150)

    if st.button("Predict churn", type="primary"):
        payload = {
            "recency": recency, "frequency": frequency,
            "monetary": float(np.log1p(total_spend)),   # model expects log1p
            "avg_order_value": aov, "n_products": n_products,
            "total_qty": total_qty, "tenure": tenure,
        }
        res = post("/predict_churn", payload)
        prob = res["churn_probability"]
        colour = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[res["risk_level"]]
        st.metric("Churn probability", f"{prob:.1%}",
                  f"{colour} {res['risk_level']} risk")

        factors = pd.DataFrame(res["top_factors"])
        factors["direction"] = np.where(factors["shap_value"] > 0,
                                        "↑ increases churn", "↓ reduces churn")
        fig = px.bar(factors[::-1], x="shap_value", y="feature",
                     orientation="h", color="direction",
                     title="Top factors (SHAP)",
                     color_discrete_map={"↑ increases churn": "#dc2626",
                                         "↓ reduces churn": "#16a34a"})
        fig.update_layout(yaxis_title="", xaxis_title="SHAP value")
        st.plotly_chart(fig, use_container_width=True)