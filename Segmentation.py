import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.decomposition import PCA


kmeans = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("Customer Segmentation Dashboard")
st.markdown("Predict customer segment and analyze behavioral positioning.")


st.sidebar.header("Enter Customer Details")

age = st.sidebar.number_input('Age', 18, 100, 40)
income = st.sidebar.number_input('Income', 0, 200000, 50000)
total_spending = st.sidebar.number_input('Total Spending', 0, 5000, 1000)
num_web_purchases = st.sidebar.number_input('Web Purchases', 0, 100, 10)
num_catalog_purchases = st.sidebar.number_input('Catalog Purchases', 0, 100, 5)
num_store_purchases = st.sidebar.number_input('Store Purchases', 0, 100, 8)
num_web_visit = st.sidebar.number_input('Website Visits / Month', 0, 50, 5)
recency = st.sidebar.number_input('Recency (days)', 0, 365, 30)


feature_order = [
    'Age','Income','Total_spending',
    'NumWebPurchases','NumCatalogPurchases',
    'NumStorePurchases','NumWebVisitsMonth','Recency'
]

input_data = pd.DataFrame([[
    age, income, total_spending,
    num_web_purchases, num_catalog_purchases,
    num_store_purchases, num_web_visit, recency
]], columns=feature_order)

input_scaled = scaler.transform(input_data)

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("Predict Segment"):

    cluster = kmeans.predict(input_scaled)[0]
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_profile = centroids[cluster]

    st.success(f"Predicted Segment: Cluster {cluster}")

    # ----------------------------
    # Segment Labels + Strategy
    # ----------------------------
    segment_info = {
        0: ("Moderate Seniors",
            "Older moderate spenders. Focus on catalog promotions and bundled offers."),
        1: ("High Income Active Buyers",
            "Affluent and highly engaged. Ideal for loyalty rewards and upselling."),
        2: ("Low Spend – Recently Active",
            "Low-value but engaged customers. Push cross-sell campaigns."),
        3: ("Elite High-Value Customers",
            "Top premium customers. Prioritize VIP programs and exclusivity."),
        4: ("Affluent but Less Recent",
            "High spenders at risk of churn. Send reactivation campaigns."),
        5: ("Low Value – Inactive",
            "Low engagement and low spenders. Use discounts and retargeting.")
    }

    st.subheader(segment_info[cluster][0])
    st.write(segment_info[cluster][1])

    # ----------------------------
    # Metrics Comparison
    # ----------------------------
    st.subheader("Customer vs Cluster Average")

    col1, col2, col3 = st.columns(3)

    col1.metric("Income",
                f"{income}",
                f"{round(income - cluster_profile[1],2)} vs avg")

    col2.metric("Total Spending",
                f"{total_spending}",
                f"{round(total_spending - cluster_profile[2],2)} vs avg")

    col3.metric("Recency",
                f"{recency}",
                f"{round(recency - cluster_profile[7],2)} vs avg")

    # ----------------------------
    # Radar Chart
    # ----------------------------
    st.subheader("Behavioral Profile Radar Chart")

    categories = feature_order
    user_values = input_data.values.flatten()
    cluster_values = cluster_profile

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Customer'
    ))

    fig.add_trace(go.Scatterpolar(
        r=cluster_values,
        theta=categories,
        fill='toself',
        name='Cluster Avg'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # PCA Visualization
    # ----------------------------
    st.subheader("Customer Positioning in Cluster Space (PCA)")

    X_all = scaler.transform(pd.DataFrame(kmeans.cluster_centers_, columns=feature_order))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)

    user_pca = pca.transform(input_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = range(len(pca_df))

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=pca_df["PC1"],
        y=pca_df["PC2"],
        mode='markers+text',
        text=pca_df["Cluster"],
        name="Cluster Centers"
    ))

    fig2.add_trace(go.Scatter(
        x=user_pca[:,0],
        y=user_pca[:,1],
        mode='markers',
        marker=dict(size=12),
        name="Customer"
    ))

    st.plotly_chart(fig2, use_container_width=True)