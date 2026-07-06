"""
Sales Forecasting & Demand Intelligence Dashboard
Run locally with:  streamlit run app.py
Deploy free at:    https://share.streamlit.io  (Streamlit Community Cloud)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")

# ----------------------------------------------------------------------------
# DATA LOADING (cached so the app doesn't reprocess on every interaction)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("train.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("train.csv", encoding="latin1")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date"]).sort_values("Order Date").reset_index(drop=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    return df


@st.cache_data
def get_monthly_series(_df, mask=None):
    d = _df if mask is None else _df[mask]
    s = d.set_index("Order Date").resample("MS")["Sales"].sum()
    s.index.freq = "MS"
    return s


@st.cache_data
def get_weekly_series(_df):
    s = _df.set_index("Order Date").resample("W")["Sales"].sum()
    return s


@st.cache_resource
def fit_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


df = load_data()

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

# ============================================================================
# PAGE 1 — SALES OVERVIEW
# ============================================================================
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Sales by Year")
        yearly = df.groupby("Year")["Sales"].sum().reset_index()
        fig = px.bar(yearly, x="Year", y="Sales", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Sales Trend")
        monthly = get_monthly_series(df).reset_index()
        monthly.columns = ["Month", "Sales"]
        fig = px.line(monthly, x="Month", y="Sales", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales by Region & Category")
    f1, f2 = st.columns(2)
    with f1:
        region_filter = st.multiselect(
            "Region", options=sorted(df["Region"].unique()), default=sorted(df["Region"].unique())
        )
    with f2:
        category_filter = st.multiselect(
            "Category", options=sorted(df["Category"].unique()), default=sorted(df["Category"].unique())
        )

    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)]
    grouped = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig = px.bar(grouped, x="Region", y="Sales", color="Category", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2 — FORECAST EXPLORER
# ============================================================================
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    col1, col2 = st.columns(2)
    with col1:
        dim = st.selectbox("Select dimension", ["Category", "Region"])
    with col2:
        value = st.selectbox(f"Select {dim}", sorted(df[dim].unique()))

    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    mask = df[dim] == value
    series = get_monthly_series(df, mask)

    train_series, test_series = series.iloc[:-3], series.iloc[-3:]

    with st.spinner("Fitting SARIMA model..."):
        model = fit_sarima(train_series)
        forecast_obj = model.get_forecast(steps=max(horizon, 3))
        forecast = forecast_obj.predicted_mean.iloc[:horizon]
        ci = forecast_obj.conf_int().iloc[:horizon]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Historical", mode="lines"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast",
                              mode="lines+markers", line=dict(color="red")))
    fig.add_trace(go.Scatter(
        x=list(ci.index) + list(ci.index[::-1]),
        y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
        fill="toself", fillcolor="rgba(255,0,0,0.1)", line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval",
    ))
    fig.update_layout(title=f"{horizon}-Month Forecast — {value} ({dim})")
    st.plotly_chart(fig, use_container_width=True)

    test_pred = model.get_forecast(steps=3).predicted_mean
    model_mae = mae(test_series, test_pred)
    model_rmse = rmse(test_series, test_pred)

    m1, m2 = st.columns(2)
    m1.metric("MAE (on last 3 known months)", f"${model_mae:,.0f}")
    m2.metric("RMSE (on last 3 known months)", f"${model_rmse:,.0f}")

# ============================================================================
# PAGE 3 — ANOMALY REPORT
# ============================================================================
elif page == "Anomaly Report":
    st.title("Anomaly Report")

    weekly = get_weekly_series(df).reset_index()
    weekly.columns = ["Week", "WeeklySales"]
    weekly["week_of_year"] = weekly["Week"].dt.isocalendar().week.astype(int)

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["anomaly"] = iso.fit_predict(weekly[["WeeklySales", "week_of_year"]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["Week"], y=weekly["WeeklySales"], name="Weekly Sales", mode="lines"))
    anomalies = weekly[weekly["anomaly"] == -1]
    fig.add_trace(go.Scatter(x=anomalies["Week"], y=anomalies["WeeklySales"], name="Anomaly",
                              mode="markers", marker=dict(color="red", size=10)))
    fig.update_layout(title="Weekly Sales with Detected Anomalies (Isolation Forest)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomaly Weeks")
    st.dataframe(
        anomalies[["Week", "WeeklySales"]].sort_values("WeeklySales", ascending=False),
        use_container_width=True,
    )

# ============================================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ============================================================================
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")

    subcat = df.groupby("Sub-Category").agg(
        total_sales=("Sales", "sum"), avg_order_value=("Sales", "mean")
    ).reset_index()

    subcat_year = df.groupby(["Sub-Category", "Year"])["Sales"].sum().reset_index()

    def growth_rate(g):
        g = g.sort_values("Year")
        if len(g) < 2 or g["Sales"].iloc[0] == 0:
            return np.nan
        return (g["Sales"].iloc[-1] / g["Sales"].iloc[0]) ** (1 / (len(g) - 1)) - 1

    growth = subcat_year.groupby("Sub-Category").apply(growth_rate).rename("growth_rate")

    subcat_month = df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="MS")])["Sales"].sum().reset_index()
    volatility = subcat_month.groupby("Sub-Category")["Sales"].std().rename("volatility")

    features = subcat.set_index("Sub-Category").join(growth).join(volatility).dropna()

    X = features[["total_sales", "growth_rate", "volatility", "avg_order_value"]]
    X_scaled = StandardScaler().fit_transform(X)

    k = st.slider("Number of clusters (k)", min_value=2, max_value=6, value=4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    features["cluster"] = kmeans.fit_predict(X_scaled).astype(str)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    features["pca1"], features["pca2"] = coords[:, 0], coords[:, 1]

    fig = px.scatter(
        features.reset_index(), x="pca1", y="pca2", color="cluster",
        hover_name="Sub-Category", size="total_sales",
        title="Product Sub-Category Demand Clusters (PCA projection)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Categories by Cluster")
    st.dataframe(
        features.reset_index()[["Sub-Category", "cluster", "total_sales", "growth_rate", "volatility"]]
        .sort_values("cluster"),
        use_container_width=True,
    )
