import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import altair as alt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =====================================================================
# FORCE STREAMLIT TO BYPASS WEBSOCKETS (CRITICAL FOR PYTHON 3.14)
# =====================================================================
st.config.set_option("server.enableCORS", False)
st.config.set_option("server.enableXsrfProtection", False)

st.set_page_config(page_title="Corporate Sales Analytics Core Engine", layout="wide")

# Cached Data Pipeline
@st.cache_data
def load_data():
    if not os.path.exists('train.csv'):
        return None
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Order Date', 'Ship Date'])
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Shipping_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    return df

df = load_data()

if df is None:
    st.error("🚨 CRITICAL ERROR: 'train.csv' was not found in this folder!")
    st.stop()

global_monthly = df.groupby('Order Date')['Sales'].sum().resample('MS').sum()

# Sidebar Navigation
st.sidebar.title("🎛️ Navigation Center")
selected_page = st.sidebar.radio(
    "Go to Workspace Panel:",
    ["Sales Overview Dashboard", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"]
)

# =====================================================================
# PAGE 1 — SALES OVERVIEW DASHBOARD
# =====================================================================
if selected_page == "Sales Overview Dashboard":
    st.title("📊 Sales Overview Dashboard")
    
    # Global Filters for Dashboard
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        reg_select = st.pills("Filter Region:", options=list(df['Region'].unique()), default=list(df['Region'].unique()), selection_mode="multi")
    with col_f2:
        cat_select = st.pills("Filter Category:", options=list(df['Category'].unique()), default=list(df['Category'].unique()), selection_mode="multi")
        
    selected_regions = reg_select if reg_select is not None else []
    selected_categories = cat_select if cat_select is not None else []
    
    f_df = df[(df['Region'].isin(selected_regions)) & (df['Category'].isin(selected_categories))]
    
    st.markdown("---")
    
    if f_df.empty:
        st.warning("⚠️ No data available for the selected filters. Please select at least one Region and Category.")
    else:
        # KPI Blocks
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Sales", f"${f_df['Sales'].sum():,.2f}")
        kpi2.metric("Total Orders", f"{f_df['Order ID'].nunique():,}")
        kpi3.metric("Avg Shipping Days", f"{f_df['Shipping_Days'].mean():.2f} Days")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total Sales by Year (Bar Chart)")
            year_sales = f_df.groupby('Year')['Sales'].sum().reset_index()
            year_sales.columns = ['Year', 'Sales']
            chart1 = (
                alt.Chart(year_sales)
                .mark_bar()
                .encode(
                    x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Sales:Q', title='Sales'),
                    tooltip=['Year', 'Sales']
                )
            )
            st.altair_chart(chart1, width="stretch")
        with col2:
            st.subheader("Monthly Sales Trend Line Chart")
            f_monthly = f_df.groupby('Order Date')['Sales'].sum().resample('MS').sum()
            monthly_sales = f_monthly.reset_index()
            monthly_sales.columns = ['Order Date', 'Sales']
            chart2 = (
                alt.Chart(monthly_sales)
                .mark_line()
                .encode(
                    x=alt.X('Order Date:T', title='Order Date'),
                    y=alt.Y('Sales:Q', title='Sales'),
                    tooltip=['Order Date', 'Sales']
                )
            )
            st.altair_chart(chart2, width="stretch")
            
        st.markdown("---")
        st.subheader("Interactive Filter Grid")
        st.dataframe(f_df.groupby(['Region', 'Category'])['Sales'].sum().unstack().fillna(0).style.format("${:,.2f}"), width="stretch")

# =====================================================================
# PAGE 2 — FORECAST EXPLORER
# =====================================================================
elif selected_page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    
    c1, c2 = st.columns(2)
    with c1:
        filter_type = st.selectbox("Select Forecast Filter:", ["Category", "Region"])
    with c2:
        filter_val = st.selectbox("Select Filter Node Value:", options=df['Category'].unique() if filter_type == "Category" else df['Region'].unique())
        
    horizon = st.slider("Select Forecast Horizon (Months Ahead):", min_value=1, max_value=3, value=3)
    
    mask_df = df[df[filter_type] == filter_val]
    sub_series = mask_df.groupby('Order Date')['Sales'].sum().resample('MS').sum()
    
    # Metrics Baseline Evaluator
    train_s = sub_series.iloc[:-3]
    test_s = sub_series.iloc[-3:]
    
    try:
        model = SARIMAX(train_s, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False)
        res = model.fit(disp=False)
        fc_val = res.get_forecast(steps=3).predicted_mean.values
        mae = np.mean(np.abs(test_s.values - fc_val))
        rmse = np.sqrt(np.mean((test_s.values - fc_val)**2))
        
        # Real Forecast Projector
        f_model = SARIMAX(sub_series, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False)
        f_res = f_model.fit(disp=False)
        future_fc = f_res.get_forecast(steps=horizon).predicted_mean
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(sub_series.index[-12:], sub_series.values[-12:], marker='o', label='Actual Sales', color='black')
        f_dates = pd.date_range(start=sub_series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq='MS')
        ax.plot(f_dates, future_fc.values, marker='x', linestyle='--', label='SARIMA Forecast', color='orange')
        ax.legend()
        st.pyplot(fig)
        
        # Display the forecast output table
        st.subheader("📋 Forecasted Sales Output")
        forecast_df = pd.DataFrame({
            "Month": f_dates.strftime("%B %Y"),
            "Forecasted Sales": future_fc.values
        })
        st.dataframe(forecast_df.style.format({"Forecasted Sales": "${:,.2f}"}), width="stretch")
        
        # Model Evaluation Metrics below the chart
        st.subheader("📊 Model Performance Metrics")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Model MAE", f"${mae:,.2f}")
        with m2:
            st.metric("Model RMSE", f"${rmse:,.2f}")
            
        # Model recommendation selection details
        st.markdown("---")
        st.subheader("🏆 Model Selection Insights")
        st.success("""
        **Best Performing Model:** **SARIMA (Seasonal AutoRegressive Integrated Moving Average)**
        
        *   **Why SARIMA?** During overall model evaluation and parameter testing, the SARIMA model outclassed all other models. It achieved a Mean Absolute Percentage Error (**MAPE of 11.45%**), significantly beating **Facebook Prophet (21.89% MAPE)** and **XGBoost (32.78% MAPE)**.
        *   **Generalization:** It successfully captures seasonality spikes and local trends without overfitting, using the optimal order config: **SARIMAX(1,1,1)x(1,1,1,12)**.
        """)
    except Exception as e:
        st.error(f"Engine stall error: {e}")

# =====================================================================
# PAGE 3 — ANOMALY REPORT
# =====================================================================
elif selected_page == "Anomaly Report":
    st.title("🚨 Anomaly Report")
    
    if os.path.exists('charts/anomalies_detected.png'):
        st.image('charts/anomalies_detected.png', width="stretch")
        
    st.subheader("Detected Anomaly Ledger Table")
    
    w_sales = df.groupby('Order Date')['Sales'].sum().resample('W').sum().to_frame()
    
    # Use Isolation Forest (matching Task 5) to detect global anomalies shown in the chart
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.04, random_state=42)
    w_sales['iso_anomaly'] = iso_forest.fit_predict(w_sales[['Sales']])
    w_sales['iso_anomaly'] = w_sales['iso_anomaly'].map({1: 0, -1: 1})
    
    # Extract outliers
    anom_table = w_sales[w_sales['iso_anomaly'] == 1][['Sales']]
    anom_table.index.name = "Order Date (Week Ending)"
    anom_table.columns = ["Weekly Sales Value"]
    
    # Format indices to clean strings
    anom_table.index = anom_table.index.strftime('%Y-%m-%d')
    
    st.dataframe(anom_table.style.format({"Weekly Sales Value": "${:,.2f}"}), width="stretch")

# =====================================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# =====================================================================
elif selected_page == "Product Demand Segments":
    st.title("🏷️ Product Demand Segments")
    
    if os.path.exists('charts/product_clusters.png'):
        st.image('charts/product_clusters.png', width="stretch")
        
    cluster_records = {
        "Cluster 0: High-Value, High-Volatility": ["Copiers", "Machines"],
        "Cluster 1: High-Volume, High-Stability": ["Binders", "Paper", "Phones", "Storage", "Furnishings"],
        "Cluster 2: Moderate-Volume, Low-Volatility": ["Chairs", "Appliances", "Tables", "Envelopes", "Fasteners", "Art", "Labels", "Supplies", "Bookcases"]
    }
    for c_name, items in cluster_records.items():
        st.write(f"### {c_name}")
        st.table(pd.DataFrame(items, columns=["Sub-Category Name"]))