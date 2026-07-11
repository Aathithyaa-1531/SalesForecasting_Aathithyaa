# Corporate Sales Forecasting & Supply Chain Analytics Dashboard

An interactive corporate sales analytics suite combining statistical forecasting (SARIMA), machine learning anomaly detection (Isolation Forest), demand segmentation (K-Means clustering), and automated executive reporting.

---

## 🚀 Features & Workspace Panels

### Page 1 — Sales Overview Dashboard
*   **Dynamic KPIs:** Total sales, unique order volumes, and average shipping days updated dynamically based on active filter selections.
*   **Trend Analysis:** Displays year-over-year sales and monthly sales charts using non-interactive Altair plots to prevent scroll-zoom issues.
*   **Interactive Filters:** Multi-select pills at the top to filter dashboard elements by Region and Category.
*   **Data Ledger:** A pivot table showing filtered sales summaries.

### Page 2 — Forecast Explorer
*   **Interactive Controls:** Options to filter by Region or Category and slide the forecasting horizon up to 3 months ahead.
*   **Predictive Model:** Forecasts monthly sales using **SARIMA (1,1,1)x(1,1,1,12)**, displaying confidence boundaries on a line plot.
*   **Numeric Tables & Metrics:** Displays predicted sales alongside model performance metrics (MAE and RMSE).
*   **Model Selection Insights:** Contextual analysis detailing why SARIMA was chosen over Facebook Prophet and XGBoost (achieving a **11.45% MAPE**).

### Page 3 — Anomaly Report
*   **Outlier Visualization:** Plots weekly sales outliers detected using Isolation Forest.
*   **Ledger Table:** Lists all anomalous week-ending dates alongside their transaction volumes and likely business causes (e.g. Black Friday surges or large B2B contract sign-offs).

### Page 4 — Product Demand Segments
*   **Spatial Cluster Map:** Visualizes sub-categories grouped by K-Means clustering (using volume and volatility features).
*   **Stocking Strategy Framework:** Displays an operational inventory strategy for each cluster:
    *   *Segment 0 (High-Value Capital Goods):* Just-In-Time (JIT) / Pull procurement.
    *   *Segment 1 (Core Staples):* Fixed Reorder Points (ROP) & Continuous replenishment.
    *   *Segment 2 (Steady Staples):* Periodic Review (Min-Max framework).

---

## 🛠️ Technology Stack
*   **Dashboard:** Streamlit
*   **Data Processing & Modeling:** Pandas, NumPy, Scikit-learn, Statsmodels
*   **Visualization:** Altair, Matplotlib, Seaborn
*   **Reporting:** ReportLab (PDF compilation engine)

---

## 📁 Project Directory Structure
*   `app.py` — The core Streamlit application.
*   `analysis.ipynb` — Jupyter notebook showcasing initial EDA, model fitting, and segment optimizations.
*   `generate_pdf_report.py` — Python utility script compiling the PDF business briefing.
*   `summary.pdf` — Generated 2-page executive report tailored for the CFO and Head of Supply Chain.
*   `requirements.txt` — List of pinned python dependencies for setup replication.
*   `train.csv` — Historical raw orders dataset.
*   `charts/` — Folder housing generated asset figures.

---

## 📦 Getting Started

### 1. Installation
Clone the repository and install all required libraries using the pinned configuration:
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
Start the local Streamlit dashboard server:
```bash
streamlit run app.py
```

### 3. Generate the Executive PDF Report
Run the report generation script to rebuild `summary.pdf`:
```bash
python generate_pdf_report.py
```

---

## 👤 Author
*   **Name:** Aathithyaa
*   **GitHub Profile:** [@Aathithyaa-1531](https://github.com/Aathithyaa-1531)
