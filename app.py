
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📊",
    layout="wide"
)

CSV_PATH = Path(__file__).parent / "Customer_Intelligence_Dashboard.csv"


@st.cache_data
def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            "Customer_Intelligence_Dashboard.csv was not found "
            "in the repository root."
        )

    data = pd.read_csv(CSV_PATH, low_memory=False)

    # Remove accidental spaces from column names.
    data.columns = data.columns.str.strip()

    # Convert known numeric dashboard fields safely.
    numeric_columns = [
        "ChurnProbability",
        "ChurnProbabilityPercent",
        "RevenueAtRisk",
        "EstimatedCLV",
        "PriorityScore",
        "MonthlyCharges",
        "TotalCharges",
        "tenure"
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the dashboard data: {exc}")
    st.stop()


st.title("📊 Customer Retention Intelligence")
st.caption(
    "IBM Telco Customer Churn | XGBoost-based risk analysis "
    "and customer prioritization"
)

st.info(
    "This dashboard displays predictions already calculated "
    "by the project's Python workflow. It does not retrain "
    "or execute the XGBoost model in this web app."
)

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.header("Dashboard Filters")

filtered = df.copy()

if "RiskCategory" in filtered.columns:
    risk_options = sorted(
        filtered["RiskCategory"].dropna().astype(str).unique()
    )
    selected_risks = st.sidebar.multiselect(
        "Risk category",
        options=risk_options,
        default=risk_options
    )
    filtered = filtered[
        filtered["RiskCategory"].astype(str).isin(selected_risks)
    ]

for col in ["Contract", "InternetService", "PaymentMethod"]:
    if col in filtered.columns:
        options = sorted(
            filtered[col].dropna().astype(str).unique()
        )
        selected = st.sidebar.multiselect(
            col,
            options=options,
            default=options
        )
        filtered = filtered[
            filtered[col].astype(str).isin(selected)
        ]

# ---------------- KPI CARDS ----------------

total_customers = len(filtered)

critical_count = (
    filtered["RiskCategory"].astype(str).eq("Critical").sum()
    if "RiskCategory" in filtered.columns else 0
)

revenue_at_risk = (
    filtered["RevenueAtRisk"].sum()
    if "RevenueAtRisk" in filtered.columns else 0
)

if "ChurnProbability" in filtered.columns:
    avg_probability = filtered["ChurnProbability"].mean()
    if pd.notna(avg_probability) and avg_probability > 1:
        avg_probability = avg_probability / 100
else:
    avg_probability = None

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("Critical-Risk Customers", f"{critical_count:,}")
k3.metric("Revenue at Risk", f"${revenue_at_risk:,.2f}")
k4.metric(
    "Average Churn Probability",
    f"{avg_probability:.1%}"
    if pd.notna(avg_probability) else "N/A"
)

st.divider()

# ---------------- CHARTS ----------------

left, right = st.columns(2)

with left:
    st.subheader("Customer Risk Distribution")

    if "RiskCategory" in filtered.columns:
        risk_counts = (
            filtered["RiskCategory"]
            .value_counts()
            .rename_axis("Risk Category")
            .to_frame("Customers")
        )
        st.bar_chart(risk_counts)
    else:
        st.warning("RiskCategory column is unavailable.")

with right:
    st.subheader("Revenue at Risk by Contract")

    if all(
        col in filtered.columns
        for col in ["Contract", "RevenueAtRisk"]
    ):
        contract_risk = (
            filtered.groupby("Contract")["RevenueAtRisk"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(contract_risk)
    else:
        st.warning(
            "Contract or RevenueAtRisk column is unavailable."
        )

st.divider()

# ---------------- CUSTOMER PRIORITY TABLE ----------------

st.subheader("Customers Requiring Attention")

sort_column = (
    "PriorityScore"
    if "PriorityScore" in filtered.columns
    else "RevenueAtRisk"
    if "RevenueAtRisk" in filtered.columns
    else None
)

if sort_column:
    priority_df = filtered.sort_values(
        sort_column, ascending=False, na_position="last"
    )
else:
    priority_df = filtered

preferred_columns = [
    "customerID",
    "CustomerID",
    "Churn",
    "RiskCategory",
    "ChurnProbability",
    "ChurnProbabilityPercent",
    "MonthlyCharges",
    "RevenueAtRisk",
    "EstimatedCLV",
    "PriorityScore",
    "CustomerSegment",
    "RetentionRecommendation"
]

display_columns = [
    col for col in preferred_columns
    if col in priority_df.columns
]

if not display_columns:
    display_columns = priority_df.columns.tolist()

st.dataframe(
    priority_df[display_columns].head(100),
    use_container_width=True,
    hide_index=True
)

csv_data = priority_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered customer data",
    data=csv_data,
    file_name="filtered_customer_intelligence.csv",
    mime="text/csv"
)

st.caption(
    "Portfolio demonstration only. Revenue at risk and estimated "
    "customer lifetime value are model-based estimates, not "
    "guaranteed financial outcomes."
)
