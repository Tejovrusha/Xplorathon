import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

# Page Config
st.set_page_config(page_title="AI-Powered SOAR Dashboard", layout="wide")
st.title("⚙️ Context-Aware Automated Cybersecurity Workflow")

# --- Generate Mock Data ---
np.random.seed(42)
alerts = pd.DataFrame({
    "Alert ID": [f"A-{i:04}" for i in range(1, 11)],
    "Source IP": np.random.choice(["192.168.1.10", "10.0.0.22", "172.16.0.5", "203.0.113.7"], 10),
    "User": np.random.choice(["alice", "bob", "charlie", "root", "guest"], 10),
    "Severity": np.random.choice(["Low", "Medium", "High", "Critical"], 10),
    "Confidence (%)": np.random.randint(55, 99, 10),
    "Classification": np.random.choice(["True Positive", "False Positive", "Suspicious"], 10),
    "Action": np.random.choice(["Auto-Blocked", "Pending Approval", "Dismissed"], 10)
})

# --- Summary Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("🔍 Alerts Processed", "10", "+3 new")
col2.metric("⚡ Auto-Resolved", "5", "+2 improved")
col3.metric("📉 False Positive Rate", "22%", "-8% vs last run")
col4.metric("🧠 Model Confidence Avg", f"{alerts['Confidence (%)'].mean():.1f}%", "+3.2%")

# --- Visualization ---
st.markdown("### 📊 Alert Confidence Distribution")
fig = px.histogram(alerts, x="Confidence (%)", nbins=10, color="Severity", title="Confidence Score Distribution")
st.plotly_chart(fig, use_container_width=True)

# --- Live Alert Feed ---
st.markdown("### 🚨 Active Alerts Overview")
st.dataframe(alerts, use_container_width=True)

# --- Explainability Section ---
st.markdown("### 🧩 Explainability Insights (SHAP-style Example)")

example_alert = alerts.sample(1).iloc[0]
st.info(f"**Selected Alert:** {example_alert['Alert ID']} | {example_alert['Classification']} | Confidence: {example_alert['Confidence (%)']}%")

exp_data = pd.DataFrame({
    "Feature": ["IP Reputation", "Failed Logins (1h)", "Asset Criticality", "Recent Alerts", "User Role Risk"],
    "Impact": [0.45, 0.30, 0.12, 0.08, 0.05]
})
exp_fig = px.bar(exp_data, x="Impact", y="Feature", orientation="h", title="Top Contributing Features")
st.plotly_chart(exp_fig, use_container_width=True)

# --- Analyst Feedback Simulation ---
st.markdown("### 🧍 Analyst Feedback Portal")
selected = st.selectbox("Select an alert to review:", alerts["Alert ID"])
feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
if feedback_col1.button("✅ Confirm True Positive"):
    st.success(f"Feedback recorded: {selected} marked as **True Positive**")
elif feedback_col2.button("❌ Mark False Positive"):
    st.warning(f"Feedback recorded: {selected} marked as **False Positive**")
elif feedback_col3.button("🧩 Needs Investigation"):
    st.info(f"Feedback recorded: {selected} marked as **Needs Investigation**")

# --- Model Update Simulation ---
st.markdown("### 🔄 Model Feedback & Continuous Learning")
progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i + 1)
st.success("Model retrained using latest analyst feedback and new alert patterns ✅")

# Footer
st.markdown("---")
st.caption("🚀 Developed for Cybersecurity Hackathon | Context-Aware SOAR Prototype")
