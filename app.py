# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import time
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import plotly.express as px
import json
import os

# ---------- Helper: DB for feedback ----------
DB_PATH = "feedback.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id TEXT,
        label TEXT,
        comment TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

def save_feedback(alert_id, label, comment):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback (alert_id, label, comment) VALUES (?,?,?)",
                (alert_id, label, comment))
    conn.commit()
    conn.close()

def get_feedback_count():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback")
    c = cur.fetchone()[0]
    conn.close()
    return c

# ---------- Synthetic dataset + model training ----------
@st.cache_resource
def generate_synthetic_alerts(n=1200, seed=42):
    rng = np.random.RandomState(seed)
    # features: failed_logins_1h, hour_of_day, asset_criticality(0-2), ip_reputation(0-1), recent_related_alerts
    failed = rng.poisson(2, size=n)
    hour = rng.randint(0,24,size=n)
    asset = rng.choice([0,1,2], size=n, p=[0.6,0.3,0.1])
    ip_rep = rng.rand(n)
    related = rng.poisson(1, size=n)
    # label: true incident if failed_logins high AND ip_rep high and asset critical
    label = ((failed >= 5) & (ip_rep > 0.6) & (asset==2)).astype(int)
    df = pd.DataFrame({
        "alert_id":[f"A-{i:04d}" for i in range(n)],
        "failed_logins_1h": failed,
        "hour_of_day": hour,
        "asset_criticality": asset,
        "ip_reputation": ip_rep,
        "recent_related_alerts": related,
        "label": label
    })
    return df

@st.cache_resource
def train_models(df):
    X = df[["failed_logins_1h","hour_of_day","asset_criticality","ip_reputation","recent_related_alerts"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=False)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # anomaly detector trained on X (treat labelless unsupervised)
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train)
    # simple baseline metrics (for UI)
    preds = clf.predict(X_test)
    precision = precision_score(y_test, preds, zero_division=0)
    return clf, iso, precision

# ---------- Explainability approximation ----------
def approximate_shap_like(clf, X_row, feature_names):
    """
    Quick per-row contribution approximation: feature_value * (feature importance normalized)
    This is NOT SHAP but a demo-friendly approximation to display top contributors.
    """
    fi = clf.feature_importances_
    # normalize importances
    if fi.sum() == 0:
        fi_norm = fi
    else:
        fi_norm = fi / fi.sum()
    contributions = {}
    for i, f in enumerate(feature_names):
        # scale numeric features to similar ranges for demo
        v = X_row[f]
        # scale hour_of_day into 0..1
        if f == "hour_of_day":
            v = v / 23.0
        if f == "asset_criticality":
            v = v / 2.0
        contrib = fi_norm[i] * float(v)
        contributions[f] = contrib
    # sort descending
    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    return sorted_contrib[:4]

# ---------- App UI ----------
st.set_page_config(layout="wide", page_title="Context-Aware SOAR Demo")
init_db()

st.title("Context-Aware SOAR Dashboard — Demo")

# left: summary top
col1, col2, col3 = st.columns([3,1,1])
df = generate_synthetic_alerts()
clf, iso, base_precision = train_models(df)
total_alerts = len(df)
avg_confidence_demo = 87  # mock stat for header
false_positive_rate_demo = 0.20  # mock

with col1:
    st.metric("Total Alerts (demo)", total_alerts)
with col2:
    st.metric("Confidence Avg", f"{avg_confidence_demo}%")
with col3:
    st.metric("False Positive Rate", f"{int(false_positive_rate_demo*100)}%")

# Confidence distribution (simulate using classifier predict_proba)
X = df[["failed_logins_1h","hour_of_day","asset_criticality","ip_reputation","recent_related_alerts"]]
probs = clf.predict_proba(X)[:,1]
bins = np.linspace(0,1,11)
hist, edges = np.histogram(probs, bins=bins)
hist_df = pd.DataFrame({"bin":[f"{int(edges[i]*100)}%" for i in range(len(edges)-1)], "count":hist})

fig = px.bar(hist_df, x="bin", y="count", title="Confidence Distribution (demo)")
st.plotly_chart(fig, use_container_width=True)

# Contextual Alert Table (top N alerts by confidence)
df["confidence"] = (probs*100).round().astype(int)
table_df = df[["alert_id","asset_criticality","confidence"]].sort_values("confidence", ascending=False).head(10)
table_df["asset"] = table_df["asset_criticality"].map({0:"Workstation",1:"Server",2:"Critical Server"})
table_df = table_df[["alert_id","asset","confidence"]]
st.subheader("Top Alerts (by confidence)")
st.dataframe(table_df, height=260)

# Side explainability + feedback
st.sidebar.header("Alert Inspector")
sel = st.sidebar.selectbox("Select alert", table_df["alert_id"].tolist())
alert_row = df[df["alert_id"]==sel].iloc[0]
feature_names = ["failed_logins_1h","hour_of_day","asset_criticality","ip_reputation","recent_related_alerts"]
explanation = approximate_shap_like(clf, alert_row, feature_names)
st.sidebar.subheader("Explainability (approx.)")
for feat, val in explanation:
    st.sidebar.write(f"- **{feat}** : {val:.3f}")

# NLP-like summary (template)
nlp_summary = f"Confidence {int(alert_row['confidence'])}% — top reasons: {', '.join([f for f,_ in explanation])}."
st.sidebar.markdown(f"**NLP Summary:** {nlp_summary}")

st.sidebar.markdown("---")
st.sidebar.subheader("Analyst Feedback")
fb_col1, fb_col2 = st.sidebar.columns([1,1])
with fb_col1:
    if st.button("Approve"):
        save_feedback(sel, "approve", "")
        st.success("Approved — feedback saved")
with fb_col2:
    if st.button("Reject"):
        save_feedback(sel, "reject", "")
        st.error("Rejected — feedback saved")
comment = st.sidebar.text_input("Comment (optional)")
if st.sidebar.button("Submit Comment"):
    save_feedback(sel, "comment", comment)
    st.sidebar.success("Comment saved")

# System metrics / KPIs area
st.subheader("System KPIs")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Alerts / min (demo)", 130)
k2.metric("Model Precision (demo)", f"{base_precision:.2f}")
k3.metric("MTTR reduction", "↓ 30%")
k4.metric("Feedback stored", get_feedback_count())

# Simple REST API stubs info for integration
st.markdown("---")
st.markdown("**Integration**: The demo exposes two example endpoints (in production these would be real API endpoints):")
st.code("""
POST /api/v1/alert_infer  -> returns confidence, explainability, suggested_action
POST /api/v1/feedback     -> store analyst feedback for retraining
""", language="bash")

st.markdown("**Notes:** This demo approximates explainability for speed. For production, integrate SHAP and persistent model training pipelines.")

st.markdown("---")
st.write("Demo created for hackathon presentation. Use `pip install -r requirements.txt` then `streamlit run app.py`.")
