# ==============================
# Cyber Intrusion Detection (FINAL PRO VERSION)
# ==============================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

# ==============================
# Config
# ==============================
st.set_page_config(page_title="Intrusion Dashboard", layout="wide")

# ==============================
# UI STYLE
# ==============================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(10px);
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #00ffd5;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Model
# ==============================
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

# ==============================
# Title
# ==============================
st.markdown('<p class="title">🔐 Cyber Intrusion Dashboard</p>', unsafe_allow_html=True)
st.markdown("### 🌐 Interactive Security Analytics")

# ==============================
# Prediction Function
# ==============================
def predict_data(df):

    df = df.dropna()

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    y_true = None
    if 'label' in df.columns:
        y_true = df['label']
        df = df.drop('label', axis=1)

    X = pd.DataFrame(columns=features)

    for col in df.columns:
        if col in X.columns:
            X[col] = df[col]

    X = X.fillna(0)

    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    preds = [0 if p == 0 else 1 for p in preds]

    return preds, X, y_true

# ==============================
# Sidebar Filter
# ==============================
st.sidebar.markdown("## 🔍 Filters")

view_option = st.sidebar.selectbox(
    "Select Data View",
    ["All", "Normal Only", "Attack Only"]
)

# ==============================
# File Upload
# ==============================
file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)

    preds, X, y_true = predict_data(df)
    df['Prediction'] = preds

    # Apply Filter
    if view_option == "Normal Only":
        df = df[df['Prediction'] == 0]
    elif view_option == "Attack Only":
        df = df[df['Prediction'] == 1]

    total = len(df)
    attacks = df['Prediction'].sum()
    normal = total - attacks

    numeric_df = X.select_dtypes(include=['int64', 'float64'])

    # ==============================
    # KPI CARDS
    # ==============================
    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="glass">📊 Total<br><h2>{total}</h2></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="glass">✅ Normal<br><h2>{normal}</h2></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="glass">🚨 Attack<br><h2>{attacks}</h2></div>', unsafe_allow_html=True)

    # ==============================
    # DASHBOARD
    # ==============================
    st.markdown("## 📊 Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.bar(x=["Normal", "Attack"], y=[normal, attacks], title="Traffic Overview"), use_container_width=True)

    with col2:
        st.plotly_chart(px.pie(names=["Normal", "Attack"], values=[normal, attacks], hole=0.5, title="Attack %"), use_container_width=True)

    # Trend
    trend = df['Prediction'].cumsum()
    st.plotly_chart(px.line(trend, title="Attack Trend"), use_container_width=True)

    # Distribution
    st.plotly_chart(px.histogram(numeric_df, x=numeric_df.columns[0], title="Distribution"), use_container_width=True)

    # Boxplot
    st.plotly_chart(px.box(numeric_df.iloc[:, :2], title="Outliers"), use_container_width=True)

    # ==============================
    # Feature Importance
    # ==============================
    st.markdown("## 📈 Feature Importance")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(10)
        st.plotly_chart(px.bar(importances, orientation='h', title="Top Features"), use_container_width=True)

    # ==============================
    # ML Evaluation (FIXED)
    # ==============================
    if y_true is not None and len(set(y_true)) > 1:

        st.markdown("## 📊 Model Evaluation")

        # Accuracy
        acc = accuracy_score(y_true, preds)
        st.success(f"🎯 Model Accuracy: {acc:.2f}")

        # Confusion Matrix (FIXED)
        cm = confusion_matrix(y_true, preds, labels=[0,1])

        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=["Normal", "Attack"],
            y=["Normal", "Attack"],
            colorscale='Blues'
        )

        st.plotly_chart(fig_cm, use_container_width=True)

    else:
        st.warning("⚠️ Confusion Matrix not available (only one class present)")

    # ==============================
    # Attack Logs
    # ==============================
    st.markdown("## 🚨 Attack Logs")
    st.dataframe(df[df['Prediction'] == 1])

    # ==============================
    # Download
    # ==============================
    st.download_button("⬇️ Download Results", df.to_csv(index=False), "results.csv")