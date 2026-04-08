# ==============================

# Cyber Intrusion Detection (FINAL CLEAN VERSION)

# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
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

# 🔥 Manual Prediction

# ==============================

st.markdown("## 🧠 Manual Prediction")

selected_features = features[:4]

with st.expander("🔍 Enter Data Manually"):

```
col1, col2 = st.columns(2)
input_data = {}

for i, feature in enumerate(selected_features):
    if i % 2 == 0:
        input_data[feature] = col1.number_input(feature, value=0.0)
    else:
        input_data[feature] = col2.number_input(feature, value=0.0)

if st.button("🚀 Predict"):
    try:
        input_df = pd.DataFrame([input_data])

        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[features]

        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]

        if pred == 1:
            st.error("🚨 Attack Detected!")
        else:
            st.success("✅ Normal Traffic")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
```

# ==============================

# Prediction Function

# ==============================

def predict_data(df):

```
df = df.dropna().copy()

y_true = None

# Handle label
if 'label' in df.columns:
    y_true = df['label']

    if y_true.dtype == 'object':
        y_true = y_true.map(lambda x: 1 if str(x).lower() in ['attack', 'anomaly', '1'] else 0)

    df = df.drop('label', axis=1)

# Encode features only
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# Match model features
X = pd.DataFrame(columns=features)

for col in df.columns:
    if col in X.columns:
        X[col] = df[col]

X = X.fillna(0)

X_scaled = scaler.transform(X)

preds = model.predict(X_scaled)
preds = np.array([0 if p == 0 else 1 for p in preds])

return preds, X, y_true
```

# ==============================

# Sidebar

# ==============================

st.sidebar.markdown("## 🔍 Filters")

view_option = st.sidebar.selectbox(
"Select Data View",
["All", "Normal Only", "Attack Only"]
)

# ==============================

# Upload CSV

# ==============================

st.markdown("## 📂 Upload Dataset")

file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
df = pd.read_csv(file)

```
preds, X, y_true = predict_data(df)
df['Prediction'] = preds

if view_option == "Normal Only":
    df = df[df['Prediction'] == 0]
elif view_option == "Attack Only":
    df = df[df['Prediction'] == 1]

total = len(df)
attacks = int(df['Prediction'].sum())
normal = total - attacks

numeric_df = X.select_dtypes(include=['int64', 'float64'])

# KPI
col1, col2, col3 = st.columns(3)

col1.markdown(f'<div class="glass">📊 Total<br><h2>{total}</h2></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="glass">✅ Normal<br><h2>{normal}</h2></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="glass">🚨 Attack<br><h2>{attacks}</h2></div>', unsafe_allow_html=True)

# Dashboard
st.markdown("## 📊 Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(px.bar(x=["Normal", "Attack"], y=[normal, attacks]), use_container_width=True)

with col2:
    st.plotly_chart(px.pie(names=["Normal", "Attack"], values=[normal, attacks]), use_container_width=True)

trend = df['Prediction'].cumsum()
st.plotly_chart(px.line(trend), use_container_width=True)

if not numeric_df.empty:
    st.plotly_chart(px.histogram(numeric_df, x=numeric_df.columns[0]), use_container_width=True)

if numeric_df.shape[1] >= 2:
    st.plotly_chart(px.box(numeric_df.iloc[:, :2]), use_container_width=True)

# Feature Importance
st.markdown("## 📈 Feature Importance")

if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(10)
    st.plotly_chart(px.bar(importances, orientation='h'), use_container_width=True)

# Evaluation
if y_true is not None and len(set(y_true)) > 1:

    st.markdown("## 📊 Model Evaluation")

    y_true = np.array(y_true).astype(int)
    preds = np.array(preds).astype(int)

    acc = accuracy_score(y_true, preds)
    st.success(f"🎯 Model Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_true, preds, labels=[0,1])

    cm_df = pd.DataFrame(cm,
                         index=["Actual Normal", "Actual Attack"],
                         columns=["Pred Normal", "Pred Attack"])

    st.plotly_chart(px.imshow(cm_df, text_auto=True), use_container_width=True)

else:
    st.warning("⚠️ Confusion Matrix not available")

# Logs
st.markdown("## 🚨 Attack Logs")
st.dataframe(df[df['Prediction'] == 1])

# Download
st.download_button("⬇️ Download Results", df.to_csv(index=False), "results.csv")
```
