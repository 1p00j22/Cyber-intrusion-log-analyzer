import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Cyber Intrusion Detection", layout="wide")

st.title("🔐 Cyber Intrusion Detection System")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_files():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_files()

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.write(data.head())

    # ==============================
    # CHECK TARGET COLUMN
    # ==============================
    if "label" not in data.columns:
        st.error("❌ 'label' column not found in dataset")
        st.stop()

    # ==============================
    # SPLIT DATA
    # ==============================
    X = data.drop("label", axis=1)
    y_true = data["label"]

    # ==============================
    # HANDLE LABELS (STRING → NUMBER)
    # ==============================
    le = LabelEncoder()
    y_true = le.fit_transform(y_true)

    # ==============================
    # SCALE DATA
    # ==============================
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"Scaling Error: {e}")
        st.stop()

    # ==============================
    # PREDICTION
    # ==============================
    try:
        preds = model.predict(X_scaled)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    # ==============================
    # CONVERT TO NUMPY
    # ==============================
    y_true = np.array(y_true)
    preds = np.array(preds)

    # ==============================
    # DEBUG (OPTIONAL)
    # ==============================
    st.write("Unique y_true:", np.unique(y_true))
    st.write("Unique preds:", np.unique(preds))

    # ==============================
    # CHECK SIZE
    # ==============================
    if len(y_true) != len(preds):
        st.error("❌ y_true and preds size mismatch")
        st.stop()

    # ==============================
    # ACCURACY
    # ==============================
    acc = accuracy_score(y_true, preds)
    st.success(f"✅ Accuracy: {acc:.2f}")

    # ==============================
    # CONFUSION MATRIX (SAFE)
    # ==============================
    try:
        cm = confusion_matrix(y_true, preds)
        st.subheader("📌 Confusion Matrix")
        st.write(cm)

        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Confusion Matrix Error: {e}")

    # ==============================
    # RESULT TABLE
    # ==============================
    data["Prediction"] = preds
    st.subheader("📄 Prediction Results")
    st.write(data.head())

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("🚀 Cyber Intrusion Detection System | Streamlit App")
