# app/streamlit_app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from deep_translator import GoogleTranslator

# ----------------------------
# Load Models
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "../models/model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "../models/scaler.pkl"), "rb"))
scaler_before_cluster = pickle.load(open(os.path.join(BASE_DIR, "../models/scaler_before_cluster.pkl"), "rb"))
le = pickle.load(open(os.path.join(BASE_DIR, "../models/label_encoder.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "../models/columns.pkl"), "rb"))
kmeans = pickle.load(open(os.path.join(BASE_DIR, "../models/kmeans.pkl"), "rb"))

# ----------------------------
# Load dataset (for dropdown)
# ----------------------------
data = pd.read_csv(os.path.join(BASE_DIR, "../data/crop_yield.csv"))
states_list = sorted(data["State"].dropna().unique())

# ----------------------------
# Language
# ----------------------------
st.sidebar.title("🌐 Language Settings")

language_map = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn"
}

selected_lang = st.sidebar.selectbox("Select Language", list(language_map.keys()))
target_lang = language_map[selected_lang]

def t(text):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        return text

def to_english(text):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source=target_lang, target='en').translate(text)
    except:
        return text

# ----------------------------
# UI
# ----------------------------
st.title(t("🌱 Crop Recommendation System"))
st.markdown(t("Fill in your farm details to get the best crop suggestions."))

with st.form("form"):

    col1, col2 = st.columns(2)

    with col1:
        season = st.selectbox(t("Season"), ["Kharif", "Rabi", "Whole Year"])
        state = st.selectbox(t("State"), states_list)
        area = st.number_input(t("Area"), value=100.0)

    with col2:
        rainfall = st.number_input(t("Rainfall"), value=1000.0)
        fertilizer = st.number_input(t("Fertilizer"), value=1000.0)
        pesticide = st.number_input(t("Pesticide"), value=50.0)

    submit = st.form_submit_button(t("Recommend Crop"))

# ----------------------------
# Explainability (simple)
# ----------------------------
def explain_crop(input_data):
    reasons = []

    if input_data["Annual_Rainfall"] > 800:
        reasons.append("High rainfall")

    if input_data["Fertilizer"] > 500:
        reasons.append("Good fertilizer usage")

    if input_data["Area"] > 50:
        reasons.append("Large farm size")

    if input_data["Season"] == "Kharif":
        reasons.append("Season suitability")

    return ", ".join(reasons)

# ----------------------------
# Prediction
# ----------------------------
if submit:

    input_data = {
        "Season": to_english(season),
        "State": to_english(state),
        "Area": area,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }

    df = pd.DataFrame([input_data])

    # Feature engineering
    df["Area"] = df["Area"].replace(0, 1)
    df["Fertilizer_per_ha"] = df["Fertilizer"] / df["Area"]
    df["Pesticide_per_ha"] = df["Pesticide"] / df["Area"]
    df["Rainfall_per_month"] = df["Annual_Rainfall"] / 12
    df["Rain_x_Fert"] = df["Annual_Rainfall"] * df["Fertilizer_per_ha"]
    df["Area_log"] = np.log1p(df["Area"])

    df = df.drop(columns=["Fertilizer", "Pesticide"], errors="ignore")

    # Encoding
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    # Scaling before cluster
    df_scaled_temp = scaler_before_cluster.transform(df)

    # Cluster
    cluster = kmeans.predict(df_scaled_temp)
    df["cluster"] = cluster

    # Final alignment
    final_columns = list(columns) + ["cluster"]
    df = df.reindex(columns=final_columns, fill_value=0)

    # Final scaling
    df_scaled = scaler.transform(df)

    # Prediction
    proba = model.predict_proba(df_scaled)[0]
    top_idx = np.argsort(proba)[-3:][::-1]

    crops = le.inverse_transform(top_idx)
    confidences = proba[top_idx] * 100

    # Output
    st.success(t("Top Crop Recommendations"))

    for i, (crop, conf) in enumerate(zip(crops, confidences), 1):
        st.info(f"{i}. {t(crop)} — {conf:.2f}%")
        st.caption(t("Reason: ") + explain_crop(input_data))

    # Chart
    st.subheader(t("Confidence Distribution"))

    chart_data = pd.DataFrame({
        "Crop": [t(c) for c in crops],
        "Confidence": confidences
    })

    st.bar_chart(chart_data.set_index("Crop"))