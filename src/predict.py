# src/predict.py

import os
import pickle
import pandas as pd
import numpy as np

from src.preprocess import clean_data, prepare_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "../models/model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "../models/scaler.pkl"), "rb"))
scaler_before_cluster = pickle.load(open(os.path.join(BASE_DIR, "../models/scaler_before_cluster.pkl"), "rb"))
le = pickle.load(open(os.path.join(BASE_DIR, "../models/label_encoder.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "../models/columns.pkl"), "rb"))
kmeans = pickle.load(open(os.path.join(BASE_DIR, "../models/kmeans.pkl"), "rb"))

def predict(input_dict):
    df = pd.DataFrame([input_dict])

    df = clean_data(df)

    df["Area"] = df["Area"].replace(0, 1)

    df["Fertilizer_per_ha"] = df["Fertilizer"] / df["Area"]
    df["Pesticide_per_ha"] = df["Pesticide"] / df["Area"]
    df["Rainfall_per_month"] = df["Annual_Rainfall"] / 12
    df["Rain_x_Fert"] = df["Annual_Rainfall"] * df["Fertilizer_per_ha"]
    df["Area_log"] = np.log1p(df["Area"])

    df = df.drop(columns=["Fertilizer", "Pesticide", "Crop_Year"], errors="ignore")

    df = prepare_features(df)

    df = df.reindex(columns=columns, fill_value=0)

    # BEFORE cluster
    df_scaled_temp = scaler_before_cluster.transform(df)

    # cluster
    cluster = kmeans.predict(df_scaled_temp)
    df["cluster"] = cluster

    # FINAL
    final_columns = list(columns) + ["cluster"]
    df = df.reindex(columns=final_columns, fill_value=0)

    df_scaled = scaler.transform(df)

    proba = model.predict_proba(df_scaled)[0]

    top_idx = np.argsort(proba)[-3:][::-1]

    crops = le.inverse_transform(top_idx)
    probs = proba[top_idx]

    return list(zip(crops, probs))