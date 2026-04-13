# src/preprocess.py

import pandas as pd
import numpy as np

def clean_data(df):
    for col in ["Season", "State"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def feature_engineering(df):
    df["Area"] = df["Area"].replace(0, 1)

    df["Fertilizer_per_ha"] = df["Fertilizer"] / df["Area"]
    df["Pesticide_per_ha"] = df["Pesticide"] / df["Area"]
    df["Area_log"] = np.log1p(df["Area"])

    df = df.drop(columns=["Fertilizer", "Pesticide"], errors="ignore")

    return df

def prepare_features(df):
    df = pd.get_dummies(df, columns=["Season", "State"], drop_first=True)
    return df