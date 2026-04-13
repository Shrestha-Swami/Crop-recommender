# src/train.py

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from xgboost import XGBClassifier
from sklearn.cluster import KMeans

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "../data/crop_yield.csv")

data = pd.read_csv(data_path)

for col in ["Season", "State"]:
    data[col] = data[col].str.strip()

data = data.drop(columns=["Production", "Yield"])

# =========================
# FILTER TOP 15 CROPS
# =========================
top_crops = data["Crop"].value_counts().nlargest(15).index
data = data[data["Crop"].isin(top_crops)]

# =========================
# FEATURE ENGINEERING
# =========================
data["Area"] = data["Area"].replace(0, 1)

data["Fertilizer_per_ha"] = data["Fertilizer"] / data["Area"]
data["Pesticide_per_ha"] = data["Pesticide"] / data["Area"]
data["Rainfall_per_month"] = data["Annual_Rainfall"] / 12
data["Rain_x_Fert"] = data["Annual_Rainfall"] * data["Fertilizer_per_ha"]
data["Area_log"] = np.log1p(data["Area"])

data = data.drop(columns=["Fertilizer", "Pesticide", "Crop_Year"])

# =========================
# SPLIT
# =========================
X = data.drop("Crop", axis=1)
y = data["Crop"]

# One-hot encoding
X = pd.get_dummies(X, columns=["Season", "State"], drop_first=True)

# Save base columns (before cluster)
model_columns = X.columns

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =========================
# SCALER 1 (BEFORE CLUSTER)
# =========================
scaler_before_cluster = StandardScaler()
X_scaled_temp = scaler_before_cluster.fit_transform(X)

# =========================
# CLUSTER
# =========================
kmeans = KMeans(n_clusters=8, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled_temp)

X["cluster"] = cluster_labels

# =========================
# SCALER 2 (FINAL)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = XGBClassifier(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Top-3 Accuracy:", top_k_accuracy_score(y_test, proba, k=3))

# =========================
# SAVE
# =========================
models_path = os.path.join(BASE_DIR, "../models")
os.makedirs(models_path, exist_ok=True)

pickle.dump(model, open(os.path.join(models_path, "model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(models_path, "scaler.pkl"), "wb"))
pickle.dump(scaler_before_cluster, open(os.path.join(models_path, "scaler_before_cluster.pkl"), "wb"))
pickle.dump(le, open(os.path.join(models_path, "label_encoder.pkl"), "wb"))
pickle.dump(model_columns, open(os.path.join(models_path, "columns.pkl"), "wb"))
pickle.dump(kmeans, open(os.path.join(models_path, "kmeans.pkl"), "wb"))

print("✅ Model saved successfully!")