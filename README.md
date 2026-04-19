# 🌱 Crop Recommendation System

A machine learning-based decision support system that recommends the **top 3 most suitable crops** for farmers based on their farm conditions.

---

## 🚀 Project Overview

This project leverages historical agricultural data and machine learning techniques to assist farmers in making informed crop selection decisions. Instead of predicting a single crop, the system provides **top-3 crop recommendations with confidence scores**, making it more practical for real-world use.

---

## 🎯 Key Features

* ✅ Top-3 crop recommendations with confidence scores
* ✅ Advanced feature engineering (per-hectare normalization, interaction features)
* ✅ XGBoost optimized model
* ✅ Clustering-based enhancement (KMeans)
* ✅ Multilingual Streamlit interface 🌐
* ✅ Clean and modular ML pipeline

---

## 🧠 Machine Learning Approach

* Data preprocessing and cleaning
* Feature engineering:

  * Fertilizer per hectare
  * Pesticide per hectare
  * Rainfall interactions
  * Log transformations
* One-hot encoding for categorical variables
* Model selection:

  * Random Forest
  * XGBoost (final selected model)
* Clustering using KMeans for pattern extraction
* Evaluation using:

  * Accuracy
  * Top-3 Accuracy (primary metric)

---

## 📊 Model Performance

* **Accuracy:** ~56–60%
* **Top-3 Accuracy:** ~85%

> Top-3 accuracy is prioritized to provide practical recommendations rather than a single rigid prediction.

---

## 🏗️ Project Structure

```
crop-recommendation-system/
│
├── app/                  # Streamlit application
├── src/                  # ML pipeline (train, preprocess, predict)
├── notebooks/            # Experimentation & model tuning
├── data/                 # Dataset (optional/sample)
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* XGBoost
* Pandas & NumPy
* Streamlit
* KMeans Clustering

---

## ▶️ How to Run Locally

### 1. Clone the repository

```
git clone https://github.com/your-username/crop-recommendation-system.git
cd crop-recommendation-system
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Train the model

```
python -m src.train
```

### 4. Run the application

```
streamlit run app/streamlit_app.py
```

---

## 🌐 Future Enhancements

* Soil data integration 🌱
* Real-time weather API 🌦️
* District-level recommendations 📍
* SHAP explainability 📊

---

## 💡 Motivation

This project aims to bridge the gap between **data science and agriculture**, providing farmers with actionable insights to improve crop selection and productivity.

---

## 👨‍💻 Author

Shrestha Swami
Riya Dhaked
B.Tech Student | Data Science Enthusiast



