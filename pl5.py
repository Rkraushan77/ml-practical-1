# Name: Raushan Kumar

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

st.title("🩺 Diabetes Progression Prediction using Linear Regression")
st.markdown("### Developed by Raushan Kumar")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes_prediction_dataset.csv")

diabetes = load_data()

st.subheader("Dataset Preview")
st.dataframe(diabetes.head())

# ---------------- DATA PREPROCESSING ----------------
# Change this if your target column name is different
target_column = "diabetes"

if target_column not in diabetes.columns:
    st.error(f"Target column '{target_column}' not found in dataset.")
    st.stop()

X = diabetes.drop(target_column, axis=1)
y = diabetes[target_column]

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL TRAINING ----------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.2f}")
col2.metric("R² Score", f"{r2:.2f}")

# ---------------- VISUALIZATION ----------------
st.subheader("📈 True vs Predicted Values")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
ax.set_xlabel("True Values")
ax.set_ylabel("Predicted Values")
ax.set_title("True vs Predicted")
ax.grid(True)

st.pyplot(fig)

st.success("Model trained and deployed successfully 🚀")
