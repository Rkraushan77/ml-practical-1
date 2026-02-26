# Name: Raushan Kumar

import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

st.title("🩺 Diabetes Progression Prediction using Linear Regression")
st.markdown("### Developed by Raushan Kumar")

# Load dataset
diabetes = pd.read_csv("diabetes_prediction_dataset.csv")
X = diabetes.data
y = diabetes.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.2f}")
col2.metric("R² Score", f"{r2:.2f}")

st.markdown("---")

# Create plots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: True vs Predicted
axs[0].scatter(y_test, y_pred, alpha=0.6)
axs[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
axs[0].set_title("True vs Predicted Values")
axs[0].set_xlabel("True Values")
axs[0].set_ylabel("Predicted Values")
axs[0].grid(True)

# Plot 2: BMI Feature vs Predicted
axs[1].scatter(X_test[:, 2], y_pred, alpha=0.6)
axs[1].set_title("BMI vs Predicted Values")
axs[1].set_xlabel("BMI (Feature 2)")
axs[1].set_ylabel("Predicted Diabetes Progression")
axs[1].grid(True)

st.pyplot(fig)

st.success("Model Training and Visualization Completed Successfully!")
