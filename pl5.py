# Name: Raushan Kumar
# Project: Improved Diabetes Prediction using Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# -------------------------------
# STEP 2: Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 3: Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# STEP 4: Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# STEP 5: Evaluation Metrics
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== Model Performance ==========")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-Squared Score (R2): {r2:.2f}")
print("=======================================\n")

# -------------------------------
# STEP 6: Visualization
# -------------------------------
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1️⃣ True vs Predicted Plot
axes[0].scatter(y_test, y_pred, color="royalblue", alpha=0.7)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=2)
axes[0].set_title("True vs Predicted Values", fontsize=14)
axes[0].set_xlabel("True Values")
axes[0].set_ylabel("Predicted Values")

# 2️⃣ BMI Feature vs Prediction
axes[1].scatter(X_test[:, 2], y_pred,
                color="green", alpha=0.7)
axes[1].set_title("BMI vs Predicted Diabetes Progression", fontsize=14)
axes[1].set_xlabel("BMI (Feature 2)")
axes[1].set_ylabel("Predicted Progression")

# 3️⃣ Residual Plot
residuals = y_test - y_pred
axes[2].scatter(y_pred, residuals,
                color="purple", alpha=0.7)
axes[2].axhline(y=0, color='red', linestyle='--')
axes[2].set_title("Residual Plot", fontsize=14)
axes[2].set_xlabel("Predicted Values")
axes[2].set_ylabel("Residuals")

plt.suptitle("Improved Linear Regression Analysis - Diabetes Dataset",
             fontsize=18, fontweight='bold')

plt.tight_layout()
plt.show()
