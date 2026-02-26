# Name: Raushan Kumar

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

st.title("🚢 Titanic Survival Prediction App")
st.markdown("### Developed by Raushan Kumar")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")
    return df

df = load_data()

# ------------------ DATA PREPROCESSING ------------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df = df.dropna(subset=['Embarked'])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
        'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# ------------------ TRAIN MODEL ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ------------------ METRICS SECTION ------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.write("Classification Report")
col2.text(classification_report(y_test, y_pred))

# ------------------ CONFUSION MATRIX ------------------
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)

# ------------------ SIDEBAR PREDICTION ------------------
st.sidebar.header("🧍 Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
age = st.sidebar.slider("Age", 1, 80, 25)
sibsp = st.sidebar.slider("Siblings/Spouses", 0, 5, 0)
parch = st.sidebar.slider("Parents/Children", 0, 5, 0)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 50.0)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
embarked = st.sidebar.selectbox("Embarked", ["Q", "S", "Other"])

# Convert inputs to model format
sex_male = 1 if sex == "Male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = pd.DataFrame([[
    pclass, age, sibsp, parch, fare,
    sex_male, embarked_Q, embarked_S
]], columns=X.columns)

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# ------------------ PREDICTION RESULT ------------------
st.subheader("🎯 Survival Prediction")

if prediction == 1:
    st.success("The passenger is likely to SURVIVE 🎉")
else:
    st.error("The passenger is NOT likely to survive 💔")
