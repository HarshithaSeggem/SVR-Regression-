# app.py

import streamlit as st
import pandas as pd
import pickle

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TITLE
st.title("Support Vector Regression Application")

# LOAD DATASET
data = load_diabetes()

# DATAFRAME
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# TARGET
df["target"] = data.target

# SHOW DATA
st.subheader("Dataset")
st.write(df.head())

# FEATURES AND TARGET
X = df.drop("target", axis=1)
y = df["target"]

# SCALING
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = SVR(
    kernel="rbf",
    C=100,
    gamma=0.1,
    epsilon=0.1
)

# TRAIN MODEL
model.fit(X_train, y_train)

# SAVE MODEL
pickle.dump(
    model,
    open("svr_model.pkl", "wb")
)

# SAVE SCALER
pickle.dump(
    scaler,
    open("scaler.pkl", "wb")
)

# PREDICTIONS
y_pred = model.predict(X_test)

# EVALUATION
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# SHOW METRICS
st.subheader("Model Evaluation")

st.write("MAE :", mae)
st.write("MSE :", mse)
st.write("R2 Score :", r2)

# USER INPUT
st.subheader("Predict Disease Progression")

age = st.number_input("Age", value=0.05)
sex = st.number_input("Sex", value=0.05)
bmi = st.number_input("BMI", value=0.05)
bp = st.number_input("Blood Pressure", value=0.05)
s1 = st.number_input("S1", value=0.05)
s2 = st.number_input("S2", value=0.05)
s3 = st.number_input("S3", value=0.05)
s4 = st.number_input("S4", value=0.05)
s5 = st.number_input("S5", value=0.05)
s6 = st.number_input("S6", value=0.05)

# INPUT DATA
new_data = [[
    age,
    sex,
    bmi,
    bp,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6
]]

# SCALE INPUT
new_data_scaled = scaler.transform(new_data)

# PREDICT
if st.button("Predict"):

    prediction = model.predict(
        new_data_scaled
    )

    st.success(
        f"Predicted Value : {prediction[0]:.2f}"
    )