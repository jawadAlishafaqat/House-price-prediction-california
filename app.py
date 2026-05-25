import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("California House Price Predictor")

income = st.slider("Income", 0.0, 15.0, 5.0)
age = st.slider("House Age", 1, 50, 20)
rooms = st.number_input("Rooms", 100, 10000, 2000)

if st.button("Predict"):
    data = np.array([[income, age, rooms]])
    prediction = model.predict(data)
    st.success(prediction[0])
