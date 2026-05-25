import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("California House Price Predictor")

longitude = st.number_input("Longitude", -125.0, -114.0, -122.0)
latitude = st.number_input("Latitude", 32.0, 42.0, 37.0)

housing_median_age = st.number_input("House Age", 1, 52, 20)

total_rooms = st.number_input("Total Rooms", 100, 10000, 2000)
total_bedrooms = st.number_input("Total Bedrooms", 50, 5000, 500)

population = st.number_input("Population", 100, 10000, 1000)
households = st.number_input("Households", 50, 5000, 300)

median_income = st.number_input("Median Income", 0.0, 15.0, 5.0)

# engineered features
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households

# categorical (simple checkbox style)
inland = st.checkbox("INLAND")
island = st.checkbox("ISLAND")
near_bay = st.checkbox("NEAR BAY")
near_ocean = st.checkbox("NEAR OCEAN")

if st.button("Predict"):
    data = np.array([[
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        rooms_per_household,
        bedrooms_per_room,
        population_per_household,
        inland,
        island,
        near_bay,
        near_ocean
    ]])

    prediction = model.predict(data)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
