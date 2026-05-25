import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")

st.set_page_config(page_title="House Price Predictor", layout="wide")

# ================= HEADER =================
st.title("🏠 California House Price Prediction Dashboard")
st.markdown("### AI-powered Real Estate Price Estimator")

# ================= SIDEBAR INPUT =================
st.sidebar.header("📌 Enter House Details")

longitude = st.sidebar.number_input("Longitude", -125.0, -114.0, -122.0)
latitude = st.sidebar.number_input("Latitude", 32.0, 42.0, 37.0)

housing_median_age = st.sidebar.number_input("House Age", 1, 52, 20)
total_rooms = st.sidebar.number_input("Total Rooms", 100, 10000, 2000)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", 50, 5000, 500)

population = st.sidebar.number_input("Population", 100, 10000, 1000)
households = st.sidebar.number_input("Households", 50, 5000, 300)

median_income = st.sidebar.number_input("Median Income", 0.0, 15.0, 5.0)

# ================= FEATURE ENGINEERING =================
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households

inland = st.sidebar.checkbox("INLAND")
island = st.sidebar.checkbox("ISLAND")
near_bay = st.sidebar.checkbox("NEAR BAY")
near_ocean = st.sidebar.checkbox("NEAR OCEAN")

# ================= PREDICTION =================
if st.sidebar.button("🔮 Predict Price"):

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

    prediction = model.predict(data)[0]

    # ================= CARDS =================
    col1, col2, col3 = st.columns(3)

    col1.metric("📍 Location", f"{latitude:.2f}, {longitude:.2f}")
    col2.metric("💰 Predicted Price", f"${prediction:,.0f}")
    col3.metric("🏠 Rooms", f"{total_rooms}")

    st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")

    # ================= CHART SECTION =================
    st.subheader("📊 Feature Overview")

    features = {
        "Rooms/Household": rooms_per_household,
        "Bedrooms/Room": bedrooms_per_room,
        "Population/Household": population_per_household,
        "Income": median_income
    }

    df_chart = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])

    fig, ax = plt.subplots()
    ax.bar(df_chart["Feature"], df_chart["Value"])
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ================= INFO =================
    st.info("Model trained using Linear Regression on California Housing dataset")
