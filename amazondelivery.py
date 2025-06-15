import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

st.set_page_config(layout="wide")
st.title("🚚 Delivery Time Prediction & Analysis App")

# --- Base directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load Model & Scaler ---
model_path = os.path.join(BASE_DIR, 'random_forest_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# --- Sidebar Input Form ---
st.sidebar.header("📝 Enter Details for Prediction")
with st.sidebar.form("prediction_form"):
    distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, step=0.1)
    weather = st.selectbox("Weather", ["Sunny", "Stormy", "Cloudy", "Windy", "Fog"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
    vehicle = st.selectbox("Vehicle", ["Bike", "Scooter", "Electric Scooter"])
    area = st.selectbox("Area", ["Urban", "Semi-Urban", "Rural"])
    category = st.selectbox("Delivery Type", ["Snack Box", "Meal", "Drinks", "Grocery"])
    order_hour = st.slider("Order Hour", 0, 23)
    order_weekday = st.slider("Order Day (0=Mon, 6=Sun)", 0, 6)
    time_gap = st.number_input("Order to Pickup Gap (mins)", min_value=0.0)

    submitted = st.form_submit_button("Predict")

# --- Manual Encoding ---
weather_map = {'Sunny': 4, 'Stormy': 3, 'Cloudy': 1, 'Windy': 2, 'Fog': 0}
traffic_map = {'Low': 1, 'Medium': 2, 'High': 0, 'Jam': 3}
vehicle_map = {'Bike': 0, 'Scooter': 2, 'Electric Scooter': 1}
area_map = {'Urban': 2, 'Semi-Urban': 1, 'Rural': 0}
category_map = {'Snack Box': 3, 'Meal': 2, 'Drinks': 0, 'Grocery': 1}

# --- Make Prediction ---
if submitted:
    input_df = pd.DataFrame([[distance, agent_rating,
                              weather_map[weather], traffic_map[traffic], vehicle_map[vehicle],
                              area_map[area], category_map[category],
                              order_hour, order_weekday, time_gap]],
                            columns=['Distance_km', 'Agent_Rating', 'Weather', 'Traffic', 'Vehicle',
                                     'Area', 'Category', 'Order_Hour', 'Order_Weekday', 'Order_to_Pickup_Min'])
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.success(f"📦 Estimated Delivery Time: **{prediction:.2f} minutes**")

# --- Load Dataset for Visualization ---
@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, 'amazon_delivery.csv')
    return pd.read_csv(path)

df = load_data()

# --- Prepare for Visualization ---
df['Delivery_Time'] = df['Delivery_Time'].fillna(df['Delivery_Time'].median())

# Use original encoded columns if available, else map manually
df['Weather_Original'] = df['Weather'].replace({
    4: 'Sunny', 3: 'Stormy', 1: 'Cloudy', 2: 'Windy', 0: 'Fog'
}) if df['Weather'].dtype != 'O' else df['Weather']

df['Traffic_Original'] = df['Traffic'].replace({
    1: 'Low', 2: 'Medium', 0: 'High', 3: 'Jam'
}) if df['Traffic'].dtype != 'O' else df['Traffic']

# --- Visualizations ---
st.header("📊 Delivery Time Insights")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Delivery Time")
    fig1 = plt.figure(figsize=(6, 4))
    sns.histplot(df['Delivery_Time'], bins=30, kde=True, color='skyblue')
    plt.xlabel("Delivery Time (minutes)")
    plt.ylabel("Frequency")
    plt.grid(True)
    st.pyplot(fig1)

with col2:
    st.subheader("Delivery Time vs Agent Rating")
    fig2 = plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='Agent_Rating', y='Delivery_Time', alpha=0.5)
    plt.grid(True)
    st.pyplot(fig2)

st.subheader("Impact of Weather on Delivery Time")
fig3 = plt.figure(figsize=(10, 4))
sns.boxplot(data=df, x='Weather_Original', y='Delivery_Time')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(fig3)

st.subheader("Impact of Traffic on Delivery Time")
fig4 = plt.figure(figsize=(10, 4))
sns.boxplot(data=df, x='Traffic_Original', y='Delivery_Time')
plt.grid(True)
st.pyplot(fig4)
