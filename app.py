import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(layout="wide")
# Custom UI Styling
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}
/* Title */
h1 {
    text-align: center;
    color: #2c3e50;
}
/* Sliders - default green */
.stSlider > div > div > div > div {
    background-color: #4CAF50 !important;
}
.stSlider:nth-of-type(2n) > div > div > div > div {
    background-color: #2196F3 !important;
}
.stSlider:nth-of-type(3n) > div > div > div > div {
    background-color: #FF9800 !important;
}
/* Button */
.stButton > button {
    background-color: #007BFF;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 50px;
}
.stButton > button:hover {
    background-color: #0056b3;
}
/* Inputs */
.stNumberInput input {
    border-radius: 8px;
}

.stSelectbox div {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("xgboost_model.pkl")
# Title
st.markdown("<h1>Airline Customer Satisfaction Prediction</h1>", unsafe_allow_html=True)
st.markdown("### Enter Customer Details")
col1, col2, col3, col4 = st.columns(4)
with col1:
    gender = st.radio("Gender", ["Male", "Female"])
    customer_type = st.radio("Customer Type", ["Loyal Customer", "disloyal Customer"])
    travel_type = st.radio("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    age = st.slider("Age", 0, 100, 25)
    flight_distance = st.number_input("Flight Distance", 0, 5000, 500)
with col2:
    seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
    dep_arr_time = st.slider("Departure/Arrival Time", 0, 5, 3)
    food_drink = st.slider("Food and Drink", 0, 5, 3)
    gate_location = st.slider("Gate Location", 0, 5, 3)
    wifi = st.slider("Inflight WiFi", 0, 5, 3)
    entertainment = st.slider("Entertainment", 0, 5, 3)
with col3:
    online_support = st.slider("Online Support", 0, 5, 3)
    ease_booking = st.slider("Ease of Booking", 0, 5, 3)
    onboard_service = st.slider("On-board Service", 0, 5, 3)
    legroom = st.slider("Leg Room", 0, 5, 3)
    baggage = st.slider("Baggage Handling", 0, 5, 3)
    checkin = st.slider("Check-in Service", 0, 5, 3)
with col4:
    cleanliness = st.slider("Cleanliness", 0, 5, 3)
    online_boarding = st.slider("Online Boarding", 0, 5, 3)
    dep_delay = st.number_input("Departure Delay", 0, 1000, 0)
    arr_delay = st.number_input("Arrival Delay", 0, 1000, 0)
st.markdown("<br>", unsafe_allow_html=True)
center_col = st.columns([1,2,1])
with center_col[1]:
    predict = st.button("Predict Satisfaction", use_container_width=True)
# Prediction
if predict:
    input_data = {
        'Gender': 1 if gender == "Male" else 0,
        'Customer Type': 1 if customer_type == "Loyal Customer" else 0,
        'Type of Travel': 1 if travel_type == "Personal Travel" else 0,
        'Class': {"Eco": 0, "Eco Plus": 1, "Business": 2}[travel_class],
        'Age': age,
        'Flight Distance': flight_distance,
        'Seat comfort': seat_comfort,
        'Departure/Arrival time convenient': dep_arr_time,
        'Food and drink': food_drink,
        'Gate location': gate_location,
        'Inflight wifi service': wifi,
        'Inflight entertainment': entertainment,
        'Online support': online_support,
        'Ease of Online booking': ease_booking,
        'On-board service': onboard_service,
        'Leg room service': legroom,
        'Baggage handling': baggage,
        'Checkin service': checkin,
        'Cleanliness': cleanliness,
        'Online boarding': online_boarding,
        'Departure Delay in Minutes': dep_delay,
        'Arrival Delay in Minutes': arr_delay
    }
    input_df = pd.DataFrame([input_data])
    input_df = input_df[model.feature_names_in_]
    prediction = model.predict(input_df)
    result = "Satisfied" if prediction[0] == 1 else "Not Satisfied"
    st.success(f"Prediction: {result}")

    