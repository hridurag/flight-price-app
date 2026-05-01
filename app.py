
import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# 1. Load the trained model, scaler, and column names
# We add error handling to show a clear message if files are missing
try:
    model = joblib.load('flight_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('columns.pkl', 'rb') as f:
        model_columns = joblib.load(f)
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Please ensure you have uploaded your .pkl files to GitHub.")
    st.stop()

# 2. Define mappings
departure_arrival_mapping = {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4, 'Late_Night': 5}
stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
class_mapping = {'Economy': 0, 'Business': 1}

st.set_page_config(page_title="Flight Price Predictor", layout="wide")
st.title("✈️ Flight Price Prediction App")
st.markdown("---")

# 3. User Inputs
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        airline = st.selectbox("Airline", ['Indigo', 'AirAsia', 'Vistara', 'Air_India', 'GO_FIRST', 'SpiceJet'])
        source_city = st.selectbox("Source City", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad'])
        departure_time_input = st.selectbox("Departure Time", list(departure_arrival_mapping.keys()))
    
    with col2:
        destination_city = st.selectbox("Destination City", ['Mumbai', 'Bangalore', 'Kolkata', 'Delhi', 'Hyderabad', 'Chennai'])
        arrival_time_input = st.selectbox("Arrival Time", list(departure_arrival_mapping.keys()))
        stops_input = st.selectbox("Number of Stops", list(stops_mapping.keys()))
    
    with col3:
        flight_class_input = st.selectbox("Class", ['Economy', 'Business'])
        
        # --- DURATION AS TIME INPUT ---
        st.write("**Flight Duration**")
        dur_col1, dur_col2 = st.columns(2)
        with dur_col1:
            dur_hours = st.number_input("Hours", min_value=0, max_value=50, value=2)
        with dur_col2:
            dur_mins = st.number_input("Mins", min_value=0, max_value=59, value=30)
        
        duration_decimal = dur_hours + (dur_mins / 60.0)
        days_left = st.slider("Days Left until Flight", 1, 50, 20)

    submitted = st.form_submit_button("Predict Price")

# 4. Prediction Logic
if submitted:
    # Build the input dictionary
    input_data = {
        'departure_time': departure_arrival_mapping[departure_time_input],
        'stops': stops_mapping[stops_input],
        'arrival_time': departure_arrival_mapping[arrival_time_input],
        'class': class_mapping[flight_class_input],
        'duration': duration_decimal,
        'days_left': days_left
    }

    # Handle One-Hot Encoding for Airline and Cities
    for col in model_columns:
        if col not in input_data:
            if f"airline_{airline}" == col or f"source_city_{source_city}" == col or f"destination_city_{destination_city}" == col:
                input_data[col] = 1
            else:
                input_data[col] = 0

    # Convert to DataFrame and reorder columns to match training
    input_df = pd.DataFrame([input_data])[model_columns]

    # Scale the numerical features
    input_df[['duration', 'days_left']] = scaler.transform(input_df[['duration', 'days_left']])

    # Predict
    prediction = model.predict(input_df)[0]
    
    st.success(f"### Estimated Flight Price: ₹{prediction:,.2f}")
