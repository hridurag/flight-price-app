import streamlit as st
import pandas as pd
import joblib
import pickle

# Load the trained model, scaler, and column names
model = joblib.load('flight_price_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# Define mappings for categorical features (consistent with notebook preprocessing)
departure_arrival_mapping = {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4, 'Late_Night': 5}
stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
class_mapping = {'Economy': 0, 'Business': 1}

st.set_page_config(layout="wide")
st.title("Flight Price Prediction App")

st.markdown("---")
st.header("Enter Flight Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox("Airline", ['Indigo', 'AirAsia', 'Vistara', 'Air_India', 'GO_FIRST', 'SpiceJet'])
        source_city = st.selectbox("Source City", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad'])
        departure_time_input = st.selectbox("Departure Time", ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])

    with col2:
        destination_city = st.selectbox("Destination City", ['Mumbai', 'Bangalore', 'Kolkata', 'Delhi', 'Hyderabad', 'Chennai'])
        arrival_time_input = st.selectbox("Arrival Time", ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
        stops_input = st.selectbox("Number of Stops", ['zero', 'one', 'two_or_more'])

    with col3:
        flight_class_input = st.selectbox("Class", ['Economy', 'Business'])
        duration = st.number_input("Duration (in hours)", min_value=0.5, max_value=50.0, value=5.0, step=0.1)
        days_left = st.slider("Days Left until Departure", min_value=1, max_value=50, value=30)

    submitted = st.form_submit_button("Predict Flight Price")

    if submitted:
        # Prepare new data for prediction
        new_data_dict = {
            'departure_time': departure_arrival_mapping[departure_time_input],
            'stops': stops_mapping[stops_input],
            'arrival_time': departure_arrival_mapping[arrival_time_input],
            'class': class_mapping[flight_class_input],
            'duration': duration,
            'days_left': days_left,
            'airline_AirAsia': 0, 'airline_Air_India': 0, 'airline_GO_FIRST': 0, 
            'airline_Indigo': 0, 'airline_SpiceJet': 0, 'airline_Vistara': 0,
            'source_city_Bangalore': 0, 'source_city_Chennai': 0, 'source_city_Delhi': 0, 
            'source_city_Hyderabad': 0, 'source_city_Kolkata': 0, 'source_city_Mumbai': 0,
            'destination_city_Bangalore': 0, 'destination_city_Chennai': 0, 'destination_city_Delhi': 0,
            'destination_city_Hyderabad': 0, 'destination_city_Kolkata': 0, 'destination_city_Mumbai': 0
        }

        # Set specific one-hot encoded columns based on user input
        new_data_dict[f'airline_{airline}'] = 1
        new_data_dict[f'source_city_{source_city}'] = 1
        new_data_dict[f'destination_city_{destination_city}'] = 1
        
        new_df = pd.DataFrame([new_data_dict])

        # Reindex to ensure all columns from training data are present and in correct order
        new_df = new_df.reindex(columns=model_columns, fill_value=0)

        # Scale numerical features (duration, days_left)
        new_df[['duration', 'days_left']] = scaler.transform(new_df[['duration', 'days_left']])

        # Make prediction
        prediction = model.predict(new_df)
        st.success(f"The predicted flight price is: ₹{prediction[0]:,.2f}")

st.markdown("--- ")
st.markdown("**Note:** This is a predictive model. Actual prices may vary.")
