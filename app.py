import streamlit as st
import openai
import os
import pickle
import requests
import gdown
import json
import pandas as pd
import numpy as np

# Set OpenAI API base and key from secrets
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = st.secrets["openai_api_key"]

# Streamlit app title
st.title('🚗 Cross-Selling Product Recommendation')

# Function to download files from Google Drive
def download_file_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)

# Download and load model files
download_file_from_gdrive('13H7zXyUJnBGLsRRpB3sfBJXlLRkyxkRM', 'ensemble_model.json')
download_file_from_gdrive('1hoPMuou0Bqb6Ki0ih4geuw7-1Xig6kmY', 'ensemble_model.pkl')

with open('ensemble_model.json', 'r') as f:
    model_config = json.load(f)

with open('ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# User inputs
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
driving_license = st.selectbox('Driving License', ['Yes', 'No'])
region_code = st.number_input('Region Code', min_value=1, max_value=100, value=10)
previously_insured = st.selectbox('Previously Insured', ['Yes', 'No'])
vehicle_age = st.selectbox('Vehicle Age', ['< 1 Year', '1-2 Years', '> 2 Years'])
vehicle_damage = st.selectbox('Vehicle Damage', ['Yes', 'No'])
annual_premium = st.number_input('Annual Premium', min_value=1000, max_value=100000, value=20000)
policy_sales_channel = st.number_input('Policy Sales Channel', min_value=1, max_value=200, value=10)
vintage = st.number_input('Vintage', min_value=1, max_value=500, value=100)

# Predict button
if st.button('Predict'):
    try:
        # Categorical encoding
        gender_map = {'Male': 1, 'Female': 0}
        yes_no_map = {'Yes': 1, 'No': 0}
        vehicle_age_map = {'< 1 Year': 0, '1-2 Years': 1, '> 2 Years': 2}

        input_data = pd.DataFrame([[
            gender_map[gender],
            age,
            yes_no_map[driving_license],
            region_code,
            yes_no_map[previously_insured],
            vehicle_age_map[vehicle_age],
            yes_no_map[vehicle_damage],
            annual_premium,
            policy_sales_channel,
            vintage
        ]], columns=model_config['features'])

        # Model prediction
        prediction = model.predict(input_data)[0]
        prediction_label = 'Interested' if prediction == 1 else 'Not Interested'

        st.markdown(f"### ✅ Prediction: **{prediction_label}**")

        # Explanation prompt for OpenAI
        explanation_prompt = f"""
        The prediction for the insurance holder is '{prediction_label}'.
        Here are the details of the input data:
        Gender: {gender}, Age: {age}, Driving License: {driving_license},
        Region Code: {region_code}, Previously Insured: {previously_insured},
        Vehicle Age: {vehicle_age}, Vehicle Damage: {vehicle_damage},
        Annual Premium: {annual_premium}, Policy Sales Channel: {policy_sales_channel},
        Vintage: {vintage}.

        Explain why the insurance holder is predicted to be '{prediction_label}' based on the input data.
        """

        # OpenAI call
        response = openai.ChatCompletion.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that explains predictions."},
                {"role": "user", "content": explanation_prompt}
            ]
        )
        explanation = response['choices'][0]['message']['content'].strip()
        st.markdown("### 🤖 Explanation from AI:")
        st.write(explanation)

    except Exception as e:
        st.error(f"Error during prediction or explanation: {str(e)}")
