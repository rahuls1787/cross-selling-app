import streamlit as st
import openai
import os
import pickle
import requests



# Fetch OpenAI API key from Streamlit Secrets
openai.api_base = "https://api.openai.com/v1"
openai.api_key = st.secrets["openai_api_key"]  # Access the secret key


# Streamlit app title
st.title('Cross-Selling Product Recommendation')

# Input fields for user data (same as your code)
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
    # Generate explanation prompt
    explanation_prompt = f"""
    The prediction for the insurance holder is 'Interested'.
    Here are the details of the input data:
    Gender: {gender}, Age: {age}, Driving License: {driving_license},
    Region Code: {region_code}, Previously Insured: {previously_insured},
    Vehicle Age: {vehicle_age}, Vehicle Damage: {vehicle_damage},
    Annual Premium: {annual_premium}, Policy Sales Channel: {policy_sales_channel},
    Vintage: {vintage}.

    Explain why the insurance holder is predicted to be 'Interested' based on the input data.
    """

    try:
        # Make the API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that explains predictions."},
                {"role": "user", "content": explanation_prompt}
            ]
        )
        # Access the content of the response correctly
        explanation = response['choices'][0]['message']['content'].strip()
        st.write('Explanation:', explanation)
    except Exception as e:
        st.write('Error generating explanation:', str(e))

