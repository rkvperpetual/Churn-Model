import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.title("Salary Prediction App")

# Load the models and encoders
@st.cache_resource
def load_resources():
    model = load_model("churn_model_regression.h5")  # Renamed to reflect its purpose
    scaler = pickle.load(open('regression_scaler.pkl', 'rb'))
    label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
    onehot_encoder_geography = pickle.load(open('onehot_encoder_geography.pkl', 'rb'))
    return model, scaler, label_encoder_gender, onehot_encoder_geography

model, scaler, label_encoder_gender, onehot_encoder_geography = load_resources()

# User Input Form
customer_data = {
    'CreditScore': st.number_input('Credit Score', min_value=300, max_value=850, step=1),
    'Geography': st.selectbox('Geography', ['France', 'Spain', 'Germany']),
    'Gender': st.selectbox('Gender', ['Male', 'Female']),
    'Age': st.slider('Age', 18, 80),
    'Tenure': st.slider('Tenure (Years)', 0, 10),
    'Balance': st.number_input('Balance', min_value=0.0, format="%.2f"),
    'NumOfProducts': st.number_input('Number of Products', min_value=1, max_value=4, step=1),
    'HasCrCard': st.selectbox('Has Credit Card', ['Yes', 'No']),
    'IsActiveMember': st.selectbox('Active Member', ['Yes', 'No']),
    'Exited': st.selectbox('Exited', ['Yes', 'No'])
}

# Create a DataFrame from the user input
customer_df = pd.DataFrame(customer_data, index=[0])

# Add Predict Button
if st.button('Predict'):
    # Encode Gender
    customer_df['Gender'] = label_encoder_gender.transform(customer_df['Gender'])

    # Encode Geography
    geo_encoded = onehot_encoder_geography.transform([[customer_df['Geography'][0]]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

    # Drop Geography and concatenate encoded columns
    customer_df = pd.concat([customer_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

    # Encode HasCrCard, IsActiveMember, and Exited
    customer_df['HasCrCard'] = customer_df['HasCrCard'].apply(lambda x: 1 if x == 'Yes' else 0)
    customer_df['IsActiveMember'] = customer_df['IsActiveMember'].apply(lambda x: 1 if x == 'Yes' else 0)
    customer_df['Exited'] = customer_df['Exited'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Scale the data
    customer_data_scaled = scaler.transform(customer_df)

    # Make the prediction
    prediction = model.predict(customer_data_scaled)

    # Display the predicted salary
    st.write(f"### Predicted Salary: ${prediction[0][0]:,.2f}")
