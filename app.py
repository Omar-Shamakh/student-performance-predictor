import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set up the app
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("Student Math Score Prediction")
st.write("This app predicts a student's math score based on various factors")

# --- Input Widgets ---
st.sidebar.header("Student Information")

# Define categorical options (must match training data exactly)
gender_options = ['female', 'male']
ethnicity_options = ['group A', 'group B', 'group C', 'group D', 'group E']
parental_education_options = ["some high school", "high school", "some college", 
                            "associate's degree", "bachelor's degree", "master's degree"]
lunch_options = ['standard', 'free/reduced']
test_prep_options = ['none', 'completed']

# Create input widgets
gender = st.sidebar.selectbox("Gender", gender_options)
ethnicity = st.sidebar.selectbox("Race/Ethnicity", ethnicity_options)
parental_education = st.sidebar.selectbox("Parental Education Level", parental_education_options)
lunch = st.sidebar.selectbox("Lunch Type", lunch_options)
test_prep = st.sidebar.selectbox("Test Preparation Course", test_prep_options)
reading_score = st.sidebar.slider("Reading Score", 0, 100, 70)
writing_score = st.sidebar.slider("Writing Score", 0, 100, 70)

# --- Prediction Function ---
def predict_math_score():
    try:
        # Load the full pipeline
        pipeline = joblib.load('pipeline.pkl')
        
        # Create input DataFrame (column names must match training data)
        input_data = pd.DataFrame({
            'gender': [gender],
            'race/ethnicity': [ethnicity],
            'parental level of education': [parental_education],
            'lunch': [lunch],
            'test preparation course': [test_prep],
            'reading score': [reading_score],
            'writing score': [writing_score]
        })
        
        # Make prediction (pipeline handles all preprocessing)
        prediction = pipeline.predict(input_data)[0]
        st.success(f"Predicted Math Score: {prediction:.1f}")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# --- Run Prediction ---
if st.sidebar.button("Predict Math Score"):
    predict_math_score()

# --- Explanatory Sections ---
st.header("How It Works")
st.markdown("""
This model predicts math scores using:
- **Demographics**: Gender, ethnicity
- **Background**: Parental education, lunch type
- **Scores**: Reading and writing marks
- **Preparation**: Test prep course status
""")

st.header("Key Insights")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Performance by Gender")
    st.markdown("""
    - 🚺 Females score higher in reading/writing
    - 🚹 Males score higher in math
    """)

with col2:
    st.subheader("Lunch Impact")
    st.markdown("""
    - 🍎 Standard lunch → Higher scores
    - 🥗 Free/reduced → Slightly lower scores
    """)

# --- Batch Prediction Section ---
st.header("Batch Predictions")
uploaded_file = st.file_uploader("Upload CSV for multiple predictions", type=["csv"])

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        pipeline = joblib.load('pipeline.pkl')
        
        # Check required columns
        required_cols = ['gender', 'race/ethnicity', 'parental level of education',
                       'lunch', 'test preparation course', 'reading score', 'writing score']
        
        if all(col in batch_data.columns for col in required_cols):
            predictions = pipeline.predict(batch_data)
            batch_data['predicted_math_score'] = predictions.round(1)
            
            st.dataframe(batch_data)
            
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions",
                data=csv,
                file_name='math_score_predictions.csv',
                mime='text/csv'
            )
        else:
            missing = [col for col in required_cols if col not in batch_data.columns]
            st.error(f"Missing columns: {', '.join(missing)}")
            
    except Exception as e:
        st.error(f"Batch prediction failed: {str(e)}")
