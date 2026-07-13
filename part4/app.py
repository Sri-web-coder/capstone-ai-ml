import os
import json
import joblib
import requests
import jsonschema
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Import the guardrail function from your other file
from guardrails import has_pii

# Setup API & Environment
load_dotenv()
API_KEY = os.environ.get("LLM_API_KEY") 

# JSON Schema & System Prompt
expected_schema = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string"},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"]
}

fallback_json = {
    "prediction_label": None,
    "confidence_level": None,
    "top_reason": None,
    "second_reason": None,
    "next_step": None
}

SYSTEM_PROMPT = """You are a machine learning interpretation assistant. 
Your task is to explain model predictions based on feature values, predicted class, and probability.
Output ONLY valid JSON matching this exact schema:
{
    "prediction_label": "string (e.g., At Risk, Not At Risk)",
    "confidence_level": "string (low, medium, high)",
    "top_reason": "string (main reason for prediction based on features)",
    "second_reason": "string (secondary reason)",
    "next_step": "string (recommended action)"
}
Do not include any markdown tags like ```json or any other text. Return pure JSON."""

# Helper Functions
def call_llm(system_prompt, user_prompt, temperature=0.0):
    if has_pii(user_prompt):
        return {"error": "Input blocked: PII detected."}

    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-4o-mini", 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": 512
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return {"error": f"API Error {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def encode_record(features, model):
    df_features = pd.DataFrame([features])

    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)

        for col in expected_cols:
            if col not in df_features.columns:
                df_features[col] = 0

        df_features = df_features[expected_cols]

    return df_features

# Streamlit UI
st.set_page_config(page_title="Student Mental Health Predictor", layout="wide")
st.title("Student Mental Health Predictor & AI Analyzer")
st.markdown("Enter the student's details below to get a prediction and an AI-generated analysis.")

# Load Model
@st.cache_resource
def load_model():
    try:
        return joblib.load('best_model.pkl') 
    except Exception:
        return None

model = load_model()

if model is None:
    st.error("Error: 'best_model.pkl' not found. Please ensure the model file is in the same directory.")
else:
    # User Input Form
    with st.form("input_form"):
        st.subheader("Student Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("Age", min_value=15, max_value=30, value=21, step=1)
            year = st.number_input("Year of Study", min_value=1, max_value=5, value=3, step=1)
            study_hrs = st.number_input("Study Hours/Week", min_value=0.0, max_value=50.0, value=15.0)
            sleep_hrs = st.number_input("Sleep Hours/Night", min_value=0.0, max_value=12.0, value=7.0)
            
        with col2:
            social_hrs = st.number_input("Social Media Hrs/Day", min_value=0.0, max_value=15.0, value=5.0)
            screen_hrs = st.number_input("Screen Time Hrs/Day", min_value=0.0, max_value=18.0, value=6.5)
            physical_act = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
            activity_mapping = {"Low": 0, "Moderate": 1, "High": 2}
            physical_act_encoded= activity_mapping[physical_act]
            diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
            diet_mapping = {"Poor": 0, "Average": 1, "Good": 2}
            diet_encoded = diet_mapping[diet]
            
        with col3:
            financial = st.slider("Financial Stress (0-10))", 0.0, 10.0, 5.0, 0.1)
            family = st.slider("Family Support (0-10)", 0.0, 10.0, 5.0, 0.1)
            attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=90.0)

        with col4:
            gender_male_input = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            gender_male = 1.0 if gender_male_input == "Male" else 0.0
            gender_nb = 1.0 if gender_male_input == "Non-binary" else 0.0
            job_input = st.selectbox("Part-Time Job?", ["No", "Yes"])
            job = 1.0 if job_input == "Yes" else 0.0
            history = st.selectbox("Prev. Mental Health History?", ["No", "Yes"])
            history = 1.0 if history == "Yes" else 0.0

        submit_button = st.form_submit_button(label="Predict & Explain", type="primary")

    # Processing the input
    if submit_button:
        features = {
            "Age": age, "Year_of_Study": year, "Study_Hours_Per_Week": study_hrs, 
            "Sleep_Hours": sleep_hrs, "Social_Media_Hours": social_hrs, "Screen_Time_Hours": screen_hrs,
            "Physical_Activity": physical_act_encoded, "Financial_Stress": financial, "Family_Support": family, 
            "Attendance_Rate": attendance, "Diet_Quality": diet_encoded,
            "Gender_Male": gender_male, "Gender_Non-binary": gender_nb, "Part_Time_Job_Yes": job, 
            "Previous_Mental_Health_History_Yes": history
        }
        
        with st.spinner('Analyzing data and generating explanation...'):
            df_features = encode_record(features, model)
            pred_class = int(model.predict(df_features)[0])
            pred_proba = float(model.predict_proba(df_features)[0][1])
            
            st.markdown("---")
            st.subheader("Model Output")
            col_a, col_b = st.columns(2)
            label = "AT RISK" if pred_class == 1 else "NOT at RISK"
            col_a.markdown("### Predicted Class")
            col_a.markdown(f"# **{label}**")
            col_b.markdown("### Probability of Risk")
            col_b.markdown(f"# **{pred_proba * 100:.2f}%**")
            
            user_prompt = f"Features: {features} | Predicted Class: {label} | Probability: {pred_proba:.4f}"
            raw_response = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)
            
            st.subheader("AI Analysis")
            if isinstance(raw_response, dict) and "error" in raw_response:
                st.error(raw_response["error"])
            else:
                clean_response = raw_response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                elif clean_response.startswith("```"):
                    clean_response = clean_response[3:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                try:
                    parsed_json = json.loads(clean_response)
                    jsonschema.validate(instance=parsed_json, schema=expected_schema)
                    st.info(f"**Prediction Result:**  {parsed_json['prediction_label'].upper()}")
                    st.write(f"**Confidence Level:**  {parsed_json['confidence_level']}")
                    st.write(f"**Primary Reason:**  {parsed_json['top_reason']}")
                    st.write(f"**Secondary Reason:**  {parsed_json['second_reason']}")
                    st.success(f"**Recommended Next Step:**  {parsed_json['next_step']}")
                    
                except json.JSONDecodeError:
                    st.error("JSON Decode Error: Could not parse the LLM response.")
                except jsonschema.ValidationError as e:
                    st.error(f"Schema Validation Error: {e.message}")

