# Part 4 — LLM-Powered Feature: Model Prediction Explanation

import os
import json
import re
import requests
import joblib
import pandas as pd
import jsonschema
from dotenv import load_dotenv

# Setup API & Environment
load_dotenv()
API_KEY = os.environ.get("LLM_API_KEY")

if not API_KEY:
    print("Warning: LLM_API_KEY environment variable not found!")

# Helper Functions
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
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
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Request Exception: {e}")
        return None

def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, str(text)) or re.search(phone_pattern, str(text)))

def encode_record(features, model):
    df_features = pd.DataFrame([features])
    if hasattr(model, 'feature_names_in_'):
        expected_cols = list(model.feature_names_in_)
        # Add missing columns with 0
        for col in expected_cols:
            if col not in df_features.columns:
                df_features[col] = 0
        df_features = df_features[expected_cols]
    return df_features

# JSON Schema & Prompts
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

print("==================================================")
print("TEST 1: Basic API Connection Demonstration")
print("==================================================")
test_response = call_llm(SYSTEM_PROMPT, "Reply with only the word: hello", temperature=0.0)
print(f"API Test Output: {test_response}\n")

print("==================================================")
print("TEST 2: Guardrails (PII Detection) Demonstration")
print("==================================================")
clean_input = "Features: {'Age': 21, 'GPA': 3.5} | Predicted Class: 0"
pii_input = "Student durga@email.com has Features: {'Age': 21} | Predicted Class: 1"

for i, test_input in enumerate([clean_input, pii_input], 1):
    print(f"Testing Input {i}: {test_input}")
    if has_pii(test_input):
        print("-> Result: Input blocked: PII detected.\n")
    else:
        print("-> Result: Clean input. Proceeding to LLM...\n")

print("==================================================")
print("TEST 3 & 4: End-to-End Pipeline & Temp A/B Comparison")
print("==================================================")

try:
    model = joblib.load('best_model.pkl')
    print("Model 'best_model.pkl' loaded successfully.\n")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Hand-crafted feature vectors based on Part 1/2 dataset
inputs = [
    {
        "Age": 22, "Year_of_Study": 4, "Study_Hours_Per_Week": 10, "Sleep_Hours": 4.5, 
        "Social_Media_Hours": 8, "Screen_Time_Hours": 10, "Physical_Activity": 0, 
        "Financial_Stress": 9, "Family_Support": 3, "Attendance_Rate": 65, 
        "Diet_Quality": 0, "Gender_Male": 1, 
        "Gender_Non-binary": 0, "Part_Time_Job_Yes": 1, "Previous_Mental_Health_History_Yes": 1
    },
    {
        "Age": 20, "Year_of_Study": 2, "Study_Hours_Per_Week": 25, "Sleep_Hours": 8.0, 
        "Social_Media_Hours": 2, "Screen_Time_Hours": 4, "Physical_Activity": 2, 
        "Financial_Stress": 2, "Family_Support": 8, "Attendance_Rate": 95, 
        "Diet_Quality": 2, "Gender_Male": 0, 
        "Gender_Non-binary": 0, "Part_Time_Job_Yes": 0, "Previous_Mental_Health_History_Yes": 0
    },
    {
        "Age": 21, "Year_of_Study": 3, "Study_Hours_Per_Week": 15, "Sleep_Hours": 6.5, 
        "Social_Media_Hours": 5, "Screen_Time_Hours": 6, "Physical_Activity": 1, 
        "Financial_Stress": 5, "Family_Support": 5, "Attendance_Rate": 80, 
        "Diet_Quality": 1, "Gender_Male": 1, 
        "Gender_Non-binary": 0, "Part_Time_Job_Yes": 1, "Previous_Mental_Health_History_Yes": 0
    }
]

for idx, features in enumerate(inputs, 1):
    print(f"\n--- Processing Record {idx} ---")
    
    # Preprocess and Predict
    df_features = encode_record(features, model)
    pred_class = int(model.predict(df_features)[0])
    pred_proba = float(model.predict_proba(df_features)[0][1])
    class_mapping = {0: "Not At Risk", 1: "At Risk"} 
    readable_class = class_mapping.get(pred_class, f"Unknown ({pred_class})")
    user_prompt = f"Features: {features} | Predicted Class: {readable_class} | Probability: {pred_proba:.2%}"
    print(f"Feature Input: {user_prompt[:100]}...") 
    print(f"Predicted Class: {readable_class} ({pred_class}) | Probability: {pred_proba:.2%}")

    # Guardrail Check
    if has_pii(user_prompt):
        print("Guardrail Status: Blocked (PII Detected)")
        continue
    else:
        print("Guardrail Status: Passed")
    
    # Temperature 0.0 Call (Deterministic)
    print("\nCalling LLM with Temperature = 0.0 (For Pipeline)...")
    raw_response_0 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    
    # Temperature 0.7 Call (Variability for A/B Comparison)
    print("Calling LLM with Temperature = 0.7 (For A/B Comparison)...")
    raw_response_7 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7)
    
    print(f"\n[Raw Response T=0.0]: {raw_response_0}")
    print(f"[Raw Response T=0.7]: {raw_response_7}")
    
    if raw_response_0 is None:
        print("Error: Did not get a valid response from the API.")
        continue
        
    # Structured Output Handling & Validation (On the T=0.0 response)
    print("\nValidating T=0.0 JSON Output...")
    clean_response = raw_response_0.strip() 
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
        print("Validation Status: PASS (Valid JSON matching schema)")
        print(f"Final Parsed Dict: {parsed_json}")
    except json.JSONDecodeError:
        print("Validation Status: FAIL (JSON Decode Error)")
        print(f"Fallback Applied: {fallback_json}")
    except jsonschema.ValidationError as e:
        print(f"Validation Status: FAIL (Schema Error - {e.message})")
        print(f"Fallback Applied: {fallback_json}")