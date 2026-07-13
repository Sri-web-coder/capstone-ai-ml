# Part 4: LLM-Powered Feature — Model Prediction Explanation

## Track Selection

**I have chosen Track C: Model Prediction Explanation Pipeline.**

## Project Summary

In this final part of the project, I integrated a Large Language Model (LLM) to explain the predictions made by the machine learning model from Part 3. Instead of a Jupyter Notebook, this part is structured into three Python scripts to simulate a real-world application:

* **`llm_structured.py`**: This is the core and most important file. It contains the main pipeline for formatting inputs, communicating with the LLM API, enforcing JSON schemas, and running the A/B temperature comparisons. 
* **`guardrails.py`**: Contains the regex logic to detect and block Personally Identifiable Information (PII).
* **`app.py`**: A Streamlit web application that provides a user-friendly interface for the ML model and the AI analyzer.

## Setup & Run Commands

To run the code, you need to execute the Python scripts directly from your terminal/command prompt.

* **Environment Setup:** Create a `.env` file in the same directory and add your API key:
  `LLM_API_KEY="your_api_key_here"`
  *(Never hardcode the API key in the scripts!)*

* **Run the Core LLM Pipeline:**
  Open your terminal and run:
  `python llm_structured.py`
  *(This will execute the API tests, PII guardrail checks, and the end-to-end evaluation pipeline.)*

* **Run the Streamlit Web UI:**
  Open your terminal and run:
  `streamlit run app.py`
  *(This will launch the web application in your browser.)*

## Prompts Design

### System Prompt

> You are a machine learning interpretation assistant. 
> Your task is to explain model predictions based on feature values, predicted class, and probability.
> Output ONLY valid JSON matching this exact schema:
> {
> "prediction_label": "string (e.g., At Risk, Not At Risk)",
> "confidence_level": "string (low, medium, high)",
> "top_reason": "string (main reason for prediction based on features)",
> "second_reason": "string (secondary reason)",
> "next_step": "string (recommended action)"
> }
> Do not include any markdown tags like ```json or any other text. Return pure JSON.

### User Prompt Template

> Features: {features} | Predicted Class: {readable_class} | Probability: {pred_proba:.2%}

## Temperature Settings & A/B Comparison

I used `temperature=0.0` for the main pipeline. A temperature of 0 produces deterministic and predictable outputs because the model always picks the highest-probability next token. This is absolutely critical for structured data tasks like JSON extraction, where we need the exact schema every time. A temperature of 0.7 introduces variability by sampling from a broader distribution, which can lead to creative but unpredictable formatting that breaks JSON parsing.

### Temperature A/B Comparison Table

| Input (Feature Summary) | Output at temp=0.0 (Top Reason) | Output at temp=0.7 (Top Reason) | Key Difference |
| :--- | :--- | :--- | :--- |
| **Record 1:** (Age 22, Study: 10h, Pred: Not At Risk) | "The individual has a part-time job, which may provide structure and financial support." | "The model indicates a lower risk due to a significant amount of study hours despite financial stress." | Temp 0 focused strictly on the part-time job factor, while Temp 0.7 generated a slightly more creative (but different) narrative focusing on study hours vs. stress. |
| **Record 2:** (Age 20, Study: 25h, Pred: At Risk) | "High attendance rate and significant study hours indicate commitment, but low physical activity and financial stress may contribute to risk." | "High attendance rate of 95% indicates commitment, but financial stress and low physical activity may contribute to risk." | Very similar, but Temp 0.7 explicitly pulled the exact 95% number into the string. |
| **Record 3:** (Age 21, Study: 15h, Pred: At Risk) | "High financial stress and low physical activity levels contribute significantly to the risk." | "High financial stress level at 5, which can negatively impact mental health." | Temp 0 generalized the factors, whereas Temp 0.7 started listing out the exact score values ("level at 5"). |

## Structured Output Handling & Validation

After receiving the response from the LLM, the code strips any whitespace or stray markdown tags. It then parses the string using `json.loads()`. I defined a strict JSON schema with 5 required fields. The parsed dictionary is validated using `jsonschema.validate()`. If it fails (caught by a `ValidationError` or `JSONDecodeError`), the code catches the error, logs it, and returns a fallback dictionary where all 5 required fields are set to `None`. 

## Guardrails (PII Detection)
Before sending any user input to the LLM, it passes through `has_pii()` in `guardrails.py`. This uses regex to search for emails and phone numbers.

* **Clean Input:** `Features: {'Age': 21, 'GPA': 3.5} | Predicted Class: 0` -> **Passed**
* **Blocked Input:** `Student durga@email.com has Features...` -> **Blocked (PII Detected)**

## End-to-End Demonstration

| Feature Input (Truncated) | Predicted Class & Prob | Explanation JSON (Validation Status) | Pass/Block (Guardrail) |
| :--- | :--- | :--- | :--- |
| `{'Age': 22, 'Study_Hours': 10, 'Financial_Stress': 9...}` | Not At Risk (39.20%) | **PASS:** Valid JSON matching schema. | Passed |
| `{'Age': 20, 'Study_Hours': 25, 'Financial_Stress': 2...}` | At Risk (79.36%) | **PASS:** Valid JSON matching schema. | Passed |
| `{'Age': 21, 'Study_Hours': 15, 'Financial_Stress': 5...}` | At Risk (51.78%) | **PASS:** Valid JSON matching schema. | Passed |
