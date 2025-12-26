import streamlit as st
import uuid
import random
import joblib
import pandas as pd
import shap
import csv
from datetime import datetime
import openai
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import os
import joblib
import requests

GOOGLE_SHEET_NAME = "Diabetes_Survey"  # your Google Sheet title


# OpenAI
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load service account credentials from Streamlit secrets
credentials_dict = st.secrets["GOOGLE_SERVICE_ACCOUNT"]
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
gc = gspread.authorize(credentials)
sheet = gc.open(GOOGLE_SHEET_NAME).sheet1


# =========================
# GOOGLE SHEETS SETUP
# =========================
# Make sure your service account JSON is stored in Streamlit Secrets as explained
# st.secrets["google"]["service_account"]

# creds_dict = json.loads(st.secrets["google"]["service_account"])
# creds = Credentials.from_service_account_info(creds_dict)
# client = gspread.authorize(creds)

# Replace "YourSheetName" with the name of your Google Sheet

# Utility to save a row to Google Sheets
def save_row(row: dict):
    # Get headers from first row of the sheet
    headers = sheet.row_values(1)
    
    # If sheet is empty, add headers
    if not headers:
        sheet.append_row(list(row.keys()))
    
    # Append the values in the same order as headers
    values = [row.get(h, "") for h in sheet.row_values(1)]
    if not values:  # If first row just added
        values = list(row.values())
    
    sheet.append_row(list(row.values()))

# =========================
# CONFIG
# =========================
MAX_FOLLOWUPS = 5
OPENAI_MODEL = "gpt-4o-mini"
# DATA_FILE = "responses.csv"

# =========================
# LOAD MODEL + SHAP
# =========================

MODEL_URL = "https://drive.google.com/file/d/17fPkCJ8YTysXAjIEXo8EqGoAzOJDzx1Q/view?usp=sharing"
EXPLAINER_URL = "https://drive.google.com/file/d/1EGT2L2FToTnGX7k04tSG5NNf8JeoFFSn/view?usp=sharing"
FEATURES_URL = "https://drive.google.com/file/d/1rHjMcx9HzGANQTf2jyXBjfO9Y46lCPqC/view?usp=sharing"

def download_file(url, local_path):
    if not os.path.exists(local_path):
        r = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(r.content)

download_file(MODEL_URL, "diabetes_model.pkl")
download_file(EXPLAINER_URL, "shap_explainer.pkl")
download_file(FEATURES_URL, "feature_names.pkl")

model = joblib.load("diabetes_model.pkl")
explainer = joblib.load("shap_explainer.pkl")
feature_names = joblib.load("feature_names.pkl")


if "llm_calls" not in st.session_state:
    st.session_state.llm_calls = 0

if st.session_state.llm_calls >= MAX_FOLLOWUPS:
    st.warning("LLM question limit reached.")
    st.stop()


# =========================
# OLLAMA CLIENT
# =========================

# =========================
# UTILS
# =========================
# def save_row(row):
#     file_exists = False
#     try:
#         file_exists = open(DATA_FILE).readline()
#     except:
#         pass

#     with open(DATA_FILE, "a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=row.keys())
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(row)

def generate_llm_question(user_features, shap_dict, pred_prob=None):
    # Prepare context
    risk_val = f"{pred_prob:.1%}" if pred_prob else "N/A"
    
    # This is your original list of medical questions shortened for the prompt
    medical_context = """
    Do you test your blood sugar levels? If no, why?


How many times per day do you test? One, Two, Three, Four, Five or more


When do you test? (Before breakfast, before lunch/dinner, after meals, at bedtime, other)


Have you had LOW blood sugars? If yes, how often? (Daily, Weekly, Monthly, Other)


What time(s) of day do most of your low blood sugars occur? (Morning, Mid Day, Afternoon, Evening, Night)


How do you treat low blood sugars?


Have you ever lost consciousness or required assistance to reverse low blood sugar?


When did it last occur? How often?


Do you ever have HIGH blood sugar levels? If yes, how often? (Daily, Weekly, Monthly, Other)


What time(s) of day do most of your high blood sugars occur? (Morning, Mid Day, Afternoon, Evening, Night)


How do you treat high blood sugars?


What is your sex?


Is there a family history of diabetes?


What is your Race / Ethnicity?


Do you have any other health problems? (High BP, Heart disease, Cholesterol, etc.)


Do you have any of the following problems? (Vision / Hearing issues – use glasses? use hearing aids?)

   
List any medications and when you take them


How often do you see your doctor?


When did you last see your eye doctor?


Do you live alone? If not, who do you live with?


Do you smoke? If quit, when? If yes, how much?


Do you drink alcohol? (Type, how much, how often)


Do you work? If yes: What shift? What hours?


Is there much stress in your life? How do you handle it?


Do you ever get depressed? (A lot / Some / A little)


Do you exercise? If yes: Type of exercise; Frequency; Length


Do you have limitations on exercise?


Have you had previous instruction on diet? If yes: Where? When?


Do you have a meal plan? Calories? How much do you follow it? (0–100%)


Do you follow dietary restrictions or special meals? (Vegetarian, Low-carb, etc.)


Has your weight changed in the last 6 months? (Pounds gained/lost)


What is your Height, Age, Current weight?


Are you happy with your weight? What would you like to weigh?


What was your highest weight? If current is less, how did you lose weight?


Do you have any food allergies?


Do you have any food/beverage intolerances?


How is your appetite? (Good / Fair / Poor)


Any eating/digestion problems? (Chewing, Swallowing, Stomachache, etc.)


Who prepares meals at home?


Who does the grocery shopping?


Do you follow any cultural/religious dietary restrictions?


Do you take vitamins or nutrition supplements? (Multivitamins, Iron, etc.)


Has there been any recent change in your appetite?


Do you take herbal supplements? (Garlic, Ginseng, etc.)


Check favorite beverages and amount: Coffee / Tea (cups/day); What do you add? (Milk, Sugar, etc.); Juice, Soda, Water (amount)


What do you eat in a typical day? (Time + meals/snacks, content, quantity)


How often do you eat listed foods? (Bread, Sausage, Pasta, Candy, etc.) Frequency: Daily, 1–3x/week, 4+ x/week, Monthly, Rarely; Quantity


How many times/week do you eat: Breakfast, Lunch, Dinner


What dairy products do you eat/drink? (Milk, Yogurt, Cheese – fat content)


Milk intake: How many cups per day?


If using: Lactaid / Soy milk / Rice milk — how much?


What fruits do you like? How often? (Canned in syrup/juice, Fresh, Frozen, etc.)


What vegetables do you like? How often? (Fresh, Canned, Frozen)


Foods you dislike and will not eat


How often do you eat out (restaurants, cafeterias, etc.)?


What eating concerns do you have?
    """

    prompt = f"""
    You are a health assistant helping users understand their diabetes risk.
    Current Model Risk Probability: {risk_val}

    User Data:
    {chr(10).join(f"- {k}: {v}" for k, v in user_features.items() if k in feature_names)}

    SHAP Feature Impact:
    {chr(10).join(f"- {k}: {'↑' if v > 0 else '↓'} {abs(v):.3f}" for k, v in shap_dict.items())}

    INSTRUCTION:
    Ask ONE (1) clear, natural follow-up question. 
    Focus on the features with the biggest SHAP impact. 
    Use the following medical questions as a guide for style and topic:
    {medical_context}
    
    DO NOT provide a list. Ask only ONE question.
    """
    
    response = client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)
    return response.choices[0].message.content.strip()

# =========================
# STREAMLIT STATE
# =========================
st.set_page_config(page_title="Health Research Survey")

if "pid" not in st.session_state:
    st.session_state.pid = None
    st.session_state.step = 0
    st.session_state.features = {}
    st.session_state.questions = []
    st.session_state.group = None

# =========================
# CONSENT
# =========================
if st.session_state.pid is None:
    st.title("Research Study Consent")

    st.markdown("""
This research study evaluates how AI systems ask follow-up health questions.

• 5–10 minutes  
• Anonymous  
• No medical advice  
• 18+ only
""")

    if st.checkbox("I confirm I am 18 or older") and st.checkbox("I agree to participate"):
        if st.button("Start"):
            st.session_state.pid = str(uuid.uuid4())
            st.session_state.group = random.choice(["LLM", "RANDOM"])
            st.rerun()

    st.stop()

# =========================
# BASELINE QUESTIONS
# =========================
if st.session_state.step == 0:
    st.title("General Health Questions")

    # ================= Baseline Quiz =================
    st.session_state.features.update({
        "HighBP": st.radio("Have you ever been told by a doctor that you have high blood pressure?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "HighChol": st.radio("Have you ever been told by a doctor that you have high cholesterol?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "CholCheck": st.radio("Have you had your cholesterol checked in the past 5 years?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "BMI": st.number_input("What is your Body Mass Index (BMI)?", 10.0, 60.0, step=0.1),
        "Smoker": st.radio("Have you smoked at least 100 cigarettes in your lifetime?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "Stroke": st.radio("Have you ever had a stroke?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "HeartDiseaseorAttack": st.radio("Have you ever had coronary heart disease or a heart attack?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "PhysActivity": st.radio("Have you engaged in any physical activity in the past 30 days (not including your job)?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "Fruits": st.radio("Do you eat fruit at least once per day?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "Veggies": st.radio("Do you eat vegetables at least once per day?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "HvyAlcoholConsump": st.radio("Do you drink heavily (men >14 drinks/week, women >7 drinks/week)?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "AnyHealthcare": st.radio("Do you currently have health insurance?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "NoDocbcCost": st.radio("In the past 12 months, was there a time you could not see a doctor because of cost?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "GenHlth": st.select_slider("In general, how would you rate your health?", options=[1, 2, 3, 4, 5], format_func=lambda x: ["Excellent","Very Good","Good","Fair","Poor"][x-1]),
        "MentHlth": st.slider("During the past 30 days, how many days was your mental health not good?", 0, 30),
        "PhysHlth": st.slider("During the past 30 days, how many days was your physical health not good?", 0, 30),
        "DiffWalk": st.radio("Do you have serious difficulty walking or climbing stairs?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        "Sex": st.radio("What is your sex?", [0, 1], format_func=lambda x: "Female" if x==0 else "Male"),
        "Age": st.select_slider("Which age range do you fall into?", options=list(range(1,14)), format_func=lambda x: ["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"][x-1]),
        "Education": st.select_slider("What is your highest level of education?", options=list(range(1,7)), format_func=lambda x: ["Never attended school","Grades 1-8","Some high school","High school graduate","Some college","College graduate"][x-1]),
        "Income": st.select_slider("What is your total household income?", options=list(range(1,9)), format_func=lambda x: ["< $10k","$10–15k","$15–20k","$20–25k","$25–35k","$35–50k","$50–75k","$75k+"][x-1])
    })

    if st.button("Continue"):
        st.session_state.step = 1
        st.rerun()


# # =========================
# # FOLLOW-UP LOOP
# # =========================
# elif 1 <= st.session_state.step <= MAX_FOLLOWUPS:
#     st.title("Follow-Up Question")

#     # =========================
#     # Prepare input for the model
#     # =========================
#     # Ensure features match the model's expected columns and types
#     X_model = pd.DataFrame([st.session_state.features], columns=feature_names).fillna(0)

#     # =========================
#     # Get model prediction
#     # =========================
#     # Probability for diabetes (adjust index depending on your encoding: 1 = diabetes)
#     try:
#         pred_proba = model.predict_proba(X_model)[0, 1]
#     except:
#         pred_proba = None  # fallback if predict_proba fails

#     # =========================
#     # Compute SHAP values
#     # =========================
#     shap_output = explainer(X_model)
#     shap_vals = shap_output.values

#     # Handle multiclass vs. single-class outputs
#     if shap_vals.ndim == 3:  # shape: (num_samples, num_classes, num_features)
#         shap_vals = shap_vals[:, 1, :]  # take class “1” (diabetes)
#     elif shap_vals.ndim == 2:  # shape: (num_samples, num_features)
#         shap_vals = shap_vals[0]  # take first (and only) sample

#     # Flatten any remaining nested structures
#     if isinstance(shap_vals[0], (list, pd.Series, np.ndarray)):
#         shap_vals = np.array(shap_vals).flatten()

#     # Map feature names to SHAP values
#     shap_dict = {f: float(v) for f, v in zip(feature_names, shap_vals)}

#     # Sort features by absolute impact (top 5)
#     shap_sorted = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

# =========================
# FOLLOW-UP LOOP
# =========================
elif 1 <= st.session_state.step <= MAX_FOLLOWUPS:
    st.title("Follow-Up Question")

    # =========================
    # Prepare input for the model
    # =========================
    X_model = pd.DataFrame([st.session_state.features], columns=feature_names).fillna(0)

    # =========================
    # Get model prediction
    # =========================
    try:
        pred_proba = model.predict_proba(X_model)[0, 1]  # probability for diabetes
        risk_val = f"{pred_proba:.1%}"
    except:
        pred_proba = None
        risk_val = "N/A"

    # =========================
    # Compute SHAP values
    # =========================

    
    shap_output = explainer(X_model)
    shap_vals = shap_output.values

    if shap_vals is None or len(shap_vals) == 0:
        st.error("Internal model error. Please refresh.")
        st.stop()

    # Handle multiclass vs single-class outputs
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, 1, :]  # class “1” (diabetes)
    elif shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    if isinstance(shap_vals[0], (list, pd.Series, np.ndarray)):
        shap_vals = np.array(shap_vals).flatten()

    shap_dict = {f: float(v) for f, v in zip(feature_names, shap_vals)}
    shap_sorted = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

    # =========================
    # Generate or retrieve question (cached)
    # =========================
    question_key = f"question_{st.session_state.step}"

    if question_key not in st.session_state:
        with st.spinner("Generating personalized question..."):
            if st.session_state.group == "LLM":
                # Add SHAP impact context to prompt
                shap_context = "\n".join(
                    f"- {k}: {'high' if abs(v) > 0.05 else 'low'} impact" 
                    for k, v in shap_sorted.items()
                )
                prompt_features = {k: v for k, v in st.session_state.features.items() if k in feature_names}

                st.session_state[question_key] = generate_llm_question(
                    prompt_features,
                    shap_sorted,
                    pred_prob=pred_proba
                )
            else:
                # Random question for control group
                st.session_state[question_key] = random.choice([
                    "How often do you exercise?",
                    "How would you describe your diet?",
                    "How many hours do you sleep?",
                    "Do you experience stress regularly?"
                ])

    question = st.session_state[question_key]

    # =========================
    # Display question + input
    # =========================
    answer = st.text_input(question)

    if st.button("Next"):
    # Save answer and question
        st.session_state.features[f"Q{st.session_state.step}"] = answer
        st.session_state.questions.append(question)
    
    # ✅ Increment LLM calls only for LLM group
        if st.session_state.group == "LLM":
            st.session_state.llm_calls += 1

    # Move to next step
        st.session_state.step += 1
        st.rerun()



# =========================
# FINAL LABEL + SAVE
# =========================
else:
    st.title("Final Question")

    # Ask final label
    label = st.selectbox(
        "Have you ever been diagnosed with diabetes by a medical professional?",
        ["Yes", "No", "Prefer not to say"]
    )

    if st.button("Submit"):
        # Build row to save
        row = {
            "participant_id": st.session_state.pid,
            "group": st.session_state.group,
            "label": label,
            "timestamp": datetime.utcnow().isoformat()
        }
        row.update(st.session_state.features)

        # Save function for Google Sheets
        def save_row_to_gs(row: dict):
            # Get existing headers
            headers = sheet.row_values(1)

            # If sheet is empty, add headers first
            if not headers:
                headers = list(row.keys())
                sheet.append_row(headers)

            # Append the row values in the same order as headers
            values = [row.get(h, "") for h in headers]
            sheet.append_row(values)


        # Save row
        save_row_to_gs(row)

        st.success("Thank you for participating.")

