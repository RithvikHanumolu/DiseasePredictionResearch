import logging
import os
import random
import time
import uuid
from datetime import datetime

import gspread
import joblib
import numpy as np
import openai
import pandas as pd
import shap
import streamlit as st
from google.oauth2.service_account import Credentials
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Student Health Research Survey")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info("=== APP STARTUP BEGIN ===")

GOOGLE_SHEET_NAME = "Diabetes_Survey"
MAX_FOLLOWUPS = 5
OPENAI_MODEL = "gpt-4o"

QUESTION_TYPES = [
    "lifestyle_behavior",
    "medical_history",
    "emotional_wellbeing",
    "practical_barriers",
    "knowledge_awareness",
]

QUESTION_BANK = """
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

FEATURE_DESCRIPTIONS = {
    "HighBP": {
        "name": "High Blood Pressure",
        "values": {0: "No", 1: "Yes"},
        "context": "History of hypertension diagnosis",
    },
    "HighChol": {
        "name": "High Cholesterol",
        "values": {0: "No", 1: "Yes"},
        "context": "History of high cholesterol diagnosis",
    },
    "CholCheck": {
        "name": "Cholesterol Check (past 5 years)",
        "values": {0: "No", 1: "Yes"},
        "context": "Preventive health screening behavior",
    },
    "BMI": {
        "name": "Body Mass Index",
        "context": "Weight relative to height, key diabetes risk factor",
    },
    "Smoker": {
        "name": "Smoker (100+ cigarettes lifetime)",
        "values": {0: "No", 1: "Yes"},
        "context": "Smoking history affects cardiovascular and metabolic health",
    },
    "Stroke": {
        "name": "History of Stroke",
        "values": {0: "No", 1: "Yes"},
        "context": "Cardiovascular event history",
    },
    "HeartDiseaseorAttack": {
        "name": "Heart Disease or Heart Attack",
        "values": {0: "No", 1: "Yes"},
        "context": "Cardiovascular disease history, strongly linked to diabetes",
    },
    "PhysActivity": {
        "name": "Physical Activity (past 30 days)",
        "values": {0: "No", 1: "Yes"},
        "context": "Exercise habits outside of work",
    },
    "Fruits": {
        "name": "Daily Fruit Consumption",
        "values": {0: "No", 1: "Yes"},
        "context": "Dietary habits - fruit intake",
    },
    "Veggies": {
        "name": "Daily Vegetable Consumption",
        "values": {0: "No", 1: "Yes"},
        "context": "Dietary habits - vegetable intake",
    },
    "HvyAlcoholConsump": {
        "name": "Heavy Alcohol Consumption",
        "values": {0: "No", 1: "Yes"},
        "context": "Men >14 drinks/week, Women >7 drinks/week",
    },
    "AnyHealthcare": {
        "name": "Has Health Insurance",
        "values": {0: "No", 1: "Yes"},
        "context": "Access to healthcare services",
    },
    "NoDocbcCost": {
        "name": "Avoided Doctor Due to Cost",
        "values": {0: "No", 1: "Yes"},
        "context": "Financial barrier to healthcare access",
    },
    "GenHlth": {
        "name": "General Health Self-Rating",
        "values": {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"},
        "context": "Self-perceived overall health status",
    },
    "MentHlth": {
        "name": "Poor Mental Health Days (past 30 days)",
        "range": "0-30 days",
        "context": "Mental health affects lifestyle and self-care",
    },
    "PhysHlth": {
        "name": "Poor Physical Health Days (past 30 days)",
        "range": "0-30 days",
        "context": "Physical health limitations",
    },
    "DiffWalk": {
        "name": "Difficulty Walking/Climbing Stairs",
        "values": {0: "No", 1: "Yes"},
        "context": "Mobility limitations affecting exercise ability",
    },
    "Sex": {
        "name": "Biological Sex",
        "values": {0: "Female", 1: "Male"},
        "context": "Sex-based risk differences",
    },
    "Age": {
        "name": "Age Category",
        "values": {
            1: "18-24",
            2: "25-29",
            3: "30-34",
            4: "35-39",
            5: "40-44",
            6: "45-49",
            7: "50-54",
            8: "55-59",
            9: "60-64",
            10: "65-69",
            11: "70-74",
            12: "75-79",
            13: "80+",
        },
        "context": "Age is a major diabetes risk factor",
    },
    "Education": {
        "name": "Education Level",
        "values": {
            1: "Never attended",
            2: "Elementary",
            3: "Some high school",
            4: "High school graduate",
            5: "Some college",
            6: "College graduate",
        },
        "context": "Education level correlates with health literacy",
    },
    "Income": {
        "name": "Income Level",
        "values": {
            1: "<$10k",
            2: "$10-15k",
            3: "$15-20k",
            4: "$20-25k",
            5: "$25-35k",
            6: "$35-50k",
            7: "$50-75k",
            8: "$75k+",
        },
        "context": "Income affects access to healthy food and healthcare",
    },
}


def _get_secret_dict(key: str):
    value = st.secrets[key]
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json

        return json.loads(value)
    return dict(value)


def _format_feature_value(feature_name, value):
    desc = FEATURE_DESCRIPTIONS.get(feature_name, {})
    name = desc.get("name", feature_name)

    if feature_name == "BMI" and value is not None:
        if value < 18.5:
            category = "underweight"
        elif value < 25:
            category = "normal weight"
        elif value < 30:
            category = "overweight"
        else:
            category = "obese"
        return f"{name}: {value:.1f} ({category})"

    if "values" in desc and value is not None:
        if isinstance(value, np.generic):
            value = value.item()
        key = int(value) if isinstance(value, (int, float)) else value
        readable = desc["values"].get(key, str(value))
        return f"{name}: {readable}"

    if "range" in desc and value is not None:
        return f"{name}: {value} days"

    return f"{name}: {value}"


@st.cache_resource
def load_assets():
    start_time = time.time()
    logger.info("[CACHED] Loading assets from HuggingFace.")
    token = st.secrets["HUGGINGFACE_TOKEN"]
    model_file = hf_hub_download(
        repo_id="rmaster123/diabetes-model-assets",
        repo_type="dataset",
        filename="diabetes_model.pkl",
        token=token,
    )
    explainer_file = hf_hub_download(
        repo_id="rmaster123/diabetes-model-assets",
        repo_type="dataset",
        filename="shap_explainer.pkl",
        token=token,
    )
    features_file = hf_hub_download(
        repo_id="rmaster123/diabetes-model-assets",
        repo_type="dataset",
        filename="feature_names.pkl",
        token=token,
    )

    model = joblib.load(model_file)
    explainer = joblib.load(explainer_file)
    feature_names = joblib.load(features_file)

    try:
        _ = explainer.model.threshold_types
    except Exception:
        logger.info("Rebuilding SHAP TreeExplainer for compatibility.")
        explainer = shap.TreeExplainer(model)

    logger.info("[CACHED] Assets loaded in %.2fs", time.time() - start_time)
    return model, explainer, feature_names


client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@st.cache_resource
def get_google_sheet():
    logger.info("[CACHED] Initializing Google Sheets...")
    start_time = time.time()

    credentials_dict = _get_secret_dict("GOOGLE_SERVICE_ACCOUNT")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
    gspread_client = gspread.authorize(credentials)
    sheet = gspread_client.open(GOOGLE_SHEET_NAME).sheet1

    logger.info("[CACHED] Google Sheets ready in %.2fs", time.time() - start_time)
    return sheet


sheet = get_google_sheet()
model, explainer, feature_names = load_assets()
logger.info("=== MODEL LOADING COMPLETE ===")


def save_row(row: dict):
    headers = sheet.row_values(1)
    if not headers:
        headers = list(row.keys())
        sheet.append_row(headers)
    else:
        missing_headers = [key for key in row.keys() if key not in headers]
        if missing_headers:
            sheet.add_cols(len(missing_headers))
            for col_idx, header in enumerate(missing_headers, start=len(headers) + 1):
                sheet.update_cell(1, col_idx, header)
            headers.extend(missing_headers)

    values = [row.get(header, "") for header in headers]
    sheet.append_row(values)


def compute_model_context(user_features):
    x_model = pd.DataFrame(
        [{feature: user_features.get(feature, 0) for feature in feature_names}],
        columns=feature_names,
    ).fillna(0)

    try:
        pred_proba = model.predict_proba(x_model)[0, 1]
    except Exception:
        pred_proba = None

    shap_output = explainer(x_model)
    shap_vals = shap_output.values

    if shap_vals is None or len(shap_vals) == 0:
        raise ValueError("Internal model error. Please refresh.")

    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, 1, :]
    elif shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    if isinstance(shap_vals[0], (list, pd.Series, np.ndarray)):
        shap_vals = np.array(shap_vals).flatten()

    shap_dict = {feature: float(value) for feature, value in zip(feature_names, shap_vals)}
    shap_sorted = dict(sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
    return pred_proba, shap_sorted


def ensure_model_context():
    feature_signature = tuple((feature, st.session_state.features.get(feature)) for feature in feature_names)
    if st.session_state.get("model_context_signature") == feature_signature:
        return

    pred_proba, shap_sorted = compute_model_context(st.session_state.features)
    st.session_state.model_pred_proba = pred_proba
    st.session_state.model_shap_sorted = shap_sorted
    st.session_state.model_context_signature = feature_signature


def generate_llm_question(user_features, shap_dict, pred_prob=None):
    shap_summary = []
    for feature_name, shap_value in shap_dict.items():
        desc = FEATURE_DESCRIPTIONS.get(feature_name, {})
        magnitude = abs(shap_value)

        if magnitude > 0.1:
            impact = "HIGH IMPACT"
        elif magnitude > 0.05:
            impact = "MODERATE IMPACT"
        else:
            impact = "low impact"

        direction = "increases risk" if shap_value > 0 else "decreases risk"
        shap_summary.append(
            {
                "feature": feature_name,
                "user_value": _format_feature_value(feature_name, user_features.get(feature_name)),
                "context": desc.get("context", ""),
                "impact": f"{impact} - {direction}",
            }
        )

    previous_questions = st.session_state.get("questions", [])
    asked_types = st.session_state.get("asked_question_types", [])
    available_types = [question_type for question_type in QUESTION_TYPES if question_type not in asked_types]
    next_type = available_types[0] if available_types else QUESTION_TYPES[0]

    if pred_prob is None:
        risk_context = "The user's diabetes risk probability is unavailable. Focus on their strongest model features."
    elif pred_prob > 0.6:
        risk_context = (
            f"The user's predicted diabetes risk is {pred_prob:.0%} (HIGH RISK). "
            "Focus on actionable questions about immediate lifestyle changes, medical follow-up, and barriers to care."
        )
    elif pred_prob > 0.3:
        risk_context = (
            f"The user's predicted diabetes risk is {pred_prob:.0%} (MODERATE RISK). "
            "Focus on preventive habits and early warning signs."
        )
    else:
        risk_context = (
            f"The user's predicted diabetes risk is {pred_prob:.0%} (LOW RISK). "
            "Focus on maintaining healthy habits and understanding protective factors."
        )

    question_type_guidance = f"""
Generate a question of type: {next_type.upper().replace('_', ' ')}

Question type definitions:
- LIFESTYLE BEHAVIOR: Daily habits like diet, exercise, sleep, smoking, alcohol
- MEDICAL HISTORY: Health conditions, medications, doctor visits, family history of diabetes
- EMOTIONAL WELLBEING: Stress levels, depression, social support, coping mechanisms
- PRACTICAL BARRIERS: Obstacles to healthy living (time, cost, access to healthcare, knowledge)
- KNOWLEDGE AWARENESS: Understanding of diabetes risks, symptoms, prevention strategies
"""

    system_msg = (
        "You are a research survey assistant conducting a diabetes risk assessment. "
        "Output only the follow-up question text. "
        "Ask exactly one question, one or two sentences max. "
        "Make it specific and personalized based on the user's risk factors. "
        "Do not mention risk scores, models, or probabilities."
    )

    prompt = f"""
{risk_context}

{question_type_guidance}

TOP RISK FACTORS:
{chr(10).join(f"- {item['user_value']} -> {item['impact']}. Context: {item['context']}" for item in shap_summary[:5])}

Previous questions asked (do not repeat these topics):
{previous_questions if previous_questions else "None yet"}

Reference question bank for style (adapt to the user, do not copy verbatim):
{QUESTION_BANK}

Generate one personalized follow-up question that:
1. Relates to the user's top risk factors
2. Fits the question type `{next_type}`
3. Is specific, not generic
4. Does not mention model internals
"""

    text = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=100,
                timeout=30,
            )
            text = response.choices[0].message.content.strip()
            break
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            if attempt == 2:
                raise Exception("OpenAI API connection failed after 3 attempts.") from exc
            time.sleep(2**attempt)

    question = text.splitlines()[0].strip()
    if question.lower().startswith("question:"):
        question = question.split("question:", 1)[1].strip()
    if question.startswith('"') and question.endswith('"'):
        question = question[1:-1].strip()

    st.session_state.asked_question_types.append(next_type)
    return question


if "pid" not in st.session_state:
    st.session_state.pid = None
    st.session_state.step = 0
    st.session_state.features = {}
    st.session_state.questions = []
    st.session_state.group = None
    st.session_state.asked_question_types = []
    st.session_state.llm_calls = 0
    st.session_state.model_pred_proba = None
    st.session_state.model_shap_sorted = None
    st.session_state.model_context_signature = None
    st.session_state.submitted = False
    st.session_state.submitted_at = None


if st.session_state.submitted:
    st.title("Survey Complete")
    st.success("Thank you for participating.")
    st.info("This browser session has already submitted a response.")
    st.stop()


if st.session_state.pid is None:
    st.title("Student Research Study Consent")
    st.markdown(
        """
This research study evaluates how AI systems ask follow-up health questions.

• 5–10 minutes
• Anonymous
• No medical advice
• 18+ only
"""
    )

    is_adult = st.checkbox("I confirm I am 18 or older")
    agrees = st.checkbox("I agree to participate")

    if st.button("Start", disabled=not (is_adult and agrees)):
        st.session_state.pid = str(uuid.uuid4())
        st.session_state.group = "LLM"
        st.rerun()

    st.stop()


if st.session_state.step == 0:
    st.title("General Health Questions")
    st.session_state.features.update(
        {
            "HighBP": st.radio(
                "Have you ever been told by a doctor that you have high blood pressure?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "HighChol": st.radio(
                "Have you ever been told by a doctor that you have high cholesterol?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "CholCheck": st.radio(
                "Have you had your cholesterol checked in the past 5 years?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "BMI": st.number_input("What is your Body Mass Index (BMI)?", 10.0, 60.0, step=0.1),
            "Smoker": st.radio(
                "Have you smoked at least 100 cigarettes in your lifetime?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "Stroke": st.radio(
                "Have you ever had a stroke?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "HeartDiseaseorAttack": st.radio(
                "Have you ever had coronary heart disease or a heart attack?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "PhysActivity": st.radio(
                "Have you engaged in any physical activity in the past 30 days (not including your job)?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "Fruits": st.radio(
                "Do you eat fruit at least once per day?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "Veggies": st.radio(
                "Do you eat vegetables at least once per day?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "HvyAlcoholConsump": st.radio(
                "Do you drink heavily (men >14 drinks/week, women >7 drinks/week)?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "AnyHealthcare": st.radio(
                "Do you currently have health insurance?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "NoDocbcCost": st.radio(
                "In the past 12 months, was there a time you could not see a doctor because of cost?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "GenHlth": st.select_slider(
                "In general, how would you rate your health?",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x - 1],
            ),
            "MentHlth": st.slider(
                "During the past 30 days, how many days was your mental health not good?",
                0,
                30,
            ),
            "PhysHlth": st.slider(
                "During the past 30 days, how many days was your physical health not good?",
                0,
                30,
            ),
            "DiffWalk": st.radio(
                "Do you have serious difficulty walking or climbing stairs?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            ),
            "Sex": st.radio(
                "What is your sex?",
                [0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
            ),
            "Age": st.select_slider(
                "Which age range do you fall into?",
                options=list(range(1, 14)),
                format_func=lambda x: [
                    "18-24",
                    "25-29",
                    "30-34",
                    "35-39",
                    "40-44",
                    "45-49",
                    "50-54",
                    "55-59",
                    "60-64",
                    "65-69",
                    "70-74",
                    "75-79",
                    "80+",
                ][x - 1],
            ),
            "Education": st.select_slider(
                "What is your highest level of education?",
                options=list(range(1, 7)),
                format_func=lambda x: [
                    "Never attended school",
                    "Grades 1-8",
                    "Some high school",
                    "High school graduate",
                    "Some college",
                    "College graduate",
                ][x - 1],
            ),
            "Income": st.select_slider(
                "What is your total household income?",
                options=list(range(1, 9)),
                format_func=lambda x: [
                    "< $10k",
                    "$10–15k",
                    "$15–20k",
                    "$20–25k",
                    "$25–35k",
                    "$35–50k",
                    "$50–75k",
                    "$75k+",
                ][x - 1],
            ),
        }
    )

    if st.button("Continue"):
        ensure_model_context()
        st.session_state.step = 1
        st.rerun()

elif 1 <= st.session_state.step <= MAX_FOLLOWUPS:
    st.title("Follow-Up Question")

    try:
        ensure_model_context()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    pred_proba = st.session_state.model_pred_proba
    shap_sorted = st.session_state.model_shap_sorted

    question_key = f"question_{st.session_state.step}"
    if question_key not in st.session_state:
        with st.spinner("Generating personalized question..."):
            prompt_features = {
                key: value for key, value in st.session_state.features.items() if key in feature_names
            }
            st.session_state[question_key] = generate_llm_question(
                prompt_features,
                shap_sorted,
                pred_prob=pred_proba,
            )

    question = st.session_state[question_key]
    with st.form(key=f"followup_form_{st.session_state.step}"):
        answer = st.text_input(question)
        submitted = st.form_submit_button("Next")

    if submitted:
        st.session_state.features[f"QuestionText{st.session_state.step}"] = question
        st.session_state.features[f"Q{st.session_state.step}"] = answer
        st.session_state.questions.append(question)
        st.session_state.llm_calls += 1
        st.session_state.step += 1
        st.rerun()

else:
    st.title("Final Question")
    label = st.selectbox(
        "Have you ever been diagnosed with diabetes by a medical professional?",
        ["Yes", "No", "Prefer not to say"],
    )

    if st.button("Submit", disabled=st.session_state.submitted):
        row = {
            "participant_id": st.session_state.pid,
            "group": st.session_state.group,
            "label": label,
            "timestamp": datetime.utcnow().isoformat(),
        }
        row.update(st.session_state.features)
        save_row(row)
        st.session_state.submitted = True
        st.session_state.submitted_at = row["timestamp"]
        st.rerun()
