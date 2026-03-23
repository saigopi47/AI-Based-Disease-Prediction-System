import streamlit as st
import joblib
import requests
import os

# -------------------------------------------------
# Load model
# -------------------------------------------------
model = joblib.load("disease_model.joblib")

# -------------------------------------------------
# EXACT 132 symptoms (FIXED)
# -------------------------------------------------
ALL_SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering',
    'chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue',
    'muscle_wasting','vomiting','burning_micturition','spotting_urination',
    'fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings',
    'weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness',
    'sweating','dehydration','indigestion','headache','yellowish_skin',
    'dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain',
    'constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
    'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose',
    'congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising',
    'obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
    'enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger',
    'extra_marital_contacts','drying_and_tingling_lips','slurred_speech',
    'knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort',
    'foul_smell_of_urine','continuous_feel_of_urine','passage_of_gases',
    'internal_itching','toxic_look_(typhos)','depression','irritability',
    'muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic_patches','watering_from_eyes',
    'increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma',
    'stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'blood_in_sputum','prominent_veins_on_calf','palpitations',
    'painful_walking','pus_filled_pimples','blackheads','scurring',
    'skin_peeling','silver_like_dusting','small_dents_in_nails',
    'inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]

# -------------------------------------------------
# Preprocess input
# -------------------------------------------------
def preprocess_symptoms(user_input):
    user_symptoms = [s.strip().lower() for s in user_input.split(",")]
    return [1 if symptom in user_symptoms else 0 for symptom in ALL_SYMPTOMS]

# -------------------------------------------------
# SerpAPI key (secure)
# -------------------------------------------------
SERP_API_KEY = os.getenv("SERPAPI_KEY")

def fetch_images(query):
    url = "https://serpapi.com/search.json"
    params = {"q": query, "tbm": "isch", "api_key": SERP_API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    return [img["original"] for img in data.get("images_results", [])[:3]]

# -------------------------------------------------
# UI
# -------------------------------------------------
st.set_page_config(page_title="Disease Prediction AI")

st.title("🩺 AI-Based Disease Prediction System")

symptoms = st.text_area(
    "Enter symptoms (comma separated)",
    placeholder="itching, skin_rash"
)

if st.button("Predict Disease"):
    if symptoms.strip() == "":
        st.warning("Please enter symptoms")
    else:
        input_data = preprocess_symptoms(symptoms)

        # Safety check
        if len(input_data) != model.n_features_in_:
            st.error("Feature mismatch error. Check symptoms list.")
        else:
            prediction = model.predict([input_data])[0]

            st.success(f"🧠 Predicted Disease: {prediction}")

            st.subheader("📘 Basic Info")
            st.write(f"""
            - Based on your symptoms, this condition may be **{prediction}**
            - This is only for awareness
            - Please consult a doctor
            """)

            st.subheader("🖼️ Related Images")
            images = fetch_images(prediction)

            for img in images:
                st.image(img, width=250)
