import streamlit as st
import joblib
import requests
import os
import pandas as pd

# -------------------------------------------------
# Load model
# -------------------------------------------------
model = joblib.load("disease_model.joblib")

# -------------------------------------------------
# LOAD FEATURES FROM DATASET (FINAL FIX 🔥)
# -------------------------------------------------
df = pd.read_csv("Training.csv")

# remove unwanted columns
ALL_SYMPTOMS = list(df.columns)
ALL_SYMPTOMS.remove("prognosis")

if "Unnamed: 133" in ALL_SYMPTOMS:
    ALL_SYMPTOMS.remove("Unnamed: 133")

# -------------------------------------------------
# Preprocess input
# -------------------------------------------------
def preprocess_symptoms(user_input):
    user_symptoms = [s.strip().lower() for s in user_input.split(",")]

    vector = []
    for symptom in ALL_SYMPTOMS:
        if symptom.strip().lower() in user_symptoms:
            vector.append(1)
        else:
            vector.append(0)

    return vector

# -------------------------------------------------
# API key
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

        if len(input_data) != model.n_features_in_:
            st.error(f"Mismatch: Model expects {model.n_features_in_}, got {len(input_data)}")
        else:
            prediction = model.predict([input_data])[0]

            st.success(f"🧠 Predicted Disease: {prediction}")

            st.subheader("📘 Basic Info")
            st.write(f"""
            - Possible condition: **{prediction}**
            - Based on your symptoms
            - For awareness only
            """)

            st.subheader("🖼️ Images")
            images = fetch_images(prediction)

            for img in images:
                st.image(img, width=250)
