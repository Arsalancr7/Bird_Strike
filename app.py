import streamlit as st
import joblib
import pandas as pd
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS

import os

# Load ML models
clf_model = joblib.load("models/rf_damage_predictor.pkl")
reg_model = joblib.load("models/gbr_repair_cost_model.pkl")

# Load FAISS vectorstore 
try:
    vectorstore = FAISS.load_local("vectorstore", embedding=None, allow_dangerous_deserialization=True)
except:
    vectorstore = None

st.title("FAA Bird Strike Incident Predictor")

# API Key
openai_key = st.text_input("Enter your OpenAI API Key", type="password")

st.header("Incident Details")
# Classification features
classification_features = {}
classification_features["INCIDENT_YEAR"] = st.number_input("Incident Year", value=2024)
classification_features["TIME_OF_DAY"] = st.selectbox("Time of Day", ["DAY", "NIGHT"])
classification_features["PHASE_OF_FLIGHT"] = st.selectbox("Phase of Flight", ["TAKEOFF", "LANDING"])
classification_features["STATE"] = st.text_input("State", "NY")
classification_features["SKY"] = st.selectbox("Sky Condition", ["CLEAR", "CLOUDY"])
classification_features["PRECIPITATION"] = st.selectbox("Precipitation", ["NONE", "RAIN", "SNOW"])
classification_features["NUM_ENGS"] = st.number_input("Number of Engines", 1, 4)
classification_features["TYPE_ENG"] = st.selectbox("Engine Type", ["JET", "TURBOPROP"])
classification_features["AC_MASS"] = st.number_input("Aircraft Mass", 1, 10)
classification_features["SPEED"] = st.number_input("Speed (knots)", 0)
classification_features["HEIGHT"] = st.number_input("Height (feet)", 0)
classification_features["DISTANCE"] = st.number_input("Distance from Airport (nm)", 0)
classification_features["SIZE"] = st.selectbox("Bird Size", ["SMALL", "MEDIUM", "LARGE"])
classification_features["AOS"] = st.number_input("AOS", 0)

# Regression features
regression_features = {
    "AC_MASS": classification_features["AC_MASS"],
    "NUM_ENGS": classification_features["NUM_ENGS"],
    "INGESTED_OTHER": 1,
    "DAM_ENG1": 1,
    "DAM_ENG2": 0,
    "ENG_1_POS": 1,
    "STR_ENG2": 0,
    "STR_ENG1": 1,
    "REMAINS_SENT": 1,
    "REMAINS_COLLECTED": 1,
    "EMA": 20
}

if st.button("Predict and Explain"):
    if not openai_key:
        st.error("Please provide your OpenAI API Key.")
    else:
        # Classification prediction
        X_clf = pd.DataFrame([classification_features])
        damage_pred = clf_model.predict(X_clf)[0]
        damage_label = "Damage Occurred" if damage_pred == 1 else "No Damage"

        # Regression prediction
        X_reg = pd.DataFrame([regression_features])
        predicted_cost_log = reg_model.predict(X_reg)[0]
        predicted_cost = np.expm1(predicted_cost_log)

        # Retrieve historical incidents
        retrieved_records = "No historical data available."
        if vectorstore:
            retrieved_docs = vectorstore.similarity_search("bird strike incident", k=3)
            retrieved_records = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Prompt for LLM
        explanation_prompt = PromptTemplate(
            input_variables=["features", "damage_label", "predicted_cost", "retrieved_records"],
            template="""
            You are an aviation safety analyst.

            Incident Features:
            {features}

            Model Predictions:
            - Damage Assessment: {damage_label}
            - Predicted Repair Cost: ${predicted_cost:,.2f}

            Historical Similar Incidents:
            {retrieved_records}

            Explain why this outcome is likely, referencing both the classification and cost predictions,
            and compare it to historical patterns.
            """
        )

        # LLM Chain
        os.environ["OPENAI_API_KEY"] = openai_key
        llm = ChatOpenAI(openai_api_key=openai_key, temperature=0)
        chain = LLMChain(llm=llm, prompt=explanation_prompt)
        explanation = chain.run(
            features={**classification_features, **regression_features},
            damage_label=damage_label,
            predicted_cost=predicted_cost,
            retrieved_records=retrieved_records
        )

        # Output
        st.subheader("Prediction Results")
        st.write(f"**Damage Prediction:** {damage_label}")
        st.write(f"**Estimated Repair Cost:** ${predicted_cost:,.2f}")
        st.subheader("Explanation")
        st.write(explanation)

