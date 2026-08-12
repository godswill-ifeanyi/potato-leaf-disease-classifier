import google.generativeai as genai
import streamlit as st

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-3.5-flash")


def get_recommendation(disease):

    prompt = f"""
            You are an agricultural expert.

            The detected potato disease is:

            {disease}

            Provide:

            1. Disease description
            2. Possible causes
            3. Symptoms
            4. Treatment
            5. Prevention
            6. Farmer-friendly advice

            Keep it under 250 words.
            """

    response = model.generate_content(prompt)

    return response.text