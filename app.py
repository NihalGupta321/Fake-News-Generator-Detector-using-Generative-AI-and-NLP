# app.py
import streamlit as st
import joblib
from detector import detector
from generator.generator import generate_fake_news

st.title("📰 Fake News Generator & Detector")

option = st.sidebar.selectbox("Choose Function", ["Detect News", "Generate Fake News"])

if option == "Detect News":
    text_input = st.text_area("Enter news text:")
    if st.button("Detect"):
        model = joblib.load("detector/model.pkl")
        vectorizer = joblib.load("detector/vectorizer.pkl")
        vec = vectorizer.transform([text_input])
        prediction = model.predict(vec)
        st.success(f"This news is: **{prediction[0]}**")

elif option == "Generate Fake News":
    prompt = st.text_input("Enter a news headline or idea:")
    if st.button("Generate"):
        result = generate_fake_news(prompt)
        st.write(result)
