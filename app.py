import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("language_model.pkl")
cv = joblib.load("cv.pkl")

# Add logo/banner
st.image("logo.jpg", width=200)

# Title and description
st.title(" Language Detection App")
st.write("A clean and simple app to detect the language of any text you enter.")


# Load model and vectorizer
model = joblib.load("language_model.pkl")
cv = joblib.load("cv.pkl")

# Streamlit UI
st.title(" Language Detection App")
st.write("Enter text below and I will predict the language.")

user_input = st.text_area("Enter text:")

if st.button("Detect Language"):
    if user_input.strip():
        data = cv.transform([user_input]).toarray()
        output = model.predict(data)[0]
        st.success(f"Detected language: {output}")
    else:
        st.warning("Please enter some text.")

