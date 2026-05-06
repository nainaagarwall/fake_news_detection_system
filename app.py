import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Enter a news article and check if it's Real or Fake")

user_input = st.text_area("Paste news text here", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")
