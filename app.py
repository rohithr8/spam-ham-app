import streamlit as st
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = pickle.load(open("logistic_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("📧 Spam or Ham Email Classifier")
st.write("Enter your email message below to check if it's spam or not.")

# Text input
user_input = st.text_area("Email content", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Vectorize the input
        input_features = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_features)

        # Display result
        if prediction[0] == 1:
            st.success("✅ email is classified as **ham ( safe)**.")
        else:
            st.error(" 🚫This email is classified as **Hspam**.")
