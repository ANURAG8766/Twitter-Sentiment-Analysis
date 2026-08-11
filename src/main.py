import streamlit as st
import joblib

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("Twitter Sentiment Analysis")

tweet = st.text_area("Enter a tweet")

if st.button("Analyze Sentiment"):

    tweet_vec = vectorizer.transform([tweet])

    prediction = model.predict(tweet_vec)[0]

    if prediction == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative ☹️")