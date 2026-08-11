import joblib

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

while True:
    tweet = input("Enter Tweet: ")

    tweet_vec = vectorizer.transform([tweet])

    prediction = model.predict(tweet_vec)

    if prediction[0] == 1:
        print("Positive 😊")
    else:
        print("Negative 😞")