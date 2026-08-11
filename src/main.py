import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)
data = data[[0, 5]]
data.columns = ["sentiment", "tweet"]

data["sentiment"] = data["sentiment"].replace(4, 1)

print("Twitter Sentiment Analysis Project")
print("\nDataset Preview:")
print(data.head())

tweet = input("\nEnter a tweet: ").lower()

positive_words = [
    "love", "amazing", "happy", "excellent",
    "awesome", "fantastic", "best"
]

negative_words = [
    "hate", "terrible", "sad",
    "worst", "bad", "disappointing"
]

positive_score = 0
negative_score = 0

for word in positive_words:
    if word in tweet:
        positive_score += 1

for word in negative_words:
    if word in tweet:
        negative_score += 1

print(f"\nPositive Score: {positive_score}")
print(f"Negative Score: {negative_score}")

if positive_score > negative_score:
    print("Sentiment: Positive 😊")
elif negative_score > positive_score:
    print("Sentiment: Negative ☹️")
else:
    print("Sentiment: Neutral 😐")
# Machine Learning Part

X = data["tweet"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()

model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nModel Accuracy:", accuracy * 100, "%")
import joblib

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel Saved Successfully!")