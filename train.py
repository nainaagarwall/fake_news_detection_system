print("TRAIN SCRIPT STARTED")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


import pickle


true = pd.read_csv("data/True.csv")
fake = pd.read_csv("data/Fake.csv")

true["label"] = 1
fake["label"] = 0

df = pd.concat([true, fake])
df = df.sample(frac=1).reset_index(drop=True)

print("RAW DATA SAMPLE:")
print(df["text"].head(2))


stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

print("\nCLEANED TEXT SAMPLE:")
print(df["clean_text"].head(2))



vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

print("\nTF-IDF Shape:", X.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)



model = LogisticRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
print("\nPrediction sample:", y_pred[:10])


acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


sns.countplot(x=y)
plt.title("Label Distribution")
plt.show()


df["length"] = df["clean_text"].apply(len)
sns.histplot(df["length"], bins=50)
plt.title("Text Length Distribution")
plt.show()


pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and Vectorizer Saved Successfully!")
