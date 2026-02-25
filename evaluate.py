import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Load dataset
df = pd.read_csv("data.csv")
df = df.drop(columns=["id", "Unnamed: 32"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
X = df.drop(columns=["diagnosis"])
Y = df["diagnosis"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

#Load saved model
model = joblib.load("model.pkl")

#Predict
Y_pred = model.predict(X_test)
print("Evaluation Report:")
print(classification_report(Y_test, Y_pred))
